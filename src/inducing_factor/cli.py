import argparse
import logging
import os

import numpy as np
import pandas as pd

from inducing_factor import AnnualProbabilityMapper, EcoDisturbanceAttributionSystem, PipelineProcessor
from inducing_factor.config import load_config
from inducing_factor.core.class_conditional_analysis import FactorClassAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inducing-factor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config")

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument("--config")

    predict_parser = subparsers.add_parser("predict-annual-map")
    predict_parser.add_argument("--config")
    predict_parser.add_argument("--year", type=int, required=True)
    predict_parser.add_argument("--output-dir")

    analyze_classes_parser = subparsers.add_parser("analyze-classes")
    analyze_classes_parser.add_argument("--config", required=True)

    return parser


def _run_analyze_classes(args):
    import glob

    log = logging.getLogger("InducingFactorSystem")
    config = load_config(args.config)
    system = EcoDisturbanceAttributionSystem(config=config)
    system.load_artifacts()

    if system.feature_columns is None or len(system.feature_columns) == 0:
        import json
        metadata_path = os.path.join(system.out_dir, "model_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        system.feature_columns = metadata.get("feature_columns", [])

    chunk_files = sorted(glob.glob(os.path.join(system.cache_dir, "*.npz")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {system.cache_dir}")
    log.info("[analyze-classes] processing %d chunks", len(chunk_files))

    all_labeled = []
    all_bg = []
    n_labeled = 0
    n_bg = 0

    for i, npz_path in enumerate(chunk_files):
        try:
            data = np.load(npz_path, allow_pickle=True)
            df_chunk = pd.DataFrame({col: data[col] for col in data.files})
            training_df = system._build_chunk_training_samples(df_chunk, skip_sampling=True)
        except Exception:
            continue

        if training_df.empty:
            continue

        labeled_mask = (
            training_df.get("disturbance_class", pd.Series(dtype=int)).isin([1, 2, 3, 4])
            & (training_df["disturbance_flag"] == 1)
        )
        bg_mask = training_df["disturbance_flag"] == 0

        if labeled_mask.sum() > 0:
            all_labeled.append(training_df[labeled_mask])
            n_labeled += labeled_mask.sum()

        if bg_mask.sum() > 0:
            bg_sample = training_df[bg_mask].sample(frac=0.05, random_state=42)
            all_bg.append(bg_sample)
            n_bg += len(bg_sample)

        if (i + 1) % 50 == 0:
            log.info("[analyze-classes] processed %d/%d chunks, %d labeled, %d bg", i + 1, len(chunk_files), n_labeled, n_bg)

    if not all_labeled:
        raise ValueError("No labeled events found after processing chunks")

    labeled_df_all = pd.concat(all_labeled, ignore_index=True)
    bg_df_all = pd.concat(all_bg, ignore_index=True) if all_bg else pd.DataFrame()
    log.info("[analyze-classes] total: %d labeled events, %d background rows", len(labeled_df_all), len(bg_df_all))

    combined = pd.concat([labeled_df_all, bg_df_all], ignore_index=True)
    feature_cols = [col for col in system.feature_columns if col in combined.columns]

    combined_ready = combined[feature_cols + ["disturbance_flag", "disturbance_class"]].dropna(subset=feature_cols)
    log.info("[analyze-classes] after dropna: %d rows (labeled: %d)",
             len(combined_ready), (combined_ready["disturbance_class"] > 0).sum())

    label_df = system.discretizer.transform(combined_ready[feature_cols], as_labels=True)
    for col in label_df.columns:
        label_df[col] = label_df[col].astype(str)
    label_df["disturbance_flag"] = combined_ready["disturbance_flag"].values
    label_df["disturbance_class"] = combined_ready["disturbance_class"].values

    labeled_in_label = (label_df["disturbance_class"] > 0) & (label_df["disturbance_flag"] == 1)
    log.info("[analyze-classes] labels: %s", dict(label_df.loc[labeled_in_label, "disturbance_class"].value_counts().sort_index()))

    analyzer = FactorClassAnalyzer(min_count_per_class=system.min_count_per_class)
    profiles = analyzer.compute_class_conditional(
        factors=system.factor_library,
        label_df=label_df,
        feature_columns=feature_cols,
        discretizer=system.discretizer,
    )
    top_factors = analyzer.find_top_class_specific_factors(
        profiles,
        min_prior_lift=system.class_prior_lift_threshold,
        top_n=system.top_n_per_class,
        min_count=system.min_count_per_class,
    )

    class_priors = {}
    n_labeled_final = labeled_in_label.sum()
    for cls, cls_name in FactorClassAnalyzer.CLASS_LABELS.items():
        class_priors[cls_name] = (label_df.loc[labeled_in_label, "disturbance_class"] == cls).sum() / max(n_labeled_final, 1)

    system._save_class_conditional_results(profiles, top_factors, class_priors, system.out_dir)
    log.info("Class-conditional analysis complete.")


def _setup_file_logging(log_dir: str, prefix: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    return log_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config) if args.command not in ("analyze-classes",) else None
    log_file = None
    if config is not None and args.command in ("run", "pipeline"):
        log_dir = os.path.join(getattr(config.paths, "out_dir", "output"), "..", "logs", args.command)
        log_file = _setup_file_logging(log_dir, args.command)
        logging.getLogger().info("Log file: %s", log_file)
    if args.command == "run":
        system = EcoDisturbanceAttributionSystem(config=config)
        system.run()
    elif args.command == "pipeline":
        processor = PipelineProcessor(config=config)
        processor.process_all_chunks()
    elif args.command == "predict-annual-map":
        mapper = AnnualProbabilityMapper(config=config)
        mapper.write_maps(args.year, output_dir=args.output_dir)
    elif args.command == "analyze-classes":
        _run_analyze_classes(args)


if __name__ == "__main__":
    main()