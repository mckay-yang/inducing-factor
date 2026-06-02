import json
import os
from dataclasses import asdict
from typing import Iterable

import joblib
import pandas as pd


class ArtifactWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_training_samples(self, df_samples: pd.DataFrame) -> str:
        path = os.path.join(self.out_dir, "training_samples.csv")
        df_samples.to_csv(path, index=False)
        return path

    def save_factor_library(self, factor_library: Iterable, factor_frame: pd.DataFrame, discretizer) -> tuple[str, str]:
        csv_path = os.path.join(self.out_dir, "inducing_factor_library.csv")
        json_path = os.path.join(self.out_dir, "inducing_factor_library.json")
        factor_frame.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump([{**asdict(factor), "items": list(factor.items)} for factor in factor_library], handle, ensure_ascii=False, indent=2)
        joblib.dump(discretizer, os.path.join(self.out_dir, "discretizer.joblib"))
        return csv_path, json_path

    def save_class_model(self, class_engine, factor_class_summary: pd.DataFrame) -> tuple[str, str]:
        model_path = os.path.join(self.out_dir, "disturbance_class_model.joblib")
        summary_path = os.path.join(self.out_dir, "factor_class_summary.csv")
        class_engine.save(model_path)
        factor_class_summary.to_csv(summary_path, index=False)
        meta_path = os.path.join(self.out_dir, "class_model_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump({
                "model_status": getattr(class_engine, "model_status", "unknown"),
                "available_classes": class_engine.available_classes(),
                "is_available": class_engine.is_available(),
            }, handle, indent=2)
        return model_path, summary_path

    def save_occurrence_model(self, occurrence_engine) -> str:
        path = os.path.join(self.out_dir, "occurrence_probability_model.joblib")
        occurrence_engine.save(path)
        return path

    def save_model_metadata(self, metadata: dict) -> str:
        path = os.path.join(self.out_dir, "model_metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return path

    def save_attribution_preview(self, df_attr: pd.DataFrame) -> str:
        path = os.path.join(self.out_dir, "attribution_preview.csv")
        df_attr.to_csv(path, index=False)
        return path
