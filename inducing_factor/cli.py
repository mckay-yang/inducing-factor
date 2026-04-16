import argparse
import logging

from inducing_factor import AnnualProbabilityMapper, EcoDisturbanceAttributionSystem, PipelineProcessor
from inducing_factor.config import load_config


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

    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "run":
        system = EcoDisturbanceAttributionSystem(config=config)
        system.run()
    elif args.command == "pipeline":
        processor = PipelineProcessor(config=config)
        processor.process_all_chunks()
    elif args.command == "predict-annual-map":
        mapper = AnnualProbabilityMapper(config=config)
        mapper.write_maps(args.year, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
