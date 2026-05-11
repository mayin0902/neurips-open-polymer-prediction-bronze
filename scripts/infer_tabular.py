from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from polymer_prediction.tabular import infer_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["catboost", "xgboost"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    paths = config["paths"]
    output_file = config["files"][args.model]
    infer_tabular(
        competition_dir=paths["competition_dir"],
        descriptor_dir=paths["descriptor_dir"],
        output_path=Path(paths["output_dir"]) / output_file,
        model_name=args.model,
        seed=config.get("seed", 10),
    )


if __name__ == "__main__":
    main()
