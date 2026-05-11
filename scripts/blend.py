from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from polymer_prediction.ensemble import blend_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    paths = config["paths"]
    files = config["files"]
    output_dir = Path(paths["output_dir"])
    blend_predictions(
        graph_path=output_dir / files["graph"],
        catboost_path=output_dir / files["catboost"],
        xgboost_path=output_dir / files["xgboost"],
        output_path=output_dir / files["submission"],
        weights=config["ensemble"],
    )


if __name__ == "__main__":
    main()
