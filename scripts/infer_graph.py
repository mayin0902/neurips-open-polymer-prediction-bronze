from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from polymer_prediction.graph import infer_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    paths = config["paths"]
    infer_graph(
        competition_dir=paths["competition_dir"],
        checkpoint_path=args.checkpoint,
        output_path=Path(paths["output_dir"]) / config["files"]["graph"],
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
