import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline CSV outputs from different runs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to summary.csv files.")
    parser.add_argument("--output", default="evaluation/outputs/baseline_comparison.csv")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(Path(path))
        df["source"] = Path(path).parent.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
