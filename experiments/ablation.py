import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation summaries from per-run evaluation outputs.")
    parser.add_argument("--per-run", default="evaluation/outputs/per_run.csv")
    parser.add_argument("--output", default="evaluation/outputs/ablation_summary.csv")
    args = parser.parse_args()

    per_run = Path(args.per_run)
    if not per_run.exists():
        raise SystemExit("per_run.csv not found; run evaluation first.")

    df = pd.read_csv(per_run)
    if "image_count" not in df:
        raise SystemExit("image_count missing from per_run.csv")

    summary = df.groupby("image_count")[["volume_abs_error", "calories_abs_error"]].mean().reset_index()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
