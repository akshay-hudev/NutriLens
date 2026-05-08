import argparse
from pathlib import Path

from evaluation.evaluate import evaluate_runs
from evaluation.export_results import export_run_metrics, export_summary
from evaluation.plots import plot_ablation, plot_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NutriLens runs against ground-truth metadata.")
    parser.add_argument("--runs-root", default="static/uploads/runs", help="Root folder with run artifacts.")
    parser.add_argument("--ground-truth", default="evaluation/ground_truth.json", help="Ground-truth JSON path.")
    parser.add_argument("--output", default="evaluation/outputs", help="Output directory for CSVs and plots.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.output)

    metrics, summary = evaluate_runs(runs_root, gt_path)
    if not metrics:
        print("No runs with ground truth found; nothing to export.")
        return

    per_run_csv = export_run_metrics(metrics, out_dir)
    summary_csv = export_summary(summary, out_dir)

    plot_baselines(summary_csv, out_dir)
    plot_ablation(per_run_csv, out_dir)
    print(f"Exported {len(metrics)} run(s) to {out_dir}.")


if __name__ == "__main__":
    main()
