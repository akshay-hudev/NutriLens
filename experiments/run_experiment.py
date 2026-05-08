import argparse
from pathlib import Path

from evaluation.evaluate import evaluate_runs
from evaluation.export_results import export_run_metrics, export_summary
from evaluation.plots import plot_ablation, plot_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NutriLens evaluation experiment.")
    parser.add_argument("--runs-root", default="static/uploads/runs")
    parser.add_argument("--ground-truth", default="evaluation/ground_truth.json")
    parser.add_argument("--output", default="evaluation/outputs")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    ground_truth = Path(args.ground_truth)
    output_dir = Path(args.output)

    metrics, summary = evaluate_runs(runs_root, ground_truth)
    if not metrics:
        print("No runs with ground truth found; nothing to export.")
        return

    per_run_csv = export_run_metrics(metrics, output_dir)
    summary_csv = export_summary(summary, output_dir)
    plot_baselines(summary_csv, output_dir)
    plot_ablation(per_run_csv, output_dir)
    print(f"Exported {len(metrics)} run(s) to {output_dir}.")


if __name__ == "__main__":
    main()
