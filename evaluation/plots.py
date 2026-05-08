from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_baselines(summary_csv: Path, output_dir: Path) -> None:
    if not summary_csv.exists():
        print("Baseline plot skipped: summary.csv not found.")
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        print("Baseline plot skipped: summary.csv is empty.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].bar(df["method"], df["volume_mape"].fillna(0))
    axes[0].set_ylabel("Volume MAPE (%)")
    axes[0].set_title("Geometry error")
    axes[1].bar(df["method"], df["calories_mae"].fillna(0))
    axes[1].set_ylabel("Calorie MAE (kcal)")
    axes[1].set_title("Nutrition error")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png", dpi=240)
    plt.close(fig)


def plot_ablation(per_run_csv: Path, output_dir: Path) -> None:
    if not per_run_csv.exists():
        print("Ablation plot skipped: per_run.csv not found.")
        return

    df = pd.read_csv(per_run_csv)
    if df.empty or "image_count" not in df:
        print("Ablation plot skipped: per_run.csv missing image_count.")
        return

    grouped = df.groupby("image_count")["volume_abs_error"].mean().reset_index()
    if grouped.empty:
        print("Ablation plot skipped: no volume error data.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(grouped["image_count"], grouped["volume_abs_error"], marker="o")
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Mean absolute volume error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_views.png", dpi=240)
    plt.close(fig)
