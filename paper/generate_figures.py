from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
EVAL_OUT = Path(__file__).resolve().parent.parent / "evaluation" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = "#2F6B4F"
BLUE = "#2E6DA4"
AMBER = "#C46B32"
GRAY = "#5F6B66"
LIGHT_GREEN = "#E8F2EC"
LIGHT_BLUE = "#E7F0F8"
LIGHT_AMBER = "#F7EADF"
LINE = "#D4DDD5"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc="#FFFFFF", ec=LINE, fontsize=8):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17201B",
        weight="semibold",
    )


def arrow(ax, start, end, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def figure_system_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    layers = [
        ("Frontend", 0.78, LIGHT_BLUE),
        ("Flask services", 0.48, LIGHT_GREEN),
        ("Run storage", 0.18, LIGHT_AMBER),
    ]
    for title, y, color in layers:
        ax.add_patch(Rectangle((0.02, y - 0.13), 0.96, 0.22, facecolor=color, edgecolor=LINE))
        ax.text(0.035, y + 0.055, title, fontsize=9, weight="bold", color=GRAY)

    frontend = [
        ("Upload\n1-5 views", 0.11),
        ("Confirm\nsession", 0.30),
        ("3D report\n+ warnings", 0.49),
        ("User\nfeedback", 0.68),
        ("New run /\nreset", 0.86),
    ]
    for text, x in frontend:
        box(ax, (x - 0.06, 0.745), (0.12, 0.095), text, fc="white", fontsize=7.5)

    services = [
        ("Image I/O\nEXIF + validation", 0.10),
        ("Food\nsegmentation", 0.25),
        ("SfM / pose\nCOLMAP or prior", 0.40),
        ("3D module\n3DGS proxy / NeRF hook", 0.57),
        ("Volume\nestimation", 0.73),
        ("Nutrition\n+ confidence", 0.89),
    ]
    for text, x in services:
        box(ax, (x - 0.065, 0.435), (0.13, 0.105), text, fc="white", fontsize=7.2)
    for i in range(len(services) - 1):
        arrow(ax, (services[i][1] + 0.067, 0.49), (services[i + 1][1] - 0.067, 0.49))

    storage = [
        ("Normalized\nimages", 0.18),
        ("Masks +\noverlays", 0.36),
        ("Pose /\npoint cloud", 0.54),
        ("Volume JSON\nand report", 0.72),
    ]
    for text, x in storage:
        box(ax, (x - 0.07, 0.135), (0.14, 0.095), text, fc="white", fontsize=7.5)

    for x in [0.10, 0.25, 0.40, 0.57, 0.73, 0.89]:
        arrow(ax, (x, 0.435), (x, 0.235), color="#87928B")
    arrow(ax, (0.49, 0.745), (0.89, 0.54), color=GREEN)
    ax.text(0.04, 0.94, "NutriLens3D system architecture", fontsize=12, weight="bold")
    ax.text(
        0.04,
        0.90,
        "The contribution is system-level integration; native 3DGS/NeRF components remain modular and optional.",
        fontsize=8,
        color=GRAY,
    )
    save(fig, "system_architecture")


def figure_sparse_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.08, 0.23, 0.39, 0.55, 0.71, 0.87]
    titles = [
        "Sparse\nviews",
        "Feature\nmatches",
        "Camera\nposes",
        "Sparse\ncloud",
        "Gaussian\nproxy",
        "Volume\nestimate",
    ]
    for i, (x, title) in enumerate(zip(xs, titles)):
        box(ax, (x - 0.055, 0.68), (0.11, 0.10), title, fc="white", fontsize=7.5)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.06, 0.73), (xs[i + 1] - 0.06, 0.73), color=GREEN)

    rng = np.random.default_rng(3)
    for x in xs[:3]:
        for j in range(3):
            ax.add_patch(Rectangle((x - 0.045 + j * 0.03, 0.38 + j * 0.025), 0.055, 0.08, angle=8 * j, facecolor="#F9FBF8", edgecolor=LINE))
            ax.add_patch(Ellipse((x - 0.018 + j * 0.03, 0.42 + j * 0.025), 0.035, 0.025, facecolor=AMBER, edgecolor="none", alpha=0.85))
    for _ in range(14):
        x1, y1 = 0.23 + rng.normal(0, 0.045), 0.33 + rng.random() * 0.16
        x2, y2 = 0.39 + rng.normal(0, 0.045), 0.33 + rng.random() * 0.16
        ax.plot([x1, x2], [y1, y2], color=BLUE, linewidth=0.6, alpha=0.35)
    for x in [0.37, 0.40, 0.43]:
        ax.add_patch(Polygon([[x, 0.36], [x - 0.018, 0.31], [x + 0.018, 0.31]], facecolor=BLUE, alpha=0.75))
    cloud = rng.normal(size=(55, 2)) * [0.035, 0.025] + [0.55, 0.39]
    ax.scatter(cloud[:, 0], cloud[:, 1], s=8, color=GREEN, alpha=0.75)
    for _ in range(18):
        cx, cy = rng.normal(0.71, 0.035), rng.normal(0.39, 0.025)
        ax.add_patch(Ellipse((cx, cy), 0.04, 0.018, angle=rng.uniform(-40, 40), color=AMBER, alpha=0.25))
    ax.add_patch(Ellipse((0.87, 0.39), 0.12, 0.07, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.3))
    ax.text(0.87, 0.30, "V = 862 cm3\nlow/medium conf.", ha="center", fontsize=7.5, color=GRAY)
    ax.text(0.04, 0.91, "Sparse-view reconstruction workflow", fontsize=12, weight="bold")
    save(fig, "sparse_view_pipeline")


def food_image(size=180):
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.ones((size, size, 3), dtype=float)
    img[:] = [0.92, 0.91, 0.86]
    plate = ((xx - size / 2) ** 2 / (0.42 * size) ** 2 + (yy - size / 2) ** 2 / (0.36 * size) ** 2) <= 1
    img[plate] = [0.98, 0.98, 0.95]
    food = ((xx - size / 2) ** 2 / (0.30 * size) ** 2 + (yy - size / 2) ** 2 / (0.24 * size) ** 2) <= 1
    rng = np.random.default_rng(4)
    img[food] = [0.78, 0.43, 0.20]
    grains = rng.random((size, size)) > 0.965
    img[food & grains] = [0.98, 0.82, 0.42]
    garnish = rng.random((size, size)) > 0.985
    img[food & garnish] = [0.12, 0.42, 0.22]
    return img, food


def figure_segmentation():
    img, mask = food_image()
    overlay = img.copy()
    overlay[mask] = overlay[mask] * 0.55 + np.array([0.05, 0.42, 0.28]) * 0.45
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8))
    for ax in axes:
        ax.set_axis_off()
    axes[0].imshow(img)
    axes[0].set_title("Input view", fontsize=9)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Binary food mask", fontsize=9)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay for QA", fontsize=9)
    fig.suptitle("Segmentation evidence used by the volume stage", y=1.02, fontsize=12, weight="bold")
    save(fig, "segmentation_visualization")


def figure_reconstruction():
    rng = np.random.default_rng(7)
    theta = rng.uniform(0, 2 * np.pi, 240)
    r = np.sqrt(rng.uniform(0, 1, 240))
    x = 1.3 * r * np.cos(theta)
    y = 0.9 * r * np.sin(theta)
    z = 0.20 + 0.55 * (1 - r**1.6) + rng.normal(0, 0.035, 240)

    fig = plt.figure(figsize=(9.5, 3.2))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133)
    for ax in [ax1, ax2]:
        ax.view_init(24, -58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1.5, 1.0, 0.55))
    ax1.scatter(x[::3], y[::3], z[::3], s=6, c=GREEN, alpha=0.85)
    ax1.set_title("Sparse proxy cloud", fontsize=9)
    ax2.scatter(x, y, z, s=22, c=AMBER, alpha=0.18, edgecolors="none")
    ax2.scatter(x[::8], y[::8], z[::8], s=6, c=GREEN, alpha=0.55)
    ax2.set_title("Gaussian-like primitives", fontsize=9)
    grid = np.linspace(-1.5, 1.5, 120)
    X, Y = np.meshgrid(grid, grid)
    occ = np.exp(-((X / 1.25) ** 2 + (Y / 0.85) ** 2) * 2.2)
    ax3.contourf(X, Y, occ, levels=12, cmap="Greens")
    ax3.contour(X, Y, occ, levels=[0.32], colors=[AMBER], linewidths=2)
    ax3.set_aspect("equal")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Occupancy footprint", fontsize=9)
    fig.suptitle("Reconstruction visualization: implemented proxy, native 3DGS-compatible interface", y=1.03, fontsize=12, weight="bold")
    save(fig, "reconstruction_visualization")


def figure_volume():
    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    ax.set_axis_off()
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.0, 1.25)
    ax.add_patch(Ellipse((0, -0.1), 2.35, 1.18, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.4))
    ax.add_patch(Ellipse((0, 0.10), 1.72, 0.72, facecolor=AMBER, alpha=0.72, edgecolor="#7B3B17", linewidth=1.2))
    ax.add_patch(Ellipse((0, 0.32), 1.15, 0.42, facecolor="#D98B45", alpha=0.55, edgecolor="none"))
    ax.plot([-0.86, 0.86], [-0.58, -0.58], color=BLUE, linewidth=1.5)
    ax.plot([1.02, 1.02], [-0.46, 0.48], color=BLUE, linewidth=1.5)
    arrow(ax, (-0.86, -0.58), (-1.25, -0.58), color=BLUE)
    arrow(ax, (0.86, -0.58), (1.25, -0.58), color=BLUE)
    arrow(ax, (1.02, -0.46), (1.02, -0.80), color=BLUE)
    arrow(ax, (1.02, 0.48), (1.02, 0.82), color=BLUE)
    ax.text(0, -0.82, "footprint width/depth from masks + pose scale", ha="center", fontsize=8, color=GRAY)
    ax.text(1.18, 0.02, "height prior\nor occupancy\nthreshold", fontsize=8, color=GRAY)
    ax.text(-1.48, 0.88, "V = gamma * 4/3*pi*(w/2)*(d/2)*(h/2)", fontsize=9, weight="bold", color="#17201B")
    ax.text(-1.48, 0.70, "Reported as interval: [V_l, V_u] after uncertainty inflation", fontsize=8, color=GRAY)
    save(fig, "volume_estimation")


def figure_baselines():
    summary_csv = EVAL_OUT / "summary.csv"
    if not summary_csv.exists():
        print("Skipping baseline comparison: evaluation/outputs/summary.csv not found.")
        return

    rows = []
    with summary_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]

    if not rows:
        print("Skipping baseline comparison: summary.csv is empty.")
        return

    methods = [row["method"] for row in rows]
    volume = [float(row.get("volume_mape") or 0.0) for row in rows]
    cal = [float(row.get("calories_mae") or 0.0) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].bar(methods, volume, color=GREEN)
    axes[0].set_ylabel("Volume MAPE (%)")
    axes[0].set_title("Geometry error")
    axes[1].bar(methods, cal, color=AMBER)
    axes[1].set_ylabel("Calorie MAE (kcal)")
    axes[1].set_title("Nutrition error")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    fig.suptitle("Baseline comparison from evaluation outputs", y=1.03, fontsize=12, weight="bold")
    save(fig, "baseline_comparison")


def figure_failure_cases():
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.7))
    labels = ["Poor mask", "Low texture", "Single view", "Occlusion"]
    for ax, label in zip(axes, labels):
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.8, transform=ax.transAxes, facecolor="#F9FBF8", edgecolor=LINE))
        ax.add_patch(Ellipse((0.5, 0.45), 0.55, 0.35, transform=ax.transAxes, facecolor=AMBER, alpha=0.65))
        if label == "Poor mask":
            ax.add_patch(Ellipse((0.47, 0.43), 0.78, 0.55, transform=ax.transAxes, fill=False, edgecolor=BLUE, linewidth=2, linestyle="--"))
        elif label == "Low texture":
            ax.add_patch(Ellipse((0.5, 0.45), 0.50, 0.30, transform=ax.transAxes, facecolor="#D8CDB8", alpha=0.9))
        elif label == "Single view":
            ax.add_patch(Rectangle((0.36, 0.21), 0.28, 0.48, transform=ax.transAxes, fill=False, edgecolor=BLUE, linewidth=2))
            ax.text(0.5, 0.13, "depth ambiguous", ha="center", transform=ax.transAxes, fontsize=7, color=GRAY)
        else:
            ax.add_patch(Rectangle((0.08, 0.15), 0.45, 0.6, transform=ax.transAxes, facecolor="#FFFFFF", alpha=0.95, edgecolor=GRAY))
        ax.set_title(label, fontsize=9)
    fig.suptitle("Failure cases that reduce confidence", y=1.04, fontsize=12, weight="bold")
    save(fig, "failure_cases")


def figure_ablation():
    per_run_csv = EVAL_OUT / "per_run.csv"
    if not per_run_csv.exists():
        print("Skipping ablation plots: evaluation/outputs/per_run.csv not found.")
        return

    rows = []
    with per_run_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader if row.get("image_count")]

    if not rows:
        print("Skipping ablation plots: per_run.csv is empty.")
        return

    image_counts = sorted({int(row["image_count"]) for row in rows})
    volume_errors = []
    for count in image_counts:
        subset = [row for row in rows if int(row["image_count"]) == count]
        values = [float(row["volume_abs_error"]) for row in subset if row.get("volume_abs_error")]
        volume_errors.append(np.mean(values) if values else 0.0)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(image_counts, volume_errors, marker="o", color=GREEN)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Mean absolute volume error")
    ax.set_title("View count ablation")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "ablation_plots")


if __name__ == "__main__":
    figure_system_architecture()
    figure_sparse_pipeline()
    figure_segmentation()
    figure_reconstruction()
    figure_volume()
    figure_baselines()
    figure_failure_cases()
    figure_ablation()
    print(f"Generated figures in {OUT}")
