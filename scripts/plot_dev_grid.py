"""Vẽ độ nhạy Token Accuracy theo alpha và cutoff từ lưới dev đã xuất."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.make_figures import plt, read_table


def plot_dev_grid(grid: Path, output: Path) -> None:
    rows = read_table(grid, {"model", "param_alpha", "param_cutoff", "dev_token_accuracy", "selected"})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True, layout="constrained")
        fig.suptitle("Token Accuracy trên dev theo α và cutoff", fontweight="bold")
        for ax, model in zip(axes, ("bigram", "trigram"), strict=True):
            scores = [
                (float(row["param_alpha"]), int(row["param_cutoff"]),
                 100 * float(row["dev_token_accuracy"]), row["selected"] == "True")
                for row in rows if row["model"] == model
            ]
            for cutoff in sorted({score[1] for score in scores}):
                points = sorted((alpha, accuracy) for alpha, k, accuracy, _ in scores if k == cutoff)
                ax.plot(*zip(*points, strict=True), marker="o", label=f"cutoff = {cutoff}")
            (alpha, _, accuracy, _), = [score for score in scores if score[3]]
            ax.scatter(alpha, accuracy, s=180, facecolors="none", edgecolors="#222222",
                       linewidths=1.4, label="Cấu hình được chọn", zorder=3)
            ax.set(title=model.title(), xlabel="Hệ số làm trơn α", xscale="log")
            alphas = sorted({score[0] for score in scores})
            ax.set_xticks(alphas, labels=[f"{value:g}".replace(".", ",") for value in alphas])
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Token Accuracy (%)")
        fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncols=4,
                   fontsize=9, frameon=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("results/tables/dev/grid_expanded_dev.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/dev/grid_sensitivity.png"))
    args = parser.parse_args()
    plot_dev_grid(args.grid, args.output)
