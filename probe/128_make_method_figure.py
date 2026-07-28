#!/usr/bin/env python3
"""Draw the test-time identity distillation mechanism as an editable figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    }
)

INK = "#272727"
BASE = "#A8A8A8"
OURS = "#0F4D92"
TEACHER = "#42949E"
PALE_BLUE = "#E0F0F0"
PALE_ROSE = "#F0E0E7"
GREEN = "#2E9E44"


def box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str = INK,
    fontsize: float = 6.5,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = INK,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    lw: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def panel_label(ax, x: float, label: str, title: str) -> None:
    ax.text(x, 0.96, label, fontsize=9, fontweight="bold", va="top")
    ax.text(x + 0.025, 0.96, title, fontsize=7.5, fontweight="bold", va="top")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_label(ax, 0.01, "a", "Same-state conditional fields")
    box(ax, (0.03, 0.40), 0.075, 0.20, r"$z_\sigma$", PALE_BLUE, OURS, 9)
    box(
        ax,
        (0.145, 0.65),
        0.145,
        0.17,
        "Attribute text base\n(no reference)",
        "#EFEFEF",
        BASE,
    )
    box(
        ax,
        (0.145, 0.20),
        0.145,
        0.17,
        "Edit teacher\n(reference + text)",
        PALE_BLUE,
        TEACHER,
    )
    arrow(ax, (0.105, 0.52), (0.145, 0.72), BASE)
    arrow(ax, (0.105, 0.48), (0.145, 0.29), TEACHER)
    ax.text(0.307, 0.735, r"$v_A(z_\sigma,\sigma)$", color=BASE, va="center")
    ax.text(0.307, 0.285, r"$v_R(z_\sigma,\sigma)$", color=TEACHER, va="center")
    ax.text(
        0.217,
        0.08,
        "Both frozen, queried at identical state and time",
        ha="center",
        fontsize=5.8,
        color="#606060",
    )

    ax.plot([0.35, 0.35], [0.08, 0.90], color="#D8D8D8", lw=0.8)
    panel_label(ax, 0.37, "b", "Conditional score distillation")
    box(
        ax,
        (0.39, 0.60),
        0.235,
        0.18,
        r"$\Delta s_R=-\frac{1-\sigma}{\sigma}(v_R-v_A)$",
        PALE_ROSE,
        OURS,
        8,
    )
    box(
        ax,
        (0.39, 0.29),
        0.235,
        0.18,
        r"$z^{j+1}=z^j+\alpha\eta d^j$"
        "\n"
        r"$d^j=\mathrm{RMSNorm}_{\mathrm{token}}(\Delta s_R)$",
        "#F7F7FA",
        OURS,
        6.4,
    )
    arrow(ax, (0.508, 0.60), (0.508, 0.47), OURS)
    arrow(
        ax,
        (0.625, 0.36),
        (0.625, 0.69),
        OURS,
        connectionstyle="arc3,rad=-0.42",
    )
    ax.text(0.635, 0.51, "reevaluate\n3 times", color=OURS, fontsize=5.7)
    ax.text(
        0.507,
        0.11,
        "Detached pseudo-loss has this exact latent gradient",
        ha="center",
        fontsize=5.8,
        color="#606060",
    )

    ax.plot([0.665, 0.665], [0.08, 0.90], color="#D8D8D8", lw=0.8)
    panel_label(ax, 0.685, "c", "Trust, then text-only scheduling")
    anchor = (0.75, 0.51)
    ax.add_patch(
        Circle(anchor, 0.105, facecolor=PALE_BLUE, edgecolor=OURS, lw=1.0)
    )
    ax.add_patch(
        Circle(anchor, 0.052, facecolor="white", edgecolor=TEACHER, lw=1.0)
    )
    ax.scatter([anchor[0]], [anchor[1]], s=18, color=INK, zorder=4)
    ax.text(anchor[0], 0.645, "global RMS cap", ha="center", fontsize=5.8)
    ax.text(anchor[0], 0.565, "token cap", ha="center", fontsize=5.6)
    ax.text(anchor[0], 0.475, "anchor", ha="center", fontsize=5.6)
    box(
        ax,
        (0.855, 0.42),
        0.115,
        0.18,
        "Next Euler step\nfrom text base",
        "#EFEFEF",
        BASE,
        6.2,
    )
    arrow(ax, (0.795, 0.51), (0.855, 0.51), GREEN, lw=1.2)
    ax.text(
        0.828,
        0.57,
        "project",
        ha="center",
        fontsize=5.5,
        color=GREEN,
    )
    ax.text(
        0.825,
        0.20,
        "Teacher guides the latent.\nIt never advances the scheduler.",
        ha="center",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
    )

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.03, top=0.98)
    base = OUT / "method_score_distillation"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
