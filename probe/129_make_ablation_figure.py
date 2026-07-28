#!/usr/bin/env python3
"""Create the mechanism-ablation figure from the locked ablation report."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "out" / "krea2_tsd_eval30_ablations_v3" / "ablation_metrics.json"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR_COPY = FIG_DIR / "fig3_ablation"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

INK = "#252525"
FULL = "#15569A"
TRUST = "#3C8C95"
OPT = "#8E6A4A"
DIAGNOSTIC = "#A85B42"
NEGATIVE = "#B33F3F"
MUTED = "#9A9A9A"

LABELS = {
    "full": "Full",
    "fixed_trust": "Fixed trust",
    "global_only": "Global only",
    "loose_trust": "No projection",
    "one_inner": "One inner",
    "global_rms": "Global RMS",
    "raw_score": "Raw score",
    "no_ref_boost": "No ref boost",
    "early_window": "Early only",
    "late_window": "Late only",
    "matched_lora": "Reconstruction residual",
    "identity_anchor_const": "Identity anchor",
    "identity_anchor_late": "Late identity anchor",
    "identity_stat_const": "Channel statistics",
    "identity_stat_late": "Late channel statistics",
    "wrong_ref_same_text": "Wrong reference",
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def color_for(variant: str) -> str:
    if variant == "full":
        return FULL
    if variant in {"fixed_trust", "global_only", "loose_trust"}:
        return TRUST
    if variant in {
        "one_inner",
        "global_rms",
        "raw_score",
        "no_ref_boost",
        "early_window",
        "late_window",
    }:
        return OPT
    if variant == "wrong_ref_same_text":
        return NEGATIVE
    return DIAGNOSTIC


def main() -> int:
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = payload["rows"]
    summary = payload["summary"]
    variants = summary["variants"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_COPY.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.0), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30)

    ax_b = fig.add_subplot(outer[0, 0])
    # Draw "full" last (highest zorder) so its star stays legible even though
    # 2 of the 7 near-identical variants (identity_stat_late/_const) sit
    # almost exactly on top of it in data space.
    all_variants = sorted(variants, key=lambda v: v == "full")
    for variant in all_variants:
        value = variants[variant]
        is_full = variant == "full"
        marker = "*" if is_full else "o"
        size = 70 if is_full else 24
        ax_b.scatter(
            value["scene_clip"],
            value["gain_vs_caption"],
            s=size,
            marker=marker,
            color=color_for(variant),
            edgecolor="white",
            linewidth=0.6 if is_full else 0.45,
            zorder=5 if is_full else 4,
        )
    # Direct labels: only the conditions the caption calls out by name
    # ("full", "trust projection", "raw score", "reference reconstruction")
    # plus the two isolated diagnostic extremes. The full-method star is
    # already unique by shape/size/colour, so it needs no text label.
    # "loose_trust" (no projection) gets an explicit leader line instead of a
    # bare offset so the label cannot read as floating/disconnected.
    ax_b.annotate(
        LABELS["loose_trust"],
        (variants["loose_trust"]["scene_clip"], variants["loose_trust"]["gain_vs_caption"]),
        xytext=(0.312, 0.026),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=5.4,
        color=INK,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=1.5, shrinkB=3.0),
    )
    plain_labels = {
        "raw_score": (4, 6),
        "matched_lora": (5, 7),
        "identity_anchor_const": (4, 5),
        "wrong_ref_same_text": (4, 6),
    }
    for variant, offset in plain_labels.items():
        value = variants[variant]
        ax_b.annotate(
            LABELS[variant],
            (value["scene_clip"], value["gain_vs_caption"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.4,
            color=INK,
        )
    # The full method plus 7 near-identical variants (delta_vs_full <= 0.017,
    # see panel b) sit in one tight cluster around scene_clip~0.36. Panel (b)
    # already names and quantifies each of them individually, so label the
    # cluster once here instead of crowding the scatter with 8 overlapping
    # text boxes.
    cluster_members = [
        "full", "fixed_trust", "global_only", "one_inner",
        "global_rms", "no_ref_boost", "identity_stat_const", "identity_stat_late",
    ]
    cx = float(np.mean([variants[v]["scene_clip"] for v in cluster_members]))
    cy = float(np.mean([variants[v]["gain_vs_caption"] for v in cluster_members]))
    ax_b.annotate(
        "8 variants ≈ full\n(quantified in b)",
        (cx, cy),
        xytext=(0.320, 0.098),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=5.4,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=1.5, shrinkB=4.0),
    )
    ax_b.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax_b.set_xlim(0.265, 0.373)
    ax_b.set_ylim(-0.11, 0.14)
    ax_b.set_xlabel("Scene CLIP ↑")
    ax_b.set_ylabel("CCIP improvement over text ↑")
    ax_b.tick_params(width=0.8, length=3, labelsize=6.2)
    panel_label(ax_b, "a")

    ax_c = fig.add_subplot(outer[0, 1])
    selected = [
        "fixed_trust", "global_only", "loose_trust", "one_inner", "global_rms",
        "raw_score", "no_ref_boost", "early_window", "late_window",
        "matched_lora", "identity_anchor_const", "identity_stat_late",
        "wrong_ref_same_text",
    ]
    # order the forest plot by effect size so the story (reconstruction residual
    # and wrong reference hurt most; the rest sit near zero) reads top-to-bottom.
    selected.sort(key=lambda key: variants[key]["delta_vs_full"], reverse=True)
    positions = np.arange(len(selected))
    # faint alternating row bands so the eye tracks each label to its bar
    # across 13 rows without needing gridlines.
    for pos in positions[::2]:
        ax_c.axhspan(pos - 0.5, pos + 0.5, color="#F2F2F2", zorder=0, linewidth=0)
    means = np.asarray([variants[key]["delta_vs_full"] for key in selected])
    intervals = np.asarray(
        [variants[key]["delta_vs_full_ci95"] for key in selected]
    )
    errors = np.vstack([means - intervals[:, 0], intervals[:, 1] - means])
    ax_c.errorbar(
        means,
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#6E6E6E",
        elinewidth=0.8,
        capsize=1.5,
        zorder=2,
    )
    ax_c.scatter(
        means,
        positions,
        s=18,
        c=[color_for(key) for key in selected],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax_c.axvline(0, color=INK, lw=0.8, zorder=1)
    ax_c.set_yticks(positions, [LABELS[key] for key in selected], fontsize=5.4)
    ax_c.set_ylim(positions[-1] + 0.5, positions[0] - 0.5)
    ax_c.tick_params(axis="x", width=0.8, length=3, labelsize=6.2)
    ax_c.tick_params(axis="y", length=0)
    ax_c.set_xlabel("CCIP change vs full (lower is better)")
    panel_label(ax_c, "b")

    from matplotlib.lines import Line2D
    legend_items = [("Full method", FULL), ("Trust geometry", TRUST),
                    ("Optimization", OPT), ("Reconstruction diagnostic", DIAGNOSTIC),
                    ("Wrong reference", NEGATIVE)]
    # "Full method" uses a star marker in the legend swatch to match its
    # star glyph in panel (a), the same shape that keeps it legible in the
    # dense cluster there.
    handles = [Line2D([0], [0], marker="*" if name == "Full method" else "o",
                      linestyle="none", markerfacecolor=c,
                      markeredgecolor="white",
                      markeredgewidth=0.4,
                      markersize=9 if name == "Full method" else 6,
                      label=name) for name, c in legend_items]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.885),
               ncol=5, fontsize=6.2, handletextpad=0.25, columnspacing=1.0)
    fig.suptitle(
        "Trust regions stabilize transfer; reconstruction fields import layout",
        x=0.50, y=0.985, fontsize=9, fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.79)

    base = FIG_DIR / "mechanism_ablation"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    csv_path = FIG_DIR / "mechanism_ablation_source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "ccip",
                "gain_vs_caption",
                "delta_vs_full",
                "delta_vs_full_ci95_low",
                "delta_vs_full_ci95_high",
                "scene_clip",
                "specificity",
                "aesthetic_percentile",
            ]
        )
        for variant, values in variants.items():
            writer.writerow(
                [
                    variant,
                    values["ccip"],
                    values["gain_vs_caption"],
                    values["delta_vs_full"],
                    *values["delta_vs_full_ci95"],
                    values["scene_clip"],
                    values["specificity"],
                    values["aesthetic_percentile"],
                ]
            )

    # keep the paper/figures/fig3_ablation/ provenance copy in sync.
    for suffix in (".svg", ".pdf", ".png"):
        shutil.copy2(base.with_suffix(suffix), FIG_DIR_COPY / f"mechanism_ablation{suffix}")
    shutil.copy2(csv_path, FIG_DIR_COPY / csv_path.name)

    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
