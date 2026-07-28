#!/usr/bin/env python3
"""Draw the test-time conditional score-distillation mechanism (Figure 1).

Layout faithfully mirrors the gpt-image-2 draft
(paper/figures/fig1_method/method_gptimage_draft.png), rebuilt as a clean
vector figure in the "decouple DMD" pastel-trapezoid aesthetic
(20260722-111414.jpg style reference): pastel trapezoid network blocks,
a circular "-" subtraction node, staged left-to-right flow, and a two-box
dashed (a)/(b) comparison on the right.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "paper" / "figures", ROOT / "paper" / "figures" / "fig1_method"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    }
)

# ---------------------------------------------------------------- palette --
SLATE = "#33404A"      # dark-slate outline / ink
CREAM = "#F3E5C0"      # frozen text base
TEAL = "#9CC3BE"       # frozen edit teacher
SALMON = "#F4C7BE"     # scene text / latent update
GREEN = "#2E9E44"      # soft green accent (text-only step)
DASHBOX_FILL = "#EEF3F7"   # light blue-gray dashed comparison boxes
VA_FILL = "#FBF3DC"    # pale token tile matching text base
VR_FILL = "#E9F2F0"    # pale token tile matching edit teacher
WHITE = "#FFFFFF"
GRAY_CAPTION = "#6B6B6B"
GRAY_DASH = "#8A8F94"


# ------------------------------------------------------------- primitives --
def trapezoid(ax, x0, x1, yc, left_h, right_h, face, edge=SLATE, lw=1.3, zorder=3):
    verts = [
        (x0, yc + left_h / 2),
        (x0, yc - left_h / 2),
        (x1, yc - right_h / 2),
        (x1, yc + right_h / 2),
    ]
    poly = Polygon(verts, closed=True, facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(poly)
    return poly


def rrect(
    ax,
    xy,
    w,
    h,
    face,
    edge=SLATE,
    lw=1.3,
    dashed=False,
    rounding=0.045,
    zorder=3,
):
    style = "round,pad=0,rounding_size={}".format(rounding)
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=style,
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    if dashed:
        patch.set_linestyle((0, (4.5, 2.8)))
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, fontsize=10, color=SLATE, weight="normal", ha="center", va="center", style="normal", zorder=5):
    ax.text(x, y, text, fontsize=fontsize, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style, zorder=zorder)


def arrow(
    ax,
    start,
    end,
    color=SLATE,
    lw=1.15,
    mutation_scale=9,
    connectionstyle="arc3,rad=0.0",
    dashed=False,
    zorder=4,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    if dashed:
        patch.set_linestyle((0, (5, 3)))
    ax.add_patch(patch)
    return patch


def lock_icon(ax, cx, cy, s=0.16, color=SLATE):
    """Small vector padlock glyph (shackle arc + body)."""
    shackle = Arc(
        (cx, cy + s * 0.30),
        width=s * 0.78,
        height=s * 0.9,
        theta1=0,
        theta2=180,
        lw=1.3,
        color=color,
        zorder=6,
    )
    ax.add_patch(shackle)
    body = FancyBboxPatch(
        (cx - s * 0.44, cy - s * 0.42),
        s * 0.88,
        s * 0.62,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=0,
        facecolor=color,
        zorder=6,
    )
    ax.add_patch(body)


def x_mark(ax, cx, cy, s=0.09, color=GRAY_DASH, lw=1.6):
    ax.plot([cx - s, cx + s], [cy - s, cy + s], color=color, lw=lw, zorder=6, solid_capstyle="round")
    ax.plot([cx - s, cx + s], [cy + s, cy - s], color=color, lw=lw, zorder=6, solid_capstyle="round")


def dipping_dashed_curve(ax, p0, p1, p2, p3, color=GRAY_DASH, lw=1.25, zorder=2):
    """A cubic-Bezier dashed curve that dips low then rises — used for the
    'editor never advances the scheduler' blocked path, kept well clear of
    the main flow row above it."""
    path = MplPath(
        [p0, p1, p2, p3],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        capstyle="round",
        zorder=zorder,
    )
    patch.set_linestyle((0, (5, 3)))
    ax.add_patch(patch)


def noise_tile(ax, x0, y0, x1, y1, seed=7):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.30, 0.88, size=(20, 20))
    ax.imshow(
        noise,
        cmap="gray",
        extent=(x0, x1, y0, y1),
        vmin=0,
        vmax=1,
        aspect="auto",
        zorder=2,
        interpolation="nearest",
    )
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=SLATE, linewidth=1.3, zorder=3)
    )


# ------------------------------------------------------------------ main --
def main() -> int:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    # =========================================================== LEFT ===
    label(ax, 0.30, 4.87, "Same-state query", fontsize=14, weight="bold", ha="left")

    # --- input tiles (top to bottom: noisy latent, scene text, reference) --
    NL_X0, NL_X1, NL_Y0, NL_Y1 = 0.30, 1.30, 3.25, 4.05
    noise_tile(ax, NL_X0, NL_Y0, NL_X1, NL_Y1)
    label(ax, (NL_X0 + NL_X1) / 2, NL_Y0 - 0.20, "noisy latent", fontsize=10.5)

    ST_X0, ST_X1, ST_Y0, ST_Y1 = 0.30, 1.30, 2.30, 2.75
    rrect(ax, (ST_X0, ST_Y0), ST_X1 - ST_X0, ST_Y1 - ST_Y0, SALMON, rounding=0.05)
    label(ax, (ST_X0 + ST_X1) / 2, (ST_Y0 + ST_Y1) / 2, "scene text", fontsize=8.5)

    RF_X0, RF_X1, RF_Y0, RF_Y1 = 0.40, 0.95, 1.20, 1.62
    rrect(ax, (RF_X0, RF_Y0), RF_X1 - RF_X0, RF_Y1 - RF_Y0, WHITE, rounding=0.04)
    label(ax, (RF_X0 + RF_X1) / 2, RF_Y0 - 0.20, "reference", fontsize=10.5)

    # --- trapezoid network blocks --------------------------------------
    TX0, TX1 = 1.75, 3.15
    TOP_YC, BOT_YC = 3.72, 2.02
    LEFT_H, RIGHT_H = 1.05, 0.50

    trapezoid(ax, TX0, TX1, TOP_YC, LEFT_H, RIGHT_H, CREAM)
    label(ax, TX0 + (TX1 - TX0) * 0.42, TOP_YC, "Frozen\ntext base", fontsize=11)
    lock_icon(ax, TX1 - 0.34, TOP_YC + RIGHT_H / 2 + 0.10)

    trapezoid(ax, TX0, TX1, BOT_YC, LEFT_H, RIGHT_H, TEAL)
    label(ax, TX0 + (TX1 - TX0) * 0.42, BOT_YC, "Frozen\nedit teacher", fontsize=11)
    lock_icon(ax, TX1 - 0.34, BOT_YC + RIGHT_H / 2 + 0.10)

    # --- input -> trapezoid arrows (noisy latent & scene text cross) ----
    nl_pt = (NL_X1, (NL_Y0 + NL_Y1) / 2)
    st_pt = (ST_X1, (ST_Y0 + ST_Y1) / 2)
    rf_pt = (RF_X1, (RF_Y0 + RF_Y1) / 2)

    top_left_hi = (TX0, TOP_YC + LEFT_H * 0.30)
    top_left_lo = (TX0, TOP_YC - LEFT_H * 0.34)
    bot_left_hi = (TX0, BOT_YC + LEFT_H * 0.34)
    bot_left_mid = (TX0, BOT_YC)
    bot_left_lo = (TX0, BOT_YC - LEFT_H * 0.32)

    # noisy latent -> top trapezoid (near-straight) and -> bottom trapezoid (crosses)
    arrow(ax, nl_pt, top_left_hi, SLATE)
    arrow(ax, nl_pt, bot_left_hi, SLATE)
    # scene text -> top trapezoid (crosses) and -> bottom trapezoid (near-straight)
    arrow(ax, st_pt, top_left_lo, SLATE)
    arrow(ax, st_pt, bot_left_mid, SLATE)
    # reference -> edit teacher ONLY
    arrow(ax, rf_pt, bot_left_lo, SLATE)

    # --- trapezoid -> v_A / v_R tokens -----------------------------------
    VA_X0, VA_X1 = 3.32, 3.68
    VA_Y0, VA_Y1 = TOP_YC - 0.225, TOP_YC + 0.225
    rrect(ax, (VA_X0, VA_Y0), VA_X1 - VA_X0, VA_Y1 - VA_Y0, VA_FILL, rounding=0.05)
    label(ax, (VA_X0 + VA_X1) / 2, TOP_YC, r"$v_A$", fontsize=13)

    VR_X0, VR_X1 = 3.32, 3.68
    VR_Y0, VR_Y1 = BOT_YC - 0.225, BOT_YC + 0.225
    rrect(ax, (VR_X0, VR_Y0), VR_X1 - VR_X0, VR_Y1 - VR_Y0, VR_FILL, rounding=0.05)
    label(ax, (VR_X0 + VR_X1) / 2, BOT_YC, r"$v_R$", fontsize=13)

    arrow(ax, (TX1, TOP_YC), (VA_X0, TOP_YC), SLATE)
    arrow(ax, (TX1, BOT_YC), (VR_X0, BOT_YC), SLATE)

    # --- minus node -------------------------------------------------------
    MX, MY, MR = 4.03, 2.87, 0.22
    ax.add_patch(Circle((MX, MY), MR, facecolor=WHITE, edgecolor=SLATE, linewidth=1.3, zorder=4))
    label(ax, MX, MY + 0.02, "−", fontsize=16, weight="bold")

    arrow(
        ax,
        (VA_X1, TOP_YC),
        (MX, MY + MR - 0.02),
        SLATE,
        connectionstyle="angle,angleA=0,angleB=-90,rad=0",
    )
    arrow(
        ax,
        (VR_X1, BOT_YC),
        (MX, MY - MR + 0.02),
        SLATE,
        connectionstyle="angle,angleA=0,angleB=90,rad=0",
    )

    # "score difference" names the minus-node output; the actual connector into
    # box (b) is drawn once latent-update coords exist (see RIGHT section). The
    # exact closed form Delta s_R = -((1-sigma)/sigma)(v_R - v_A) is stated in the
    # caption at full body size rather than crammed here below 9pt.
    label(ax, 4.72, MY + 0.19, "score\ndifference", fontsize=9.5, ha="center")

    label(
        ax,
        (TX0 + TX1) / 2,
        0.85,
        "same latent, same time",
        fontsize=10.5,
        color=GRAY_CAPTION,
        style="italic",
    )

    # =========================================================== RIGHT ==
    RX0, RX1 = 5.20, 9.50

    # --- (a) Direct editor -------------------------------------------------
    A_Y0, A_Y1 = 3.55, 4.85
    rrect(ax, (RX0, A_Y0), RX1 - RX0, A_Y1 - A_Y0, DASHBOX_FILL, dashed=True, rounding=0.035)
    label(ax, RX0 + 0.16, A_Y1 - 0.24, "(a) Direct editor", fontsize=12, weight="bold", ha="left")

    A_TX0, A_TX1 = 5.55, 6.85
    A_YC = 4.05
    trapezoid(ax, A_TX0, A_TX1, A_YC, 0.62, 0.30, TEAL)
    label(ax, A_TX0 + (A_TX1 - A_TX0) * 0.42, A_YC, "Frozen\nedit teacher", fontsize=9.5)
    lock_icon(ax, A_TX1 - 0.24, A_YC + 0.30 / 2 + 0.075, s=0.12)

    SC_X0, SC_X1, SC_Y0, SC_Y1 = 7.65, 8.95, 3.75, 4.35
    rrect(ax, (SC_X0, SC_Y0), SC_X1 - SC_X0, SC_Y1 - SC_Y0, WHITE, rounding=0.05)
    label(ax, (SC_X0 + SC_X1) / 2, (SC_Y0 + SC_Y1) / 2, "scheduler\nstep", fontsize=10.5)
    arrow(ax, (A_TX1, A_YC), (SC_X0, A_YC), SLATE)

    # --- (b) Ours -----------------------------------------------------------
    B_Y0, B_Y1 = 0.35, 3.35
    rrect(ax, (RX0, B_Y0), RX1 - RX0, B_Y1 - B_Y0, DASHBOX_FILL, dashed=True, rounding=0.035)
    label(ax, RX0 + 0.16, B_Y1 - 0.24, "(b) Ours — guide, not sampler", fontsize=12, weight="bold", ha="left")

    ROW_YC = 1.85

    LU_X0, LU_X1, LU_Y0, LU_Y1 = 5.55, 6.40, ROW_YC - 0.30, ROW_YC + 0.30
    rrect(ax, (LU_X0, LU_Y0), LU_X1 - LU_X0, LU_Y1 - LU_Y0, SALMON, rounding=0.05)
    label(ax, (LU_X0 + LU_X1) / 2, ROW_YC, "latent\nupdate", fontsize=10.5)
    # the score difference (minus-node output) flows down-right into latent update
    arrow(ax, (MX + MR + 0.02, MY), (LU_X0, ROW_YC), SLATE, connectionstyle="arc3,rad=-0.2")

    X3_CX, X3_CY, X3_R = 5.75, ROW_YC + 0.70, 0.205
    ax.add_patch(Circle((X3_CX, X3_CY), X3_R, facecolor=SALMON, edgecolor=SLATE, linewidth=1.2, zorder=4))
    label(ax, X3_CX, X3_CY, "×3", fontsize=10, weight="bold")
    arrow(
        ax,
        (X3_CX - 0.02, X3_CY - X3_R),
        (LU_X0 + 0.16, LU_Y1 + 0.01),
        SLATE,
        connectionstyle="arc3,rad=0.55",
        mutation_scale=8,
        lw=1.0,
    )

    TR_X0, TR_X1, TR_Y0, TR_Y1 = 6.70, 8.00, 1.10, 2.60
    rrect(ax, (TR_X0, TR_Y0), TR_X1 - TR_X0, TR_Y1 - TR_Y0, WHITE, rounding=0.04)
    label(ax, (TR_X0 + TR_X1) / 2, TR_Y1 - 0.28, "trust\nregion", fontsize=10.5, weight="bold")

    ELL_CX, ELL_CY = (TR_X0 + TR_X1) / 2, ROW_YC - 0.22
    ax.add_patch(
        Ellipse(
            (ELL_CX, ELL_CY),
            width=1.05,
            height=0.78,
            facecolor="none",
            edgecolor="#5F7A93",
            linewidth=1.2,
            linestyle=(0, (4, 2.4)),
            zorder=4,
        )
    )
    IR_W, IR_H = 0.52, 0.28
    rrect(
        ax,
        (ELL_CX - IR_W / 2, ELL_CY - IR_H / 2),
        IR_W,
        IR_H,
        "#DCEEEA",
        edge="#4E8A80",
        lw=1.1,
        dashed=True,
        rounding=0.03,
    )
    ax.add_patch(Circle((ELL_CX, ELL_CY), 0.035, facecolor=SLATE, edgecolor="none", zorder=6))
    label(ax, ELL_CX, ELL_CY - 0.30, "anchor", fontsize=8.5, color=GRAY_CAPTION)

    arrow(ax, (LU_X1, ROW_YC), (TR_X0, ROW_YC), SLATE)

    TXT_X0, TXT_X1, TXT_Y0, TXT_Y1 = 8.30, 9.25, ROW_YC - 0.30, ROW_YC + 0.30
    rrect(ax, (TXT_X0, TXT_Y0), TXT_X1 - TXT_X0, TXT_Y1 - TXT_Y0, WHITE, edge=GREEN, lw=1.8, rounding=0.05)
    label(ax, (TXT_X0 + TXT_X1) / 2, ROW_YC, "text-only\nstep", fontsize=10.5, weight="bold")

    arrow(ax, (TR_X1, ROW_YC), (TXT_X0, ROW_YC), GREEN, lw=2.3, mutation_scale=11)

    # dashed "editor never advances the scheduler" path: sweeps low, well
    # below the main (b) row, and stops short of the green box with an X.
    dash_p0 = (TX1, BOT_YC - RIGHT_H / 2)
    dash_p3 = (TXT_X0 + 0.40, 0.62)  # end directly under the text-only step
    dash_dip_y = 0.50
    dash_p1 = (dash_p0[0] + 0.35 * (dash_p3[0] - dash_p0[0]), dash_dip_y)
    dash_p2 = (dash_p0[0] + 0.70 * (dash_p3[0] - dash_p0[0]), dash_dip_y)
    dipping_dashed_curve(ax, dash_p0, dash_p1, dash_p2, dash_p3, color=GRAY_DASH, lw=1.25)
    x_mark(ax, dash_p3[0] + 0.05, dash_p3[1] + 0.02, s=0.085)

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "method_score_distillation"
        fig.savefig(base.with_suffix(".pdf"), dpi=400)
        fig.savefig(base.with_suffix(".svg"), dpi=400)
        fig.savefig(base.with_suffix(".png"), dpi=400)
        print(base)

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
