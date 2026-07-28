#!/usr/bin/env python3
"""Emit data-driven LaTeX table fragments for the paper from the locked frozen
metrics. Fragments are \\input from method_v3/30_tables.tex (main) and
supplementary.tex (full ablation). Regenerate after any metric change; never
hand-edit the emitted files."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = json.loads((ROOT / "out/krea2_tsd_eval30_tagged_v3/eval30_metrics.json").read_text())
ABL = json.loads((ROOT / "out/krea2_tsd_eval30_ablations_v3/ablation_metrics.json").read_text())
TAB = ROOT / "paper" / "tables"
TAB.mkdir(exist_ok=True)
S = EVAL["summary"]
N = S["n"]


def w(name, text):
    (TAB / name).write_text(text.rstrip() + "\n")
    print("wrote", TAB / name)


# ---------- Table: closed-set retrieval ----------
ret = S["retrieval"]
order = [("caption_base", "Attribute text"), ("distilled", "\\textbf{Distilled (ours)}"),
         ("teacher", "Edit teacher"), ("negative_teacher", "Wrong reference")]
lines = [
    r"\begin{table}[tb]", r"\centering",
    r"\caption{Closed-set identity retrieval against all 30 evaluation references"
    r" (\(n=180\)). Identity is scored as discrimination, not only as an absolute"
    r" distance: Top-1 and same-class Top-1 are correct-reference hit rates, mean"
    r" rank is the position of the correct reference, and the margin is the CCIP"
    r" gap to the nearest incorrect identity (positive is safe).}",
    r"\label{tab:retrieval}", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
    r"\begin{tabular}{lrrrr}", r"\toprule",
    r"Method & Top-1 & Same-cls & Rank \(\downarrow\)"
    r" & Margin \(\uparrow\) \\", r"\midrule",
]
for key, label in order:
    r = ret[key]
    t1 = 100.0 * r["top1"] / N
    sc = 100.0 * r["same_class_top1"] / N
    bold = key == "distilled"
    fmt = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
    lines.append(f"{label} & {fmt(f'{t1:.1f}\\%')} & {sc:.1f}\\% & "
                 f"{fmt(f'{r['mean_rank']:.2f}')} & {fmt(f'{r['mean_margin']:+.3f}')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_retrieval.tex", "\n".join(lines))


# ---------- Table: per-scene ----------
scene_order = [("portrait", "Portrait \(640^2\)"), ("rain", "Rain \(512{\\times}768\)"),
               ("kitchen", "Kitchen \(768{\\times}512\)")]
lines = [
    r"\begin{table}[tb]", r"\centering",
    r"\caption{Per-scene identity distance (CCIP \(\downarrow\)) and distilled"
    r" scene retention (CLIP \(\uparrow\)); each scene has \(n=60\). Identity"
    r" transfer holds across close portrait, full-body rain, and side-view"
    r" kitchen-action compositions.}",
    r"\label{tab:perscene}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{lrrrr}", r"\toprule",
    r"Scene & Attr.\ text & \textbf{Distilled} & Teacher & Scene CLIP \\",
    r"\midrule",
]
for key, label in scene_order:
    p = S["per_scene"][key]
    lines.append(f"{label} & {p['caption_base_ccip']:.3f} & "
                 f"\\textbf{{{p['distilled_ccip']:.3f}}} & {p['teacher_ccip']:.3f} & "
                 f"{p['distilled_scene_clip']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_perscene.tex", "\n".join(lines))


# ---------- Table: full 16-condition mechanism ablation ----------
va = ABL["summary"]["variants"]
groups = [
    ("Full method", [("full", "Full method (ours)")]),
    ("Trust geometry", [("fixed_trust", "Fixed trust radius"),
                        ("global_only", "Global-only projection"),
                        ("loose_trust", "No projection (loose)")]),
    ("Optimization", [("one_inner", "One inner update"),
                      ("global_rms", "Global RMS norm"),
                      ("raw_score", "Raw score magnitude")]),
    ("Teacher / window", [("no_ref_boost", "No reference-attn.\\ boost"),
                          ("early_window", "Early-only window"),
                          ("late_window", "Late-only window")]),
    ("Residual construction", [("matched_lora", "Matched-LoRA residual"),
                               ("identity_anchor_const", "Recon.\\ anchor, constant"),
                               ("identity_anchor_late", "Recon.\\ anchor, late"),
                               ("identity_stat_const", "Channel-stat, constant"),
                               ("identity_stat_late", "Channel-stat, late")]),
    ("Causal control", [("wrong_ref_same_text", "Wrong reference")]),
]
lines = [
    r"\begin{table}[tb]", r"\centering",
    r"\caption{Mechanism ablation on the fixed 10-identity subset (three scenes,"
    r" seed 2027, \(n=30\)). CCIP is a distance; \(\Delta\) is the paired change"
    r" versus the full method with a 95\% subject-bootstrap interval. Trust"
    r" projection, a normalized score direction, repeated reevaluation, and the"
    r" correct reference are each independently supported; reconstruction"
    r" diagnostics trade scene fidelity or aesthetics for appearance.}",
    r"\label{tab:ablation}", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
    r"\begin{tabular}{lrrrr}", r"\toprule",
    r"Variant & CCIP \(\downarrow\) & \(\Delta\) vs full [95\% CI] & Scene \(\uparrow\)"
    r" & Aesth. \\", r"\midrule",
]
for gi, (gname, items) in enumerate(groups):
    if gi > 0:
        lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{gname}}}}} \\\\")
    for key, label in items:
        v = va[key]
        ci = v["delta_vs_full_ci95"]
        if key == "full":
            row = (f"\\textbf{{{label}}} & \\textbf{{{v['ccip']:.3f}}} & "
                   f"--- & \\textbf{{{v['scene_clip']:.3f}}} & {v['aesthetic_percentile']:.2f}")
        else:
            row = (f"\\quad {label} & {v['ccip']:.3f} & "
                   f"{v['delta_vs_full']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] & "
                   f"{v['scene_clip']:.3f} & {v['aesthetic_percentile']:.2f}")
        lines.append(row + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_ablation.tex", "\n".join(lines))

# quick echo of per-scene numbers for verification
print("\nper-scene CCIP (attr/dist/teacher) + dist scene clip:")
for key, _ in scene_order:
    p = S["per_scene"][key]
    print(f"  {key:9s} {p['caption_base_ccip']:.3f} / {p['distilled_ccip']:.3f} / "
          f"{p['teacher_ccip']:.3f}  sceneCLIP={p['distilled_scene_clip']:.3f}")
