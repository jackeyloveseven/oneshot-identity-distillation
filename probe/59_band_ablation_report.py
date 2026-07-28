#!/usr/bin/env python
"""
59_band_ablation_report.py -- prints a Wilson-95%-CI comparison table across
the four band-position-ablation arms (mid 10-25 r8 already existed as
data/gemini_judge_devprobe_base_rep30_step4000.json; early/late/full-depth
are written by probe/58_band_ablation_run.sh as
data/gemini_judge_band_<band>_r<r>_step4000.json).

CAVEAT (2026-07-16): the mid reference arm (run_mix1/step_4000) is NOT
protocol-matched to the other three rows -- different training manifest
(mix2, no ref_caption field) and its published generations carried the
anime_2d prompt prefix. See probe/58 header + data/cleanmid_summary.json
(confound-free re-run) before quoting this table's mid row.

Pre-registered decision rule (agreed before any of the three missing arms
were run): band 10-25 counts as "specially favored" only if its clean-yes
point estimate is the highest of the four AND its 95% CI does not overlap
the CI of the worst-performing arm. If all four arms' CIs overlap, the
result is "band position insensitive / identity is distributed" -- not a
failure, a valid finding either way. This script only reports numbers; it
does not itself render the verdict in prose (read the printed CIs for that
call, don't let this script silently declare a winner).

Run: $PY probe/59_band_ablation_report.py
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

ARMS = [
    ("mid 10-25 r8 (reference, pre-existing)", "gemini_judge_devprobe_base_rep30_step4000.json"),
    ("early 0-15 r8", "gemini_judge_band_0-15_r8_step4000.json"),
    ("late 14-29 r8", "gemini_judge_band_14-29_r8_step4000.json"),
    ("full-depth 0-29 r4", "gemini_judge_band_0-29_r4_step4000.json"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p * 100, max(0.0, (center - half)) * 100, min(1.0, (center + half)) * 100)


def stats(entries):
    n = len(entries)
    defect = sum(1 for e in entries if e.get("visible_defect") == "yes")
    clean_yes = sum(1 for e in entries if e.get("same_character") == "yes" and e.get("visible_defect") != "yes")
    clean_usable = sum(1 for e in entries if e.get("same_character") in ("yes", "partial") and e.get("visible_defect") != "yes")
    return n, wilson(clean_yes, n), wilson(defect, n), wilson(clean_usable, n)


rows = []
for label, fname in ARMS:
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"{label:38s}  MISSING ({fname} not found -- arm not run or failed)")
        rows.append(None)
        continue
    d = json.load(open(path))
    n, cy, df, cu = stats(list(d.values()))
    rows.append((label, n, cy, df, cu))
    print(f"{label:38s} n={n:3d}  clean-yes={cy[0]:5.1f}% [{cy[1]:.0f},{cy[2]:.0f}]  "
          f"defect={df[0]:5.1f}% [{df[1]:.0f},{df[2]:.0f}]  "
          f"clean-usable={cu[0]:5.1f}% [{cu[1]:.0f},{cu[2]:.0f}]")

present = [r for r in rows if r]
if len(present) >= 2:
    best = max(present, key=lambda r: r[2][0])
    worst = min(present, key=lambda r: r[2][0])
    print()
    print(f"highest clean-yes point estimate: {best[0]} ({best[2][0]:.1f}%, CI [{best[2][1]:.0f},{best[2][2]:.0f}])")
    print(f"lowest  clean-yes point estimate: {worst[0]} ({worst[2][0]:.1f}%, CI [{worst[2][1]:.0f},{worst[2][2]:.0f}])")
    overlap = not (best[2][1] > worst[2][2] or worst[2][1] > best[2][2])
    print(f"CI overlap between best and worst arm: {overlap}")
    print("--> per pre-registered rule: " +
          ("NOT specially favored (CIs overlap -- band position insensitive / identity distributed)"
           if overlap else "SPECIALLY FAVORED (CIs do not overlap)"))
else:
    print("\nfewer than 2 arms present -- cannot compare yet")
