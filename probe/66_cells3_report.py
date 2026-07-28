#!/usr/bin/env python
"""
66_cells3_report.py -- closes the rep30 2x2 protocol factorial and evaluates
the step6000 operating point under the clean protocol.

The 2x2 (subject set fixed = rep30, band 10-25 r8 arms only):

                     prompt WITH prefix          prompt raw (prefix-free)
  run_mix1/step4000  (published mid, 3.3%)       Cell A  m1raw   <- new
  run_mix2/step4000  Cell B  m2pfx  <- new       clean-mid (30.0%)

plus Cell C: run_mix1/step_6000 on holdout30, raw prompt_B -- the
pre-registered "identity-max" operating point (38% clean-yes / 70% usable
under the OLD prefixed protocol, design doc 2026-07-11 SS11.1) re-measured
under the clean protocol, compared against run_mix1/step_4000 clean-protocol
ho30 numbers and the krea2 baseline (same 30 ho30 subjects, its own frozen
plain-prompt_B protocol).

Selftest discipline: --selftest must pass before the main report is trusted.
It (a) reproduces four already-reported cells exactly from their JSONs
(published mid rep30, clean-mid rep30, fresh mid ho30 step4000, krea2
holdout30), and (b) proves the local generic-pid clustered bootstrap (needed
because krea2 pids are '<uid>_krea2_s<seed>', which probe/64's stricter
'_lora_' regex rejects) is numerically identical to probe/64's function on a
lora-style file. Stats helpers otherwise importlib-reused from probe/64.

Run:
  $PY probe/66_cells3_report.py --selftest
  $PY probe/66_cells3_report.py            # writes data/cells3_summary.json
"""
import argparse
import importlib.util
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

_spec = importlib.util.spec_from_file_location(
    "r64", os.path.join(HERE, "64_ho30_band_replication_report.py"))
r64 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r64)

GENERIC_PID_RE = re.compile(r"^(\d+)_.*_s([01])$")

FILES = {
    "pub_m1_prefix_rep30": "gemini_judge_devprobe_base_rep30_step4000.json",
    "A_m1_raw_rep30":      "gemini_judge_m1raw_rep30_band_10-25_r8_step4000.json",
    "B_m2_prefix_rep30":   "gemini_judge_m2pfx_rep30_band_10-25_r8_step4000.json",
    "cm_m2_raw_rep30":     "gemini_judge_cleanmid_rep30_band_10-25_r8_step4000.json",
    "C_m1s6k_raw_ho30":    "gemini_judge_m1s6k_ho30_band_10-25_r8_step6000.json",
    "m1s4k_raw_ho30":      "gemini_judge_ho30_band_10-25_r8_step4000.json",
    "krea2_ho30":          "gemini_judge_krea2.json",
}


def clustered_defect_generic(d, n_boot=r64.BOOT_N, seed=r64.BOOT_SEED):
    """Same algorithm as r64.clustered_bootstrap_defect but with a laxer pid
    regex so non-lora pid schemes (krea2: '<uid>_krea2_s<s>') cluster too.
    Selftest asserts numerical identity with r64's function on a lora file."""
    by_uid = {}
    seen = set()
    for pid, verdict in d.items():
        m = GENERIC_PID_RE.match(pid)
        if not m:
            raise ValueError(f"unparseable pid {pid!r}")
        uid, s = m.group(1), int(m.group(2))
        if (uid, s) in seen:
            raise ValueError(f"duplicate (uid,seed) {(uid, s)}")
        seen.add((uid, s))
        by_uid.setdefault(uid, {})[s] = verdict.get("visible_defect") == "yes"
    vals, incomplete = [], []
    for uid in sorted(by_uid):
        if set(by_uid[uid].keys()) != {0, 1}:
            incomplete.append((uid, sorted(by_uid[uid].keys())))
            continue
        vals.append(1 if (by_uid[uid][0] or by_uid[uid][1]) else 0)
    n = len(vals)
    point = 100.0 * sum(vals) / n if n else 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(100.0 * sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"n_subjects": n, "any_defect_k": sum(vals), "point_pct": point,
            "ci95_pct": (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]),
            "incomplete_subjects": incomplete}


def stats_for(fname):
    d, _ = r64.load(fname)
    if d is None:
        return None
    return {"file": fname,
            "per_image": r64.per_image_stats(list(d.values())),
            "clustered_defect": clustered_defect_generic(d)}


def fmt(s):
    if s is None:
        return "MISSING"
    cy = s["per_image"]["clean_yes"]["pct_ci"]
    df = s["per_image"]["defect"]["pct_ci"]
    cu = s["per_image"]["clean_usable"]["pct_ci"]
    cc = s["clustered_defect"]["ci95_pct"]
    return (f"n={s['per_image']['n']} err={s['per_image']['judge_errors']}  "
            f"clean-yes={cy[0]:5.1f}% [{cy[1]:.0f},{cy[2]:.0f}]  "
            f"defect={df[0]:5.1f}% [{df[1]:.0f},{df[2]:.0f}]  "
            f"usable={cu[0]:5.1f}% [{cu[1]:.0f},{cu[2]:.0f}]  "
            f"clust-defect={s['clustered_defect']['point_pct']:5.1f}% [{cc[0]:.0f},{cc[1]:.0f}]")


def selftest():
    ok = True
    # (a) exact reproduction of already-reported cells
    expect = {
        "pub_m1_prefix_rep30": {"defect": 3.3, "clean_yes": 21.7, "clustered": 6.7},
        "cm_m2_raw_rep30":     {"defect": 30.0, "clean_yes": 20.0, "clustered": 53.3},
        "m1s4k_raw_ho30":      {"defect": 20.0, "clean_yes": 18.3, "clustered": 36.7},
        "krea2_ho30":          {"defect": 1.7, "clean_yes": 33.3},  # SS11.1's "2%/33%" to one decimal
    }
    for label, exp in expect.items():
        s = stats_for(FILES[label])
        if s is None:
            print(f"SELFTEST SKIP {label}: missing file")
            ok = False
            continue
        got = {"defect": s["per_image"]["defect"]["pct_ci"][0],
               "clean_yes": s["per_image"]["clean_yes"]["pct_ci"][0],
               "clustered": s["clustered_defect"]["point_pct"]}
        row_ok = all(abs(got[k] - exp[k]) < 0.05 for k in exp)
        ok = ok and row_ok
        print(f"SELFTEST {label:22s} {'OK      ' if row_ok else 'MISMATCH'} "
              + "  ".join(f"{k}={got[k]:.1f} (exp {exp[k]})" for k in exp))
    # (b) generic clustering == r64 clustering on a lora-style file
    d, _ = r64.load(FILES["m1s4k_raw_ho30"])
    if d is not None:
        a = clustered_defect_generic(d)
        b = r64.clustered_bootstrap_defect(d)
        same = (a["point_pct"] == b["point_pct"] and a["ci95_pct"] == b["ci95_pct"]
                and a["any_defect_k"] == b["any_defect_k"])
        print(f"SELFTEST generic-vs-r64 clustering identical: {same}")
        ok = ok and same
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    out = {"_meta": {
        "purpose": ("rep30 2x2 protocol factorial (checkpoint run_mix1/run_mix2 x prompt prefix/raw) "
                    "+ step6000 clean-protocol operating point on holdout30"),
        "boot_n": r64.BOOT_N, "boot_seed": r64.BOOT_SEED, "wilson_z": 1.96,
    }, "cells": {}}

    for label, fname in FILES.items():
        s = stats_for(fname)
        out["cells"][label] = s
        print(f"{label:22s}  {fmt(s)}")

    c = out["cells"]
    verdicts = {}

    # ---- (1) 2x2 factorial main effects on per-image defect points ----
    if all(c[k] for k in ("pub_m1_prefix_rep30", "A_m1_raw_rep30", "B_m2_prefix_rep30", "cm_m2_raw_rep30")):
        d_pub = c["pub_m1_prefix_rep30"]["per_image"]["defect"]["pct_ci"][0]
        d_A = c["A_m1_raw_rep30"]["per_image"]["defect"]["pct_ci"][0]
        d_B = c["B_m2_prefix_rep30"]["per_image"]["defect"]["pct_ci"][0]
        d_cm = c["cm_m2_raw_rep30"]["per_image"]["defect"]["pct_ci"][0]
        fx = {
            "prefix_effect_within_m1": d_A - d_pub,     # raw minus prefix
            "prefix_effect_within_m2": d_cm - d_B,
            "ckpt_effect_within_prefix": d_B - d_pub,   # m2 minus m1
            "ckpt_effect_within_raw": d_cm - d_A,
            "interaction": (d_cm - d_B) - (d_A - d_pub),
            "published_vs_cleanmid_gap": d_cm - d_pub,
        }
        fx["prefix_share_of_gap_within_m1"] = (d_A - d_pub) / (d_cm - d_pub) if d_cm != d_pub else None
        verdicts["factorial_defect_points"] = {"m1_prefix": d_pub, "m1_raw": d_A, "m2_prefix": d_B, "m2_raw": d_cm}
        verdicts["factorial_effects_pp"] = fx
        # clustered CI overlap calls for the two prefix contrasts
        verdicts["m1_raw_vs_m1_prefix_clustered_overlap"] = r64.ci_overlap(
            c["A_m1_raw_rep30"]["clustered_defect"]["ci95_pct"],
            c["pub_m1_prefix_rep30"]["clustered_defect"]["ci95_pct"])
        verdicts["m2_prefix_vs_m2_raw_clustered_overlap"] = r64.ci_overlap(
            c["B_m2_prefix_rep30"]["clustered_defect"]["ci95_pct"],
            c["cm_m2_raw_rep30"]["clustered_defect"]["ci95_pct"])
        verdicts["ckpt_effect_within_raw_clustered_overlap"] = r64.ci_overlap(
            c["A_m1_raw_rep30"]["clustered_defect"]["ci95_pct"],
            c["cm_m2_raw_rep30"]["clustered_defect"]["ci95_pct"])
        verdicts["ckpt_effect_within_prefix_clustered_overlap"] = r64.ci_overlap(
            c["B_m2_prefix_rep30"]["clustered_defect"]["ci95_pct"],
            c["pub_m1_prefix_rep30"]["clustered_defect"]["ci95_pct"])
        # clean-yes across the four cells (expect flat)
        cys = {k: c[k]["per_image"]["clean_yes"]["pct_ci"] for k in
               ("pub_m1_prefix_rep30", "A_m1_raw_rep30", "B_m2_prefix_rep30", "cm_m2_raw_rep30")}
        lo = max(v[1] for v in cys.values())
        hi = min(v[2] for v in cys.values())
        verdicts["factorial_cleanyes_all_overlap"] = lo <= hi
        verdicts["factorial_cleanyes_points"] = {k: v[0] for k, v in cys.items()}

    # ---- (2) step6000 clean-protocol operating point ----
    if c.get("C_m1s6k_raw_ho30") and c.get("m1s4k_raw_ho30") and c.get("krea2_ho30"):
        s6, s4, kr = c["C_m1s6k_raw_ho30"], c["m1s4k_raw_ho30"], c["krea2_ho30"]
        verdicts["s6k_vs_s4k_ho30"] = {
            "clean_yes": (s6["per_image"]["clean_yes"]["pct_ci"][0], s4["per_image"]["clean_yes"]["pct_ci"][0]),
            "defect": (s6["per_image"]["defect"]["pct_ci"][0], s4["per_image"]["defect"]["pct_ci"][0]),
            "usable": (s6["per_image"]["clean_usable"]["pct_ci"][0], s4["per_image"]["clean_usable"]["pct_ci"][0]),
        }
        cy6 = s6["per_image"]["clean_yes"]["pct_ci"]
        cyk = kr["per_image"]["clean_yes"]["pct_ci"]
        verdicts["s6k_vs_krea2_cleanyes_overlap"] = r64.ci_overlap(tuple(cy6[1:]), tuple(cyk[1:]))
        verdicts["s6k_vs_krea2_cleanyes_points"] = (cy6[0], cyk[0])
        verdicts["s6k_vs_krea2_defect_points"] = (s6["per_image"]["defect"]["pct_ci"][0],
                                                    kr["per_image"]["defect"]["pct_ci"][0])
        if cy6[0] >= cyk[0] and verdicts["s6k_vs_krea2_cleanyes_overlap"]:
            call = "TIE, point ahead (as under old protocol)"
        elif verdicts["s6k_vs_krea2_cleanyes_overlap"]:
            call = "TIE, point behind (degraded from old 38-vs-33 point-ahead tie)"
        elif cy6[0] > cyk[0]:
            call = "UPGRADE (clean-yes CI-separated above krea2)"
        else:
            call = "DEGRADE (clean-yes CI-separated below krea2)"
        verdicts["krea2_comparison_call"] = call

    out["verdicts"] = verdicts
    outp = os.path.join(DATA, "cells3_summary.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")
    if verdicts:
        print("\n=== verdicts ===")
        print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
