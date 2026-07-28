#!/usr/bin/env python
"""
65_cleanmid_report.py -- confound-free "clean mid" arm analysis for the
band-position ablation follow-up.

Background (see probe/64_ho30_band_replication_report.py + the ho30
replication report): the PUBLISHED mid arm (run_mix1/step_4000, rep30
defect 3.3% per-image / 6.7% clustered) carried two confounds vs the other
three ablation arms: (a) trained on train_manifest_mix2.jsonl, which lacks
the ref_caption field mix3 supplies to the other arms (45_train_lora.py
falls back to a generic constant), and (b) its published images were
generated WITH the "In an anime_2d style, " prompt prefix while the other
arms used raw prompt_B. The clean-mid arm removes both: checkpoint
run_mix2/step_4000 (band 10-25 r8 alpha8, lr 1e-4 bs4 uniform-t on
train_manifest_mix3.jsonl -- recipe-verified twin of the ablation arms; the
6000-vs-4000 planned-steps difference cannot alter the step-4000 state
because the LR is constant/unscheduled and the data order is a fixed
Random(0) stream), generated with raw prompt_B on BOTH 30-subject sets.

Question this script answers (coordinator's framing, verbatim intent):
was published mid's rep30 defect advantage substantially an artifact of the
prefix/manifest confounds (clean-mid rep30 defect >> 3.3%), or does a
manifest-matched prefix-free mid still hit ~3-7% on rep30 (advantage real,
confound cosmetic)?

Selftest discipline (run --selftest first): the same stats functions used
on the new cleanmid files must reproduce (1) the published mid rep30
numbers (per-image defect 3.3%, clean-yes 21.7%, clustered defect 6.7%
[0,17]) and (2) this week's fresh mid ho30 numbers (per-image defect 20.0%,
clean-yes 18.3%, clustered 36.7% [20,53]) from their JSONs exactly. All
stats helpers are importlib-reused from probe/64 (which itself selftests
against the published clustered numbers) -- not re-derived.

Run:
  $PY probe/65_cleanmid_report.py --selftest
  $PY probe/65_cleanmid_report.py            # writes data/cleanmid_summary.json
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

_spec = importlib.util.spec_from_file_location(
    "r64", os.path.join(HERE, "64_ho30_band_replication_report.py"))
r64 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r64)  # defs only; its main() is __main__-guarded

FILES = {
    # label: (filename, role)
    "published_mid_rep30": "gemini_judge_devprobe_base_rep30_step4000.json",   # run_mix1, mix2-manifest, prefix protocol
    "fresh_mid_ho30":      "gemini_judge_ho30_band_10-25_r8_step4000.json",    # run_mix1, prefix-free protocol (this week's replication)
    "cleanmid_rep30":      "gemini_judge_cleanmid_rep30_band_10-25_r8_step4000.json",  # run_mix2, mix3-manifest, prefix-free
    "cleanmid_ho30":       "gemini_judge_cleanmid_ho30_band_10-25_r8_step4000.json",
}


def stats_for(fname):
    d, path = r64.load(fname)
    if d is None:
        return None
    return {
        "file": fname,
        "per_image": r64.per_image_stats(list(d.values())),
        "clustered_defect": r64.clustered_bootstrap_defect(d),
    }


def fmt(s):
    if s is None:
        return "MISSING"
    cy = s["per_image"]["clean_yes"]["pct_ci"]
    df = s["per_image"]["defect"]["pct_ci"]
    cc = s["clustered_defect"]["ci95_pct"]
    return (f"n={s['per_image']['n']} err={s['per_image']['judge_errors']}  "
            f"clean-yes={cy[0]:5.1f}% [{cy[1]:.0f},{cy[2]:.0f}]  "
            f"defect={df[0]:5.1f}% [{df[1]:.0f},{df[2]:.0f}]  "
            f"clustered-defect={s['clustered_defect']['point_pct']:5.1f}% [{cc[0]:.0f},{cc[1]:.0f}]")


def selftest():
    """Exact-match bars on point estimates for the two reference files."""
    expect = {
        "published_mid_rep30": {"defect": 3.3, "clean_yes": 21.7, "clustered": 6.7},
        "fresh_mid_ho30":      {"defect": 20.0, "clean_yes": 18.3, "clustered": 36.7},
    }
    ok = True
    for label, exp in expect.items():
        s = stats_for(FILES[label])
        if s is None:
            print(f"SELFTEST SKIP {label}: file missing")
            ok = False
            continue
        got = {
            "defect": s["per_image"]["defect"]["pct_ci"][0],
            "clean_yes": s["per_image"]["clean_yes"]["pct_ci"][0],
            "clustered": s["clustered_defect"]["point_pct"],
        }
        row_ok = all(abs(got[k] - exp[k]) < 0.05 for k in exp)
        ok = ok and row_ok
        print(f"SELFTEST {label:22s} {'OK      ' if row_ok else 'MISMATCH'} "
              + "  ".join(f"{k}={got[k]:.1f} (exp {exp[k]})" for k in exp))
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
        "purpose": ("confound-free mid (run_mix2/step_4000: mix3 manifest + raw prompt_B) vs "
                    "published confounded mid (run_mix1/step_4000: mix2 manifest, prefix protocol on rep30) "
                    "and vs fresh mid ho30 (run_mix1, prefix-free)"),
        "recipe_verification": ("run_mix2 == ablation-arm recipe: band 10-25 r8 alpha8, lr 1e-4 constant "
                                "(no scheduler -> step-4000 state independent of 6000-step plan), bs 4, "
                                "uniform-t, train_manifest_mix3.jsonl 5936 rows, single continuous run"),
        "boot_n": r64.BOOT_N, "boot_seed": r64.BOOT_SEED, "wilson_z": 1.96,
    }, "arms": {}}

    print(f"{'arm':22s}  stats")
    for label, fname in FILES.items():
        s = stats_for(fname)
        out["arms"][label] = s
        print(f"{label:22s}  {fmt(s)}")

    cm_rep = out["arms"]["cleanmid_rep30"]
    cm_ho = out["arms"]["cleanmid_ho30"]
    pub = out["arms"]["published_mid_rep30"]
    fmh = out["arms"]["fresh_mid_ho30"]
    verdicts = {}

    if cm_rep and pub:
        d_clean = cm_rep["per_image"]["defect"]["pct_ci"][0]
        d_pub = pub["per_image"]["defect"]["pct_ci"][0]
        verdicts["cleanmid_rep30_defect_pct"] = d_clean
        verdicts["published_mid_rep30_defect_pct"] = d_pub
        verdicts["cleanmid_rep30_within_3_7pct"] = 3.0 <= d_clean <= 7.0
        verdicts["cleanmid_vs_published_clustered_overlap"] = r64.ci_overlap(
            cm_rep["clustered_defect"]["ci95_pct"], pub["clustered_defect"]["ci95_pct"])
        # blunt call per the pre-agreed framing: >>3.3% (and clustered CIs
        # separated) = advantage was substantially confound artifact;
        # ~3-7% per-image = advantage real, confound cosmetic; between = mixed.
        if verdicts["cleanmid_rep30_within_3_7pct"]:
            call = "ADVANTAGE REAL, CONFOUND COSMETIC (clean-mid rep30 defect still in ~3-7% band)"
        elif not verdicts["cleanmid_vs_published_clustered_overlap"]:
            call = "ADVANTAGE SUBSTANTIALLY CONFOUND ARTIFACT (clean-mid rep30 defect >> published 3.3%, clustered CIs disjoint)"
        else:
            call = ("MIXED / INCONCLUSIVE (clean-mid rep30 defect above the 3-7% band but clustered CIs "
                    "still overlap published mid's)")
        verdicts["confound_verdict"] = call

        cy_overlap = r64.ci_overlap(tuple(cm_rep["per_image"]["clean_yes"]["pct_ci"][1:]),
                                     tuple(pub["per_image"]["clean_yes"]["pct_ci"][1:]))
        verdicts["cleanmid_vs_published_cleanyes_perimage_overlap"] = cy_overlap

    if cm_ho and fmh:
        verdicts["cleanmid_ho30_defect_pct"] = cm_ho["per_image"]["defect"]["pct_ci"][0]
        verdicts["fresh_mid_ho30_defect_pct"] = fmh["per_image"]["defect"]["pct_ci"][0]
        verdicts["cleanmid_vs_freshmid_ho30_clustered_overlap"] = r64.ci_overlap(
            cm_ho["clustered_defect"]["ci95_pct"], fmh["clustered_defect"]["ci95_pct"])

    out["verdicts"] = verdicts
    outp = os.path.join(DATA, "cleanmid_summary.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")
    if verdicts:
        print("\n=== verdicts ===")
        for k, v in verdicts.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
