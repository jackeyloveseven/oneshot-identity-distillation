#!/usr/bin/env python
"""
64_ho30_band_replication_report.py -- cross-set replication check for the
training-light band-position ablation keystone (see
docs/paper/section_trainlight_band_ablation.md). The published ablation
(probe/58_band_ablation_run.sh + probe/59_band_ablation_report.py) ran all
four arms (early 0-15 r8, mid 10-25 r8, late 14-29 r8, full-depth 0-29 r4)
on ONE 30-subject dev set (repprobe30_tags.json / "rep30"). This script
answers: does the mid-vs-full-depth defect-rate separation replicate on the
OTHER held-out 30-subject set (holdout30_tags.json / "ho30"), using the
SAME four pre-existing checkpoints (no training), the SAME gen harness
(probe/46_devprobe.py --stage gen --arm lora), and the SAME judge
instrument (probe/38_gemini_judge.py --set devprobe_lora, HOLDOUT_PROMPT)?

Two statistics per arm per set, matching docs/paper/data/README.md's
"Subject-clustered CI recompute" section exactly:
  (a) per-image Wilson 95% CI (n=60 = 30 subjects x 2 seeds), for both
      clean-yes (same_character=="yes" and visible_defect!="yes") and
      visible_defect=="yes" -- same formula as probe/59_band_ablation_report.py
      (z=1.96), reused verbatim, not re-derived.
  (b) subject-clustered bootstrap for defect only: collapse each subject's
      2 seeds to one "any-defect" indicator (1 if either seed's
      visible_defect=="yes"), resample subjects with replacement (n=30,
      20000 resamples), report the mean point estimate and the [2.5,97.5]
      percentile interval. This script's own bootstrap is validated against
      the published rep30 numbers before being trusted on the new ho30 data
      (see --selftest).

Run:
  $PY probe/64_ho30_band_replication_report.py --selftest   # validates the
      clustered-bootstrap function reproduces the published rep30 mid/late/
      full-depth numbers (6.7% [0,17] / 36.7% [20,53] / 43.3% [27,60]) before
      trusting it on fresh data -- run this first.
  $PY probe/64_ho30_band_replication_report.py               # full report:
      rep30 (reference) vs ho30 (replication) side by side for all 4 arms,
      writes data/ho30_band_replication_summary.json
"""
import argparse
import json
import math
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

ARMS = [
    # (label, band, r, rep30 (published) json filename, ho30 (this replication) json filename)
    ("early",      "0-15",  8, "gemini_judge_band_0-15_r8_step4000.json",
                                "gemini_judge_ho30_band_0-15_r8_step4000.json"),
    ("mid",        "10-25", 8, "gemini_judge_devprobe_base_rep30_step4000.json",
                                "gemini_judge_ho30_band_10-25_r8_step4000.json"),
    ("late",       "14-29", 8, "gemini_judge_band_14-29_r8_step4000.json",
                                "gemini_judge_ho30_band_14-29_r8_step4000.json"),
    ("full-depth", "0-29",  4, "gemini_judge_band_0-29_r4_step4000.json",
                                "gemini_judge_ho30_band_0-29_r4_step4000.json"),
]

PID_RE = re.compile(r"^(\d+)_lora_.*_s([01])$")

BOOT_N = 20000
BOOT_SEED = 2026  # frozen for reproducibility of this script's own numbers


# ---------------------------------------------------------------------------
# stats primitives
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Verbatim from probe/59_band_ablation_report.py -- do not re-derive."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p * 100, max(0.0, (center - half)) * 100, min(1.0, (center + half)) * 100)


def per_image_stats(entries):
    n = len(entries)
    defect = sum(1 for e in entries if e.get("visible_defect") == "yes")
    clean_yes = sum(1 for e in entries if e.get("same_character") == "yes" and e.get("visible_defect") != "yes")
    clean_usable = sum(1 for e in entries if e.get("same_character") in ("yes", "partial") and e.get("visible_defect") != "yes")
    # judge-error entries (same_character=="error", no visible_defect field):
    # counted in the denominator as neither clean-yes nor defect -- this
    # matches probe/59_band_ablation_report.py's handling, and the PUBLISHED
    # rep30 numbers themselves contain such entries (mid 3, early 1, late 2,
    # full-depth 0 of 60), so keeping the identical convention is required
    # for cross-set comparability. Count disclosed here for transparency.
    errors = sum(1 for e in entries if e.get("same_character") == "error")
    return {
        "n": n,
        "judge_errors": errors,
        "clean_yes": {"k": clean_yes, "pct_ci": wilson(clean_yes, n)},
        "defect": {"k": defect, "pct_ci": wilson(defect, n)},
        "clean_usable": {"k": clean_usable, "pct_ci": wilson(clean_usable, n)},
    }


def parse_pid(pid):
    m = PID_RE.match(pid)
    if not m:
        raise ValueError(f"pid does not match expected '<uid>_lora_..._s<0|1>' shape: {pid!r}")
    return m.group(1), int(m.group(2))


def subject_any_defect(d):
    """d: the raw judge dict {pid: verdict}. Returns {uid: 0/1} where 1 means
    at least one of that subject's seeds was flagged visible_defect=='yes'.
    Asserts exactly 2 seeds (0,1) seen per uid and no duplicate (uid,seed) --
    a silent partial-write or duplicate-key bug would otherwise corrupt the
    clustering silently."""
    by_uid = {}
    seen = set()
    for pid, verdict in d.items():
        uid, seed = parse_pid(pid)
        key = (uid, seed)
        if key in seen:
            raise ValueError(f"duplicate (uid,seed) {key} in judge file -- refusing to cluster silently")
        seen.add(key)
        by_uid.setdefault(uid, {})[seed] = verdict.get("visible_defect") == "yes"
    out = {}
    incomplete = []
    for uid, seeds in by_uid.items():
        if set(seeds.keys()) != {0, 1}:
            incomplete.append((uid, sorted(seeds.keys())))
            continue
        out[uid] = 1 if (seeds[0] or seeds[1]) else 0
    return out, incomplete


def clustered_bootstrap_defect(d, n_boot=BOOT_N, seed=BOOT_SEED):
    any_defect, incomplete = subject_any_defect(d)
    uids = sorted(any_defect.keys())
    n = len(uids)
    vals = [any_defect[u] for u in uids]
    point = 100.0 * sum(vals) / n if n else 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(100.0 * sum(resample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {
        "n_subjects": n,
        "any_defect_k": sum(vals),
        "point_pct": point,
        "ci95_pct": (lo, hi),
        "incomplete_subjects": incomplete,  # should be [] -- non-empty means n!=60 for that arm
    }


# ---------------------------------------------------------------------------

def load(fname):
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        return None, path
    return json.load(open(path)), path


def ci_overlap(ci_a, ci_b):
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return not (lo_a > hi_b or lo_b > hi_a)


def judge_drift_block():
    """Sanity-anchor decomposition (added on coordinator request): the OLD
    production-mid holdout30 eval (data/gemini_judge_base_holdout30_step4000
    .json, judged 2026-07-13, images generated from the SAME checkpoint
    run_mix1/step_4000 on the SAME 30 subjects x seeds {0,1} but WITH the
    'In an anime_2d style, ' prompt prefix) reported 13.3% defect; this
    replication's fresh mid reads 20.0%. To split that delta into judge
    drift vs generation-protocol change, the old images (still on disk at
    /mnt/local/cherry_out/holdout30/base_step4000/, pixel-identical to what
    the old JSON judged) were RE-judged 2026-07-16 with the identical
    instrument (38_gemini_judge.py --set devprobe_lora, HOLDOUT_PROMPT) ->
    data/gemini_judge_ho30_mid_oldimages_rejudge.json. Returns None if that
    file is absent."""
    old, _ = load("gemini_judge_base_holdout30_step4000.json")
    rej, _ = load("gemini_judge_ho30_mid_oldimages_rejudge.json")
    if old is None or rej is None or set(old.keys()) != set(rej.keys()):
        return None
    n = len(old)
    yy = sum(1 for k in old if (old[k].get("visible_defect") == "yes") and (rej[k].get("visible_defect") == "yes"))
    yn = sum(1 for k in old if (old[k].get("visible_defect") == "yes") and (rej[k].get("visible_defect") != "yes"))
    ny = sum(1 for k in old if (old[k].get("visible_defect") != "yes") and (rej[k].get("visible_defect") == "yes"))
    nn = n - yy - yn - ny
    ocy = sum(1 for v in old.values() if v.get("same_character") == "yes" and v.get("visible_defect") != "yes")
    rcy = sum(1 for v in rej.values() if v.get("same_character") == "yes" and v.get("visible_defect") != "yes")
    return {
        "what": "same 60 pixels (old mid holdout30 images), judged 2026-07-13 (old) vs 2026-07-16 (rejudge)",
        "n": n,
        "defect_flip_matrix": {"both_yes": yy, "old_only": yn, "rejudge_only": ny, "both_no": nn},
        "defect_agreement_pct": 100.0 * (yy + nn) / n,
        "defect_old_pct": 100.0 * (yy + yn) / n,
        "defect_rejudge_pct": 100.0 * (yy + ny) / n,
        "clean_yes_old_pct": 100.0 * ocy / n,
        "clean_yes_rejudge_pct": 100.0 * rcy / n,
        "clustered_defect_old": clustered_bootstrap_defect(old),
        "clustered_defect_rejudge": clustered_bootstrap_defect(rej),
        "interpretation": (
            "judge drift on identical pixels is small and downward "
            "(13.3%->10.0% defect, 93% per-image agreement, clean-yes identical at 21.7%); "
            "the fresh-mid 20.0% vs old 13.3% anchor delta is therefore dominated by the "
            "generation-protocol change (this replication drops the 'In an anime_2d style, ' "
            "prefix so all four arms share one protocol), not instrument drift"),
    }


def selftest():
    """Validates clustered_bootstrap_defect() against the three published
    rep30 numbers quoted in section_trainlight_band_ablation.md's caveats:
    mid 6.7% [0,17], late 36.7% [20,53], full-depth 43.3% [27,60]. Point
    estimates must match exactly (deterministic count, not a bootstrap
    artifact); CI bounds are bootstrap-random so are checked with a wide
    tolerance (+/-6pp) against the published rounded values, since the
    published CI was computed ad hoc (docs/paper/data/README.md: "computed
    ad hoc ... No separate output file was saved") with an unknown RNG seed
    -- only the point estimate is an exact-match bar."""
    published = {
        "mid": {"point": 6.7, "ci": (0, 17)},
        "late": {"point": 36.7, "ci": (20, 53)},
        "full-depth": {"point": 43.3, "ci": (27, 60)},
    }
    ok = True
    for label, band, r, rep_fname, _ho30_fname in ARMS:
        if label not in published:
            continue
        d, path = load(rep_fname)
        if d is None:
            print(f"SELFTEST SKIP {label}: {path} not found")
            ok = False
            continue
        res = clustered_bootstrap_defect(d)
        exp = published[label]
        pt_match = abs(res["point_pct"] - exp["point"]) < 0.05
        ci_close = (abs(res["ci95_pct"][0] - exp["ci"][0]) < 6 and
                    abs(res["ci95_pct"][1] - exp["ci"][1]) < 6)
        status = "OK" if pt_match else "MISMATCH"
        print(f"SELFTEST {label:12s} point={res['point_pct']:.1f}% (published {exp['point']}%) "
              f"[{status}]  CI=[{res['ci95_pct'][0]:.0f},{res['ci95_pct'][1]:.0f}] "
              f"(published [{exp['ci'][0]},{exp['ci'][1]}], {'close' if ci_close else 'DIFFERS (expected: ad hoc unknown seed)'})"
              f"  n_subjects={res['n_subjects']} incomplete={res['incomplete_subjects']}")
        if not pt_match:
            ok = False
    print("\nSELFTEST", "PASSED (point estimates reproduce exactly)" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                     help="validate clustered-bootstrap against published rep30 numbers, then exit")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    report = {"_meta": {
        "purpose": "holdout30 cross-set replication of the training-light band-position ablation",
        "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
        "wilson_z": 1.96,
    }, "arms": {}}

    print(f"{'arm':12s} {'set':6s} {'n':>4s}  {'clean-yes (Wilson)':>22s}  {'defect (Wilson)':>20s}  {'defect (clustered, n=30)':>26s}")
    for label, band, r, rep_fname, ho30_fname in ARMS:
        rep_d, rep_path = load(rep_fname)
        ho_d, ho_path = load(ho30_fname)
        entry = {"band": band, "r": r, "rep30_file": rep_fname, "ho30_file": ho30_fname}

        for setname, d, path in (("rep30", rep_d, rep_path), ("ho30", ho_d, ho_path)):
            if d is None:
                print(f"{label:12s} {setname:6s}  MISSING ({path})")
                entry[setname] = None
                continue
            pim = per_image_stats(list(d.values()))
            clust = clustered_bootstrap_defect(d)
            entry[setname] = {"per_image": pim, "clustered_defect": clust}
            cy = pim["clean_yes"]["pct_ci"]
            df = pim["defect"]["pct_ci"]
            cc = clust["ci95_pct"]
            print(f"{label:12s} {setname:6s} {pim['n']:4d}  "
                  f"{cy[0]:5.1f}% [{cy[1]:.0f},{cy[2]:.0f}]        "
                  f"{df[0]:5.1f}% [{df[1]:.0f},{df[2]:.0f}]      "
                  f"{clust['point_pct']:5.1f}% [{cc[0]:.0f},{cc[1]:.0f}]"
                  + (f"  INCOMPLETE={clust['incomplete_subjects']}" if clust["incomplete_subjects"] else ""))
        report["arms"][label] = entry

    # verdicts, computed only if both mid and full-depth ho30 data are present
    mid = report["arms"]["mid"].get("ho30")
    fd = report["arms"]["full-depth"].get("ho30")
    verdicts = {}
    if mid and fd:
        mid_cy = mid["per_image"]["clean_yes"]["pct_ci"][1:]
        fd_cy = fd["per_image"]["clean_yes"]["pct_ci"][1:]
        verdicts["clean_yes_still_null_perimage"] = ci_overlap(tuple(mid_cy), tuple(fd_cy))

        mid_clust_ci = mid["clustered_defect"]["ci95_pct"]
        fd_clust_ci = fd["clustered_defect"]["ci95_pct"]
        verdicts["mid_vs_fulldepth_defect_clustered_overlap"] = ci_overlap(mid_clust_ci, fd_clust_ci)
        verdicts["mid_vs_fulldepth_defect_replicates"] = not verdicts["mid_vs_fulldepth_defect_clustered_overlap"]

        mid_pim_ci = mid["per_image"]["defect"]["pct_ci"][1:]
        fd_pim_ci = fd["per_image"]["defect"]["pct_ci"][1:]
        verdicts["mid_vs_fulldepth_defect_perimage_overlap"] = ci_overlap(tuple(mid_pim_ci), tuple(fd_pim_ci))

        points = {lbl: report["arms"][lbl]["ho30"]["per_image"]["defect"]["pct_ci"][0]
                  for lbl in ("early", "mid", "late", "full-depth")
                  if report["arms"][lbl].get("ho30")}
        if len(points) == 4:
            verdicts["monotonic_early_mid_le_late_le_fulldepth"] = (
                points["mid"] <= points["early"] and points["mid"] <= points["late"] <= points["full-depth"])
            verdicts["mid_is_outlier_low"] = (points["mid"] == min(points.values()))
            verdicts["point_estimates_defect_pct"] = points

    report["verdicts"] = verdicts

    drift = judge_drift_block()
    if drift is not None:
        report["judge_drift_sanity_anchor"] = drift
        print(f"\n=== judge-drift sanity anchor (same 60 old-mid pixels, old vs re-judge) ===")
        print(f"  defect: old {drift['defect_old_pct']:.1f}% -> rejudge {drift['defect_rejudge_pct']:.1f}%  "
              f"(agreement {drift['defect_agreement_pct']:.0f}%, flips {drift['defect_flip_matrix']})")
        print(f"  clean-yes: old {drift['clean_yes_old_pct']:.1f}% -> rejudge {drift['clean_yes_rejudge_pct']:.1f}%")

    outp = os.path.join(DATA, "ho30_band_replication_summary.json")
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {outp}")
    if verdicts:
        print("\n=== verdicts ===")
        for k, v in verdicts.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
