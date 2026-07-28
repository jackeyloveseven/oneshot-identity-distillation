#!/usr/bin/env python
"""
37_ensemble_verify.py — ensemble ID verification, adapted from WithAnyone/
InfiniteYou's evaluation methodology (researched 2026-07-04).

Their literal metrics (ArcFace/FaceNet/AdaFace face-recognition embeddings +
RetinaFace detector) do NOT transfer here: those backbones are trained only on
photorealistic human faces and are undefined on anime art. Empirically checked
the one general-purpose alternative in our toolkit (LPIPS, imgutils.metrics)
too: whole-image LPIPS on u4238's T1-pregate correct(0.606)/wrong(0.641) arms
barely separates ground-truth-correct from ground-truth-wrong (both >>0.45
"similar" threshold) because ref/gen pose+background differ -- LPIPS is
composition-dominated here, not usable as a drop-in identity signal.

What DOES transfer is the METHODOLOGY: ensemble independent signals instead of
trusting one embedding (WithAnyone averages 3 face backbones). Our in-domain
equivalent: CCIP (holistic character likeness, blind to hair-color/gender per
1c and Wave-2-T1-pregate findings) + wd14 tag re-detection on the GENERATED
image (explicit, composition-robust, catches exactly what CCIP misses).

Combined verdict: TRUE_SAME := CCIP-SAME AND all expected structural tags
re-detected in the generation at >=0.35 confidence (looser than the 0.5
extraction threshold, since KV/text jointly produced pixels, not a copy).

Run: source ../env.sh && $PY 37_ensemble_verify.py --set gen10
     $PY 37_ensemble_verify.py --set bv2
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
GOUT = "/mnt/local/cherry_out/gen10"
V2OUT = "/mnt/local/cherry_out/t0v2"
WD14_VERIFY_THR = 0.35

def tag_hits(uid, img_path, expected_struct):
    from imgutils.tagging import get_wd14_tags
    _, general, _ = get_wd14_tags(img_path, model_name="SwinV2_v3",
                                  general_threshold=WD14_VERIFY_THR)
    hits = {t: round(float(general.get(t, 0.0)), 3) for t in expected_struct}
    missing = [t for t, c in hits.items() if c < WD14_VERIFY_THR]
    return hits, missing

def run_gen10():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    seeds = range(4)
    print(f"{'uid':8s}| ccip_same | wd14_missing_any | TRUE_SAME")
    n = 0; true_same = 0; ccip_same_n = 0
    for uid, rec in tags.items():
        if uid == "11663":
            continue  # excluded multi-char ref, not a valid single-identity test
        expected = rec["struct"]
        for s in seeds:
            p = f"{GOUT}/u{uid}_gen10_B_s{s}.png"
            if not os.path.exists(p):
                continue
            n += 1
            ccip = m.ccip_diff(m.ref_path(uid), p)
            ccip_same = ccip < m.SAME_THR
            hits, missing = tag_hits(uid, p, expected)
            true = ccip_same and not missing
            ccip_same_n += int(ccip_same); true_same += int(true)
            flag = "" if true == ccip_same else "  <-- CCIP/wd14 disagree"
            miss_s = ",".join(missing) if missing else "-"
            print(f"u{uid:6s} s{s} | {ccip:.3f} {'SAME' if ccip_same else 'DIFF':4s} | "
                  f"{miss_s:30s} | {'TRUE' if true else 'false'}{flag}")
    print(f"\ntotal n={n}, CCIP-SAME={ccip_same_n}/{n} ({100*ccip_same_n/n:.0f}%), "
          f"TRUE_SAME(CCIP+wd14)={true_same}/{n} ({100*true_same/n:.0f}%)")

def run_bv2():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tags = json.load(open(os.path.join(DATA, "t0_tags.json")))
    print(f"{'uid':8s}| ccip_same | wd14_missing_any | TRUE_SAME")
    for uid in ["18522", "29998"]:
        expected = tags[uid]["struct"]
        n = 0; true_same = 0; ccip_same_n = 0
        for s in range(8):
            p = f"{V2OUT}/u{uid}_bv2_s{s}.png"
            if not os.path.exists(p):
                continue
            n += 1
            ccip = m.ccip_diff(m.ref_path(uid), p)
            ccip_same = ccip < m.SAME_THR
            hits, missing = tag_hits(uid, p, expected)
            true = ccip_same and not missing
            ccip_same_n += int(ccip_same); true_same += int(true)
            flag = "" if true == ccip_same else "  <-- CCIP/wd14 disagree"
            miss_s = ",".join(missing) if missing else "-"
            print(f"u{uid:6s} s{s} | {ccip:.3f} {'SAME' if ccip_same else 'DIFF':4s} | "
                  f"{miss_s:30s} | {'TRUE' if true else 'false'}{flag}")
        print(f"  u{uid}: CCIP-SAME={ccip_same_n}/{n}, TRUE_SAME(CCIP+wd14)={true_same}/{n}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["gen10", "bv2"])
    a = ap.parse_args()
    if a.set == "gen10":
        run_gen10()
    else:
        run_bv2()
