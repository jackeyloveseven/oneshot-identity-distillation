#!/usr/bin/env python
"""
33_dualcfg.py — Wave-1 1c, re-targeted at the two T0 anomalies (not the original
plan's two "clean failures"): u18522 (regressed under tag-prior) and u29998
(unreliable KV color-transfer across seeds, mistakenly filed as a CCIP-occlusion
artifact until broader seed inspection showed real identity drift on 5/8 seeds).

Both anomalies trace to the SAME root cause discovered while building this script:
wd14 struct whitelist (31_tagprior.py) has NO hair-color category. Hair color was
always assumed to be KV's job (position-free appearance), never written to text.
  - u18522: writing "dark-skinned" (skin IS in the whitelist) pulled a correlated
    dark-skin -> black-hair prior even though hair color was never mentioned.
    A (KV-only) already anchors true brown hair correctly (0.134 best-in-cohort).
  - u29998: white/lavender hair is a rare anime color; KV transfers it in only
    3/8 seeds (s0,s3,s4), other seeds fall back to the base model's common
    brown/black hair prior with NO textual trigger at all -- real stochastic
    KV failure, not a metric artifact (verified by reading s1/s3/s5 vs ref).

Three-way attribution per subject (pre-registered, see docs/design/...):
  fixed by caption edit alone (cfg_id=0)      -> prior suppression   -> text line
  fixed only when cfg_id>0 (caption unchanged)-> channel info suppressed -> guidance line
  neither fixes it                             -> channel/seed defect -> addressing (1b) / accept

Cases (16 new images, reuses T0 B-arm cfg_id=0 for the untouched baseline):
  18522 (seeds 0,1):
    noskin_cfg0   : drop "dark-skinned", keep "short hair"          x2
    full_cfg{2,4,8}: keep "dark-skinned" as-is, sweep cfg_id        x2 each
  29998 (seeds 1,5 -- the two seeds T0 arm B got DIFF on):
    haircolor_cfg0: add "with white hair" to caption, cfg_id=0      x2
    full_cfg{2,4,8}: caption unchanged, sweep cfg_id                x2 each

Run: source ../env.sh && $PY 33_dualcfg.py --stage gen
     $PY 33_dualcfg.py --stage score
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
C1OUT = os.environ.get("C1OUT", "/mnt/local/cherry_out/c1")
T0TAGS = json.load(open(os.path.join(DATA, "t0_tags.json")))

full_18522 = T0TAGS["18522"]["prompt_B"]
noskin_18522 = full_18522.replace("a dark-skinned young man", "a young man")
assert noskin_18522 != full_18522

full_29998 = T0TAGS["29998"]["prompt_B"]
haircolor_29998 = full_29998.replace("a young man, with short hair",
                                      "a young man, with white hair, short hair")
assert haircolor_29998 != full_29998

def build_cases():
    cases = []
    for seed in [0, 1]:
        cases.append(dict(uid="18522", seed=seed, cfg_id=0.0, prompt=noskin_18522, key="noskin_cfg0"))
    for cfg_id in [2.0, 4.0, 8.0]:
        for seed in [0, 1]:
            cases.append(dict(uid="18522", seed=seed, cfg_id=cfg_id, prompt=full_18522, key=f"full_cfg{cfg_id:g}"))
    for seed in [1, 5]:
        cases.append(dict(uid="29998", seed=seed, cfg_id=0.0, prompt=haircolor_29998, key="haircolor_cfg0"))
    for cfg_id in [2.0, 4.0, 8.0]:
        for seed in [1, 5]:
            cases.append(dict(uid="29998", seed=seed, cfg_id=cfg_id, prompt=full_29998, key=f"full_cfg{cfg_id:g}"))
    return cases

def img_path(c):
    return f"{C1OUT}/u{c['uid']}_c1_{c['key']}_s{c['seed']}.png"

def stage_gen():
    cases = build_cases()
    todo = [c for c in cases if not os.path.exists(img_path(c))]
    print(f"[gen] {len(todo)} images remaining of {len(cases)}", flush=True)
    if not todo:
        return
    os.makedirs(C1OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("21-25")
    for c in todo:
        print(f"[gen] u{c['uid']} key={c['key']} seed={c['seed']} cfg_id={c['cfg_id']}", flush=True)
        im = kv.generate(c["uid"], 4.0, sched_mode="taper", seed=c["seed"],
                         prompt=c["prompt"], cfg_id=c["cfg_id"])
        im.save(img_path(c))
    print("[gen] batch done", flush=True)

def stage_score():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cases = build_cases()
    by_key = {}
    for c in cases:
        by_key.setdefault((c["uid"], c["key"]), []).append(c)
    res = {}
    for (uid, key), cs in by_key.items():
        paths = [img_path(c) for c in cs if os.path.exists(img_path(c))]
        if not paths:
            continue
        diffs = {os.path.basename(p): m.ccip_diff(m.ref_path(uid), p) for p in paths}
        same = sum(1 for d in diffs.values() if d < m.SAME_THR)
        res[f"{uid}/{key}"] = {"diffs": diffs, "n": len(diffs), "same": same}
    with open(os.path.join(DATA, "c1_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nuid/key                 | same/n | per-image ccip")
    for k, g in res.items():
        print(f"{k:24s}| {g['same']}/{g['n']}   | " +
              ", ".join(f"{v:.3f}" for v in g["diffs"].values()))
    print("\n--- attribution ---")
    print("18522 baseline (T0 arm B, full_cfg0) = 1/8 SAME (regressed from A's 6/8)")
    n_ns = res.get("18522/noskin_cfg0", {}).get("same", 0)
    print(f"  noskin_cfg0 same={n_ns}/2  -> {'PRIOR SUPPRESSION (text line)' if n_ns == 2 else 'not clean text fix'}")
    for cfg_id in [2.0, 4.0, 8.0]:
        g = res.get(f"18522/full_cfg{cfg_id:g}", {})
        print(f"  full_cfg{cfg_id:g} same={g.get('same','-')}/2")
    print("29998 baseline (T0 arm B, seeds 1&5, full_cfg0) = 0/2 SAME (real drift, not occlusion artifact)")
    n_hc = res.get("29998/haircolor_cfg0", {}).get("same", 0)
    print(f"  haircolor_cfg0 same={n_hc}/2 -> {'TEXT FIX WORKS (add hair color to whitelist)' if n_hc == 2 else 'not clean text fix'}")
    for cfg_id in [2.0, 4.0, 8.0]:
        g = res.get(f"29998/full_cfg{cfg_id:g}", {})
        print(f"  full_cfg{cfg_id:g} same={g.get('same','-')}/2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    a = ap.parse_args()
    if a.stage == "gen":
        stage_gen()
    else:
        stage_score()
