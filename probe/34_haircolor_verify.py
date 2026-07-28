#!/usr/bin/env python
"""
34_haircolor_verify.py — verify the HAIR_COLOR whitelist fix (31_tagprior.py,
2026-07-03) actually improves arm B on the two T0 anomalies, at full 8-seed
scale (1c only checked 2 seeds each).

Arm B v2 = current (post-fix) prompt_B from data/t0_tags.json (now includes
hair color). Compares against the already-recorded arm B v1 (data/t0_scores.json,
built from data/t0_tags_v1_nocolor.json): u18522 1/8 SAME, u29998 3/8 SAME.

Run: source ../env.sh && $PY 34_haircolor_verify.py --stage gen
     $PY 34_haircolor_verify.py --stage score
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
V2OUT = os.environ.get("V2OUT", "/mnt/local/cherry_out/t0v2")
SUBJECTS = ["18522", "29998"]
SEEDS = list(range(8))
TAGS = json.load(open(os.path.join(DATA, "t0_tags.json")))  # post-fix, has hair color

def img_path(uid, seed):
    return f"{V2OUT}/u{uid}_bv2_s{seed}.png"

def stage_gen():
    todo = [(u, s) for u in SUBJECTS for s in SEEDS if not os.path.exists(img_path(u, s))]
    print(f"[gen] {len(todo)} images remaining of {len(SUBJECTS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(V2OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("21-25")
    for uid, seed in todo:
        prompt = TAGS[uid]["prompt_B"]
        print(f"[gen] u{uid} s{seed} prompt={prompt!r}", flush=True)
        im = kv.generate(uid, 4.0, sched_mode="taper", seed=seed, prompt=prompt)
        im.save(img_path(uid, seed))
    print("[gen] batch done", flush=True)

def stage_score():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    v1 = json.load(open(os.path.join(DATA, "t0_scores.json")))
    res = {}
    for uid in SUBJECTS:
        paths = [p for s in SEEDS if os.path.exists(p := img_path(uid, s))]
        res[uid] = m.score_group(uid, paths)
    with open(os.path.join(DATA, "bv2_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nuid    | v1(no-color) same/mean | v2(+color) same/mean")
    for uid in SUBJECTS:
        g1 = v1[uid]["B"]; g2 = res[uid]
        print(f"u{uid:6s}| {g1['same']}/8 {g1['mean']:.3f}          | {g2['same']}/8 {g2['mean']:.3f}")
    print("\nper-seed v2 detail:")
    for uid in SUBJECTS:
        for k, v in sorted(res[uid]["diffs"].items()):
            print(f"  {k}: {v:.3f} {'SAME' if v < m.SAME_THR else 'DIFF'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    a = ap.parse_args()
    if a.stage == "gen":
        stage_gen()
    else:
        stage_score()
