#!/usr/bin/env python
"""
35_t1_pregate.py — Wave-2 T1 pre-gate: is CCIP sensitive enough to a SINGLE
slot flip to serve as the panel's accept/reject signal, or is the effect size
inside seed noise (in which case wd14 tag-match must carry the slot-level
reward instead of CCIP)? Pre-registered in docs/design/2026-07-02_wave1_exec_design.md
route to Wave-2 T1 (README §4 Wave 2, "1h 标定单槽翻转").

Design: one subject (u4238, ref=1girl/long/black hair), same KV recipe
(band 21-25, taper, w4) and identical caption template, differing ONLY in the
gender slot ("a young woman" vs "a young man"). 8 seeds each (matches T0 scale).

Gate criterion:
  delta = |mean(correct) - mean(wrong)|
  noise = max(consistency(correct), consistency(wrong))   # seed-to-seed CCIP spread
  delta >= 2*noise  -> CCIP alone can judge a single-slot flip -> keep CCIP as
                       primary panel reward, wd14 stays confirmatory.
  delta <  2*noise  -> CCIP too coarse for slot-level reward -> wd14 tag-match
                       must be promoted to primary signal for slot accept/reject.

Run: source ../env.sh && $PY 35_t1_pregate.py --stage gen
     $PY 35_t1_pregate.py --stage score
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
UID = "4238"
SEEDS = list(range(8))
PREOUT = os.environ.get("PREOUT", "/mnt/local/cherry_out/t1pregate")
TEMPLATE = ("solo, single character, {ident}, centered close-up anime portrait, "
            "upper body, detailed face, clean anime style, simple plain white background")
ARMS = {"correct": "a young woman, with black long hair",
        "wrong": "a young man, with black long hair"}  # single slot flipped: gender only

def img_path(arm, seed):
    return f"{PREOUT}/u{UID}_pregate_{arm}_s{seed}.png"

def stage_gen():
    todo = [(a, s) for a in ARMS for s in SEEDS if not os.path.exists(img_path(a, s))]
    print(f"[gen] {len(todo)} images remaining of {len(ARMS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(PREOUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("21-25")
    for arm, seed in todo:
        prompt = TEMPLATE.format(ident=ARMS[arm])
        print(f"[gen] arm={arm} seed={seed} prompt={prompt!r}", flush=True)
        im = kv.generate(UID, 4.0, sched_mode="taper", seed=seed, prompt=prompt)
        im.save(img_path(arm, seed))
    print("[gen] batch done", flush=True)

def stage_score():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = {}
    for arm in ARMS:
        paths = [p for s in SEEDS if os.path.exists(p := img_path(arm, s))]
        res[arm] = m.score_group(UID, paths)
    with open(os.path.join(DATA, "t1_pregate_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\narm     | same/n | mean CCIP | pairwise consistency (seed noise proxy)")
    for arm, g in res.items():
        print(f"{arm:8s}| {g['same']}/{g['n']}   | {g['mean']:.4f}    | {g['consistency']:.4f}")
    delta = abs(res["correct"]["mean"] - res["wrong"]["mean"])
    noise = max(res["correct"]["consistency"], res["wrong"]["consistency"])
    print(f"\ndelta(correct,wrong) = {delta:.4f}   2*noise = {2*noise:.4f}")
    verdict = "CCIP sensitive enough -> keep as primary panel reward" if delta >= 2 * noise \
        else "CCIP inside seed noise -> promote wd14 tag-match to primary slot reward"
    print(f"gate verdict: {verdict}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    a = ap.parse_args()
    if a.stage == "gen":
        stage_gen()
    else:
        stage_score()
