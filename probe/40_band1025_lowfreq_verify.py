#!/usr/bin/env python
"""
40_band1025_lowfreq_verify.py — validates the NEW best recipe found after
exhaustively closing the naive geometric-RoPE line (docs/design/ §3): widening
the injection band from 21-25 to 10-25 while limiting the extra early/mid
layers' position info to low-frequency-only (refpos="lowfreq", lowfreq_k=8),
instead of full content injection (which E4 found harmful for those layers
under strip) or full-precision position (which caused copy-paste instability
under "center").

Same 10-subject random sample + 2 seeds as every other gen10 variant this
session, for a fair apples-to-apples Gemini-judged comparison:
  strip (old default, band 21-25):        yes 20% / partial 60% / no 20%
  center (full RoPE, band 21-25):         yes 10% / partial 35% / no 55%
  lowfreq k=8 (band 21-25, unchanged):     yes  0% / partial 70% / no 30%
  lowfreq k=8 (band 10-25, THIS SCRIPT):  yes 60% / partial 40% / no  0%  <- new best

Run: source ../env.sh && $PY 40_band1025_lowfreq_verify.py --stage gen
     $PY 40_band1025_lowfreq_verify.py --stage score
     $PY 38_gemini_judge.py --set band1025   (separate call, reuses gemini judge)
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
OUT = os.environ.get("BAND1025_OUT", "/mnt/local/cherry_out/band1025_10")
TAGS = json.load(open(os.path.join(DATA, "gen10_tags.json")))
SUBJECTS = [u for u in TAGS if u != "11663"]
SEEDS = [0, 1]

def img_path(uid, seed):
    return f"{OUT}/u{uid}_b1025lf8_s{seed}.png"

def stage_gen():
    todo = [(u, s) for u in SUBJECTS for s in SEEDS if not os.path.exists(img_path(u, s))]
    print(f"[gen] {len(todo)} images remaining of {len(SUBJECTS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("10-25")
    for uid, seed in todo:
        prompt = TAGS[uid]["prompt_B"]
        print(f"[gen] u{uid} s{seed}", flush=True)
        im = kv.generate(uid, 4.0, refpos="lowfreq", lowfreq_k=8, sched_mode="taper",
                         seed=seed, prompt=prompt)
        im.save(img_path(uid, seed))
    print("[gen] batch done", flush=True)

def stage_score():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = {}
    same_n = 0; n = 0
    for uid in SUBJECTS:
        paths = [p for s in SEEDS if os.path.exists(p := img_path(uid, s))]
        if not paths:
            continue
        g = m.score_group(uid, paths)
        res[uid] = g
        n += g["n"]; same_n += g["same"]
    with open(os.path.join(DATA, "band1025_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"CCIP same={same_n}/{n} ({100*same_n/n:.0f}%)")
    print("(final identity judgment: use 38_gemini_judge.py --set band1025, "
          "CCIP alone is known to overestimate per docs/design §7)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    a = ap.parse_args()
    if a.stage == "gen":
        stage_gen()
    else:
        stage_score()
