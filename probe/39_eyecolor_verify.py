#!/usr/bin/env python
"""
39_eyecolor_verify.py — verify the EYE_COLOR whitelist fix (31_tagprior.py,
2026-07-04, added after Gemini 3.1 Pro judge flagged eye color/shape mismatch
on 29/40 gen10 random-sample images despite CCIP+wd14 ensemble saying 40/40
TRUE_SAME). Same pattern as 34_haircolor_verify.py's hair-color fix validation.

wd14 confidently detected eye color (>=0.5) on 5 of the 10 valid gen10 subjects
that lacked it before (4229, 13153, 29235, 19098, 12523) -- regenerate arm B
for all 10 with the updated (eye-color-aware) prompt_B, re-score with CCIP +
Gemini, compare against the v1 (no eye color) judge results in
data/gemini_judge_gen10.json.

Run: source ../env.sh && $PY 39_eyecolor_verify.py --stage gen
     $PY 39_eyecolor_verify.py --stage score
     $PY 38_gemini_judge.py --set gen10v2   (separate: reuses gemini judge on new set)
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
V2OUT = os.environ.get("V2OUT", "/mnt/local/cherry_out/gen10v2")
TAGS = json.load(open(os.path.join(DATA, "gen10_tags.json")))
SUBJECTS = [u for u in TAGS if u != "11663"]
SEEDS = list(range(4))

def img_path(uid, seed):
    return f"{V2OUT}/u{uid}_gen10v2_s{seed}.png"

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
    res = {}
    for uid in SUBJECTS:
        paths = [p for s in SEEDS if os.path.exists(p := img_path(uid, s))]
        res[uid] = m.score_group(uid, paths)
    with open(os.path.join(DATA, "gen10v2_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nuid    | CCIP same/mean | has eye_color in prompt_B?")
    for uid, g in res.items():
        has_eye = any(k in TAGS[uid]["struct"] for k in
                      ["blue_eyes","red_eyes","green_eyes","brown_eyes","yellow_eyes",
                       "purple_eyes","black_eyes","grey_eyes","pink_eyes","orange_eyes",
                       "amber_eyes","aqua_eyes","heterochromia"])
        print(f"u{uid:6s}| {g['same']}/{g['n']} {g['mean']:.3f}    | {'yes' if has_eye else 'no'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score"])
    a = ap.parse_args()
    if a.stage == "gen":
        stage_gen()
    else:
        stage_score()
