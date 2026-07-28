#!/usr/bin/env python
"""
36_random_generalize.py — generalization check on a RANDOM (non-cherry-picked)
sample of subjects. Every result so far (capstone, T0, 1c, whitelist fix) was
measured on the capstone-6, which docs/design explicitly flags as hand-picked
hard cases, not a random sample. This validates whether "KV(w4,band21-25,taper)
+ wd14 tag-prior caption (post whitelist fix, includes hair color)" generalizes.

Sample: 10 subjects, python random.seed(42) over all non-capstone refs in $REF
(29251 candidates) -> ['17419','4229','13153','29235','9728','17426','11663',
'5910','19098','12523'].

Two arms (same as T0 A/B, fixed KV recipe band21-25/taper/w4):
  A = generic caption + KV        (current-recipe baseline)
  B = wd14 tag-prior caption + KV (the claimed improvement)
N_SEEDS=4 to bound cost (10 subj x 2 arms x 4 seeds = 80 imgs, ~40 GPU-min);
expand seeds later only on subjects near the rescue threshold.

Run: source ../env.sh && $PY 36_random_generalize.py --stage tags
     $PY 36_random_generalize.py --stage gen
     $PY 36_random_generalize.py --stage score
"""
import os, json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
GOUT = os.environ.get("GOUT", "/mnt/local/cherry_out/gen10")
CAPSTONE = {"10000", "18522", "20582", "29998", "4238", "9172"}
N_SEEDS = 4
SEEDS = list(range(N_SEEDS))
ARMS = ["A", "B"]
TAGS_JSON = os.path.join(DATA, "gen10_tags.json")

BAD_MULTI_TAGS = {"2boys", "2girls", "multiple_girls", "multiple_boys", "multiple_others"}

def is_solo(uid):
    """PROMPT/TEMPLATE both hard-assume a single-character ref (minefield #11,
    found via u11663: a girl+baby+man family scene wd14-tagged as both 1girl
    AND 1boy AND dark_skin AND white/black hair simultaneously -> garbage
    render_phrase output, arm A itself failed 0/4 so it's a data issue, not
    a mechanism failure). Reject refs without solo>=0.5 or with multi-char tags."""
    from imgutils.tagging import get_wd14_tags
    _, general, _ = get_wd14_tags(f"{REF}/u{uid}_ref.png", model_name="SwinV2_v3",
                                  general_threshold=0.3)
    return general.get("solo", 0.0) >= 0.5 and not (BAD_MULTI_TAGS & general.keys())

def sample_subjects():
    files = [f for f in os.listdir(REF) if f.endswith("_ref.png")]
    ids = [f[1:-8] for f in files if f[1:-8] not in CAPSTONE]
    random.seed(42)
    return random.sample(ids, 10)

def stage_tags():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", os.path.join(HERE, "31_tagprior.py"))
    tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
    from imgutils.tagging import get_wd14_tags
    subs = sample_subjects()
    print(f"[tags] sample = {subs}", flush=True)
    out = {}
    for uid in subs:
        rating, general, chars = get_wd14_tags(f"{REF}/u{uid}_ref.png",
                                               model_name="SwinV2_v3",
                                               general_threshold=tp.WD14_THR)
        struct = {t: round(float(c), 3) for t, c in general.items() if t in tp.STRUCT}
        phrase = tp.render_phrase(struct)
        out[uid] = {"struct": struct, "phrase": phrase,
                    "prompt_B": tp.TEMPLATE.format(ident=phrase)}
        print(f"u{uid}: struct={sorted(struct)} -> {phrase!r}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    with open(TAGS_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {TAGS_JSON}")

def img_path(uid, arm, seed):
    return f"{GOUT}/u{uid}_gen10_{arm}_s{seed}.png"

def stage_gen():
    tags = json.load(open(TAGS_JSON))
    todo = [(u, a, s) for u in tags for a in ARMS for s in SEEDS
            if not os.path.exists(img_path(u, a, s))]
    print(f"[gen] {len(todo)} images remaining of {len(tags)*len(ARMS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(GOUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("21-25")
    for uid, arm, seed in todo:
        prompt = None if arm == "A" else tags[uid]["prompt_B"]
        print(f"[gen] u{uid} arm={arm} seed={seed}", flush=True)
        im = kv.generate(uid, 4.0, sched_mode="taper", seed=seed, prompt=prompt)
        im.save(img_path(uid, arm, seed))
    print("[gen] batch done", flush=True)

def stage_score():
    tags = json.load(open(TAGS_JSON))
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = {}
    for uid in tags:
        res[uid] = {}
        for arm in ARMS:
            paths = [p for s in SEEDS if os.path.exists(p := img_path(uid, arm, s))]
            if paths:
                res[uid][arm] = m.score_group(uid, paths)
    with open(os.path.join(DATA, "gen10_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nuid    | A same/{N_SEEDS} mean | B same/{N_SEEDS} mean")
    rescue_thr = max(1, N_SEEDS // 2)  # >=50% same, matches T0's >=4/8 convention
    rescued = {"A": 0, "B": 0}
    for uid in tags:
        cells = []
        for arm in ARMS:
            g = res[uid].get(arm)
            if not g:
                cells.append("-"); continue
            cells.append(f"{g['same']}/{g['n']} {g['mean']:.3f}")
            if g["n"] >= N_SEEDS and g["same"] >= rescue_thr:
                rescued[arm] += 1
        print(f"u{uid:6s}| " + " | ".join(cells))
    n = len(tags)
    print(f"\nrescue (>={rescue_thr}/{N_SEEDS} SAME): A={rescued['A']}/{n}, B={rescued['B']}/{n}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tags", "gen", "score"])
    a = ap.parse_args()
    if a.stage == "tags":
        stage_tags()
    elif a.stage == "gen":
        stage_gen()
    else:
        stage_score()
