#!/usr/bin/env python
"""
41_holdout_eval.py — FROZEN-recipe held-out validation. Advisor flagged that
every result on gen10 (strip/center/lowfreq x band 21-25/10-25, all winning
numbers so far: 55-60% yes / 0% no) was measured on the SAME 10 subjects that
band+lowfreq_k were tuned against — that's a dev-set number, not a validation
number, and not defensible in a paper.

This script freezes the recipe NOW (band=10-25, refpos=lowfreq, lowfreq_k=8,
sched=taper, w=4.0 — no more tuning after seeing this set's results) and
evaluates it, paired against the OLD default (band=21-25, refpos=strip, same
w/sched/caption), on a FRESH random sample disjoint from gen10 and capstone.
Judged ONCE, blind, via Gemini 3.1 Pro (38_gemini_judge.py --set holdout).
If this number looks bad, do NOT re-tune band/lowfreq_k against it — draw a
second fresh held-out set instead (re-tuning against this one recontaminates
it exactly the way gen10 got contaminated).

Sample: 30 subjects, random.Random(2024) (new seed, disjoint from gen10's 42),
excluding CAPSTONE + all gen10 dev-set uids, is_solo()-filtered (minefield #11).
2 seeds x 2 arms x 30 subjects = 120 images.

DATA INCIDENT (2026-07-06): $REF (revref_30k/img, the path env.sh normally
exports) was emptied by an external, non-project cause sometime before this
run (ruled out this project's own commands, diskmon.sh [read-only], and
r2-patrol [only touches felix's musubi-tuner dirs] — see experiment-log
2026-07-06 pitfall entry). This script's own EXCLUDE set already guarantees
disjointness from capstone/gen10 regardless of which physical directory the
images come from, so — unlike gen10/capstone reproductions, which DO need the
original revref_30k images and must wait for that to be restored — sourcing
this held-out sample from a surviving alternate snapshot is safe: run with
REF=/mnt/local/smoke_anime/anime_260623/img (9770 candidate _ref.png ids,
confirmed >>30 needed after EXCLUDE) overriding env.sh's default. Logged
explicitly here and printed at runtime so this substitution is traceable in
any paper/report that cites this experiment's numbers.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img $PY 41_holdout_eval.py --stage tags
     REF=/mnt/local/smoke_anime/anime_260623/img $PY 41_holdout_eval.py --stage gen
     REF=/mnt/local/smoke_anime/anime_260623/img $PY 41_holdout_eval.py --stage score
     $PY 38_gemini_judge.py --set holdout   (separate call, also needs REF override — see its own REF read)
"""
import os, json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
print(f"[41_holdout_eval] using REF={REF}", flush=True)
OUT = os.environ.get("HOLDOUT_OUT", "/mnt/local/cherry_out/holdout30")
TAGS_JSON = os.path.join(DATA, "holdout30_tags.json")

CAPSTONE = {"10000", "18522", "20582", "29998", "4238", "9172"}
GEN10_DEVSET = {"17419", "4229", "13153", "29235", "9728", "17426", "11663",
                "5910", "19098", "12523", "1509"}
EXCLUDE = CAPSTONE | GEN10_DEVSET
N_SUBJECTS = 30
SAMPLE_SEED = 2024
SEEDS = [0, 1]
ARMS = ["old", "new"]  # old = strip/band21-25 (prior default), new = lowfreq8/band10-25 (frozen candidate)
BAD_MULTI_TAGS = {"2boys", "2girls", "multiple_girls", "multiple_boys", "multiple_others"}

def is_solo(uid):
    from imgutils.tagging import get_wd14_tags
    _, general, _ = get_wd14_tags(f"{REF}/u{uid}_ref.png", model_name="SwinV2_v3",
                                  general_threshold=0.3)
    return general.get("solo", 0.0) >= 0.5 and not (BAD_MULTI_TAGS & general.keys())

def sample_subjects(n=N_SUBJECTS, seed=SAMPLE_SEED):
    files = [f for f in os.listdir(REF) if f.endswith("_ref.png")]
    ids = [f[1:-8] for f in files if f[1:-8] not in EXCLUDE]
    rng = random.Random(seed)
    rng.shuffle(ids)
    kept, rejected = [], 0
    for uid in ids:
        if len(kept) >= n:
            break
        if is_solo(uid):
            kept.append(uid)
        else:
            rejected += 1
    print(f"[sample] kept {len(kept)}, rejected {rejected} (multi-char/no-solo)", flush=True)
    return kept

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
        print(f"u{uid}: {phrase!r}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    with open(TAGS_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {TAGS_JSON} ({len(out)} subjects)")

def img_path(uid, arm, seed):
    return f"{OUT}/u{uid}_holdout_{arm}_s{seed}.png"

def stage_gen():
    tags = json.load(open(TAGS_JSON))
    todo = [(u, a, s) for u in tags for a in ARMS for s in SEEDS
            if not os.path.exists(img_path(u, a, s))]
    print(f"[gen] {len(todo)} images remaining of {len(tags)*len(ARMS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    RECIPE = {
        "old": dict(band="21-25", refpos="strip", lowfreq_k=4),
        "new": dict(band="10-25", refpos="lowfreq", lowfreq_k=8),
    }
    for uid, arm, seed in todo:
        r = RECIPE[arm]
        kv.INJ["band"] = kv.parse_band(r["band"])
        prompt = tags[uid]["prompt_B"]
        print(f"[gen] u{uid} arm={arm} seed={seed}", flush=True)
        im = kv.generate(uid, 4.0, refpos=r["refpos"], lowfreq_k=r["lowfreq_k"],
                         sched_mode="taper", seed=seed, prompt=prompt)
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
    with open(os.path.join(DATA, "holdout30_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    for arm in ARMS:
        same_n = sum(res[u][arm]["same"] for u in res if arm in res[u])
        n = sum(res[u][arm]["n"] for u in res if arm in res[u])
        print(f"CCIP arm={arm}: same={same_n}/{n} ({100*same_n/max(1,n):.0f}%)")
    print("(CCIP is a sanity check only, known to overestimate — "
          "final judgment: 38_gemini_judge.py --set holdout, single blind pass)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tags", "gen", "score"])
    a = ap.parse_args()
    {"tags": stage_tags, "gen": stage_gen, "score": stage_score}[a.stage]()
