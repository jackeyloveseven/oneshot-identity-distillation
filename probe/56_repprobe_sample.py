#!/usr/bin/env python
"""
56_repprobe_sample.py — representative random-30 subject sample + prompt_B
staging for the ITS dev-probe (predicts frozen-acceptance pass rate ahead of
time). Pure sampling + tagging: no generation, no training.

Sampling paradigm is copied verbatim from probe/41_holdout_eval.py (its
sample_subjects() / is_solo() / stage_tags(), lines 44-102 as reviewed):
os.listdir(REF) for u*_ref.png -> strip EXCLUDE -> rng.shuffle -> is_solo()
(wd14 SwinV2_v3, general_threshold=0.3, solo>=0.5, no BAD_MULTI_TAGS) filter,
first N kept. prompt_B rendering also copied verbatim: importlib-load
probe/31_tagprior.py, struct = wd14 general tags (threshold tp.WD14_THR)
intersected with tp.STRUCT, phrase = tp.render_phrase(struct), prompt_B =
tp.TEMPLATE.format(ident=phrase).

Only two things differ from 41_holdout_eval.py, per this task's explicit
instructions:

1. EXCLUDE is the union of FIVE sets instead of two:
     - CAPSTONE        (41's 6 capstone uids, literal)
     - GEN10_DEVSET     (41's 11 gen10 dev-set uids, literal)
     - HOLDOUT30        every key in data/holdout30_tags.json (30)
     - TRAIN_UIDS       every "uid" field in data/train_manifest_mix3.jsonl,
                         deduped (§2.5 training-line leakage guard — the
                         single most important exclusion here: a repprobe
                         subject the model was trained on would make "does
                         this predict frozen-acceptance" meaningless, exactly
                         the failure mode probe/46_devprobe.py's docstring
                         warns about for dev-probe)
     - DEVPROBE12       every key in data/devprobe12_tags.json that does not
                         start with "_" (that file has a "_amendment"
                         metadata key alongside the 12 real uids)
   All five are read-only inputs; nothing under data/ that already exists is
   modified by this script.

2. SAMPLE_SEED = 7777 (new, disjoint from holdout30's 2024, frozen-acceptance
   §5.2's 2026, dev-probe's 2025, and gen10's 42).

REF trap (per env.sh comment + 41's own DATA INCIDENT note): env.sh's default
REF (revref_30k/img) points at a destroyed directory. Must override with
REF=/mnt/local/smoke_anime/anime_260623/img at invocation time.

Run:
  source ../env.sh
  REF=/mnt/local/smoke_anime/anime_260623/img /home/ubuntu/miniconda3/bin/python 56_repprobe_sample.py
"""
import os, json, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
print(f"[56_repprobe_sample] using REF={REF}", flush=True)

TAGS_JSON = os.path.join(DATA, "repprobe30_tags.json")
REPORT_JSON = os.path.join(DATA, "repprobe30_report.json")

HOLDOUT30_TAGS_JSON = os.path.join(DATA, "holdout30_tags.json")
TRAIN_MANIFEST = os.path.join(DATA, "train_manifest_mix3.jsonl")
DEVPROBE12_TAGS_JSON = os.path.join(DATA, "devprobe12_tags.json")

# ---- literal subsets copied verbatim from probe/41_holdout_eval.py ----
CAPSTONE = {"10000", "18522", "20582", "29998", "4238", "9172"}
GEN10_DEVSET = {"17419", "4229", "13153", "29235", "9728", "17426", "11663",
                "5910", "19098", "12523", "1509"}

N_SUBJECTS = 30
SAMPLE_SEED = 7777  # new seed: disjoint from holdout30(2024)/frozen(2026)/devprobe(2025)/gen10(42)
BAD_MULTI_TAGS = {"2boys", "2girls", "multiple_girls", "multiple_boys", "multiple_others"}


def load_holdout30():
    d = json.load(open(HOLDOUT30_TAGS_JSON))
    return {k for k in d if not k.startswith("_")}


def load_devprobe12():
    d = json.load(open(DEVPROBE12_TAGS_JSON))
    return {k for k in d if not k.startswith("_")}


def load_train_uids():
    uids = set()
    with open(TRAIN_MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uids.add(d["uid"])
    return uids


def build_exclude():
    holdout30 = load_holdout30()
    train_uids = load_train_uids()
    devprobe12 = load_devprobe12()
    subsets = {
        "capstone": CAPSTONE,
        "gen10_devset": GEN10_DEVSET,
        "holdout30": holdout30,
        "train_manifest_mix3_uids": train_uids,
        "devprobe12": devprobe12,
    }
    for name, s in subsets.items():
        print(f"[exclude] {name}: {len(s)}", flush=True)
    union = set().union(*subsets.values())
    print(f"[exclude] union: {len(union)}", flush=True)
    return subsets, union


# ---- is_solo() / sample_subjects() copied verbatim from probe/41_holdout_eval.py ----
def is_solo(uid):
    from imgutils.tagging import get_wd14_tags
    _, general, _ = get_wd14_tags(f"{REF}/u{uid}_ref.png", model_name="SwinV2_v3",
                                  general_threshold=0.3)
    return general.get("solo", 0.0) >= 0.5 and not (BAD_MULTI_TAGS & general.keys())


def sample_subjects(exclude, n=N_SUBJECTS, seed=SAMPLE_SEED):
    files = [f for f in os.listdir(REF) if f.endswith("_ref.png")]
    ids = [f[1:-8] for f in files if f[1:-8] not in exclude]
    print(f"[sample] candidate pool after EXCLUDE: {len(ids)} (of {len(files)} total refs)", flush=True)
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
    return kept, rejected


def stage_tags():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", os.path.join(HERE, "31_tagprior.py"))
    tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
    from imgutils.tagging import get_wd14_tags

    subsets, union = build_exclude()
    subs, rejected = sample_subjects(union)
    if len(subs) < N_SUBJECTS:
        print(f"[STOP] only found {len(subs)} solo subjects after EXCLUDE, need {N_SUBJECTS}. "
              f"Not writing outputs.", flush=True)
        return None

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
    print(f"wrote {TAGS_JSON} ({len(out)} subjects)", flush=True)

    # explicit zero-intersection verification against all 5 EXCLUDE subsets
    sampled = set(subs)
    intersections = {}
    all_empty = True
    for name, s in subsets.items():
        inter = sorted(sampled & s)
        intersections[name] = inter
        if inter:
            all_empty = False
        print(f"[verify] sampled ∩ {name}: {inter}", flush=True)
    print(f"[verify] all intersections empty: {all_empty}", flush=True)

    report = {
        "seed": SAMPLE_SEED,
        "n_subjects": len(subs),
        "ref_dir": REF,
        "sampled_uids": subs,
        "exclude_subset_sizes": {name: len(s) for name, s in subsets.items()},
        "exclude_union_size": len(union),
        "rejected_non_solo": rejected,
        "zero_intersection_check": intersections,
        "zero_intersection_check_all_empty": all_empty,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"wrote {REPORT_JSON}", flush=True)
    return out


if __name__ == "__main__":
    stage_tags()
