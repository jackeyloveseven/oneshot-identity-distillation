#!/usr/bin/env python
"""
48_clean_manifest.py -- data cleaning pass over train_manifest_7k5.jsonl, aimed
at the "卧室双人/亲密场景" content drift observed in the LoRA trained on it
(bedroom / two-person romantic content leaking into a nominally single-subject
identity LoRA -- captions like "intimately holding a woman with brown hair" or
"while a woman ..." and target images that visibly contain a second person).

Root cause: train_manifest_7k5.jsonl (5945 rows, fields uid/ref/target/
caption_raw/caption_scrubbed) was built from a raw caption+image pool that was
never filtered for "is this actually a solo-subject training pair" -- a
nontrivial fraction of rows show two people embracing/kissing/cuddling, often
in a bedroom, and the caption spells it out just as explicitly ("passionately
embracing...", "holding a woman with brown hair", "surrounded by several other
girls...", ...). Quick grep-level confirmation before writing this script:
"intimate" 455x, "kiss" 309x, "embrac" 421x, "another (woman|man|girl|boy)"
289x, "surrounded by" 236x across the 5945 rows. That's poison for a
single-identity ref-conditioned LoRA: it teaches the model to paint a second
character into scenes and burns caption/gradient budget on someone who is not
the training subject.

This script filters on two INDEPENDENT signals, applied per manifest row:

  a) drop_not_solo -- WD14-tag the TARGET image (imgutils, SwinV2_v3, same
     model/threshold convention as probe/41_holdout_eval.py's is_solo() gate:
     general_threshold=0.3, so tags in the 0.3-0.5 range are still visible in
     the returned dict rather than silently clipped to 0). Drop if:
       general['solo'] < 0.5
       OR max(2girls, 2boys, multiple_girls, multiple_boys) >= 0.35
       OR min(1girl, 1boy) >= 0.35   (an explicit F+M pair -- WD14's "1girl"/
                                       "1boy" count tags are only both present
                                       together when there is exactly one of
                                       each, i.e. a two-person mixed scene)

  b) drop_caption_multi -- caption_scrubbed matches a regex for two-person /
     intimate content. The BASE set is the task-specified minimum:
       intimate|kiss|embrac|cuddl|in his arms|in her arms|
       holding a (woman|man|girl|boy)|while a (woman|man|girl|boy)|
       another (woman|man|girl|boy)|surrounded by
     A small EXTRA set was added after manually inspecting each candidate's
     hits on this corpus (see the "extra pattern audit" note below) --
     nuzzl/caress/snuggl/spooning/nestled against/romantic/"wrapping
     his/her/their arms around"/"two girls|boys|women|men"/making out/
     entwined. Deliberately EXCLUDED after manual review because they had a
     real false-positive rate in THIS corpus: bare "hug" (mostly "hugging her
     knees" / "hugging a plushie" -- self- or object-directed, not another
     person), bare "wrapped" (mostly "wrapped in a blanket"/"towel wrapped
     around his waist" -- not a person), "leaning against" / "pressed
     against" (mostly a wall/doorway/lamppost/glass door, not a person). Same
     pitfall probe/44_caption_scrub.py hit with bare "bow" (school-uniform
     bow tie / weapon bow / verb "to bow") -- a pattern is only safe here if
     its object is unambiguously a person.

  c) drop_nsfw -- WD14's `rating` head (imgutils' get_wd14_tags returns
     ('rating','general','character') by default -- rating IS available on
     this box, confirmed by hand before writing this script, so this
     condition is active, not skipped). Drop if rating['explicit'] >= 0.5 or
     rating['questionable'] >= 0.5.

Runtime note: this box's onnxruntime install only exposes CPUExecutionProvider
(onnxruntime-gpu is pip-installed alongside plain onnxruntime, but the two
packages share the `onnxruntime` import name and the CPU one wins -- not this
script's problem to fix, see task instructions re: dependency issues).
Measured ~0.45s/image on CPU -> ~45 min for 5945 rows, within the run budget.
No GPU workaround attempted.

Output:
  data/train_manifest_7k5_clean.jsonl -- rows with ZERO drop reasons, same 5
                                          fields/order as the input, unchanged
                                          format.
  data/manifest_clean_report.json     -- total/kept/dropped, per-reason counts
                                          (overlapping -- a row can hit more
                                          than one reason so reasons don't sum
                                          to dropped count), per-uid drop
                                          detail, and caption style-word
                                          distribution (regex
                                          `[Ii]n (an? )?([a-z_0-9]+) style`)
                                          for the full set and the kept subset.

WD14 results are cached one JSON line per uid to
/mnt/local/cherry_out/trainlight/wd14_target_cache.jsonl -- a rerun loads this
cache first and only calls the tagger for uids missing from it, so the script
is resumable after an interruption (Ctrl-C, OOM, box reboot, ...). The cache
file is opened in append mode and flushed+fsynced after every new line so a
mid-run kill doesn't lose already-computed rows.

Run (needs $PY / env vars from env.sh; GPU not required, see runtime note):
  source ../env.sh && $PY 48_clean_manifest.py
  $PY 48_clean_manifest.py --limit 50        # smoke test on first 50 rows
                                              # (still writes to the real
                                              # cache; harmless / resumable)
"""
import os
import re
import sys
import json
import time
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MANIFEST_IN = os.path.join(DATA, "train_manifest_7k5.jsonl")
MANIFEST_OUT = os.path.join(DATA, "train_manifest_7k5_clean.jsonl")
REPORT_OUT = os.path.join(DATA, "manifest_clean_report.json")

CACHE_DIR = "/mnt/local/cherry_out/trainlight"
CACHE_PATH = os.path.join(CACHE_DIR, "wd14_target_cache.jsonl")

WD14_MODEL = "SwinV2_v3"
WD14_GENERAL_THRESHOLD = 0.3  # < the 0.35/0.5 decision thresholds below, so
                               # tags in that range are never silently dropped
                               # from the returned dict before we can check them
TAGS_OF_INTEREST = ["solo", "1girl", "1boy", "2girls", "2boys",
                     "multiple_girls", "multiple_boys"]

SOLO_THR = 0.5
MULTI_THR = 0.35
NSFW_THR = 0.5

# --- caption_scrubbed multi-person / intimate content regex --------------
# BASE = task-specified minimum. EXTRA = manually-audited additions (see
# docstring "extra pattern audit" note). Combined into one compiled regex;
# kept as separate strings here purely so the provenance of each clause is
# legible in a diff/review, not because they're applied differently.
BASE_MULTI_PATTERNS = (
    r"intimate", r"kiss", r"embrac", r"cuddl",
    r"in his arms", r"in her arms",
    r"holding a (?:woman|man|girl|boy)",
    r"while a (?:woman|man|girl|boy)",
    r"another (?:woman|man|girl|boy)",
    r"surrounded by",
)
EXTRA_MULTI_PATTERNS = (
    r"nuzzl", r"caress", r"snuggl", r"spooning", r"nestled against",
    r"romantic", r"wrapp\w+ (?:his|her|their) arms around",
    r"\btwo (?:girls|boys|women|men)\b",
    r"making out", r"entwined",
)
CAPTION_MULTI_RE = re.compile(
    "|".join(BASE_MULTI_PATTERNS + EXTRA_MULTI_PATTERNS), re.IGNORECASE
)

STYLE_RE = re.compile(r"[Ii]n (?:an? )?([a-z_0-9]+) style")


def load_manifest(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line from a killed run
                cache[rec["uid"]] = rec
    return cache


def compute_wd14(target_path):
    from imgutils.tagging import get_wd14_tags
    rating, general, _chars = get_wd14_tags(
        target_path, model_name=WD14_MODEL,
        general_threshold=WD14_GENERAL_THRESHOLD,
    )
    tags = {t: round(float(general[t]), 3) for t in TAGS_OF_INTEREST if t in general}
    rating_out = {k: round(float(v), 3) for k, v in rating.items()}
    return {"rating": rating_out, "tags": tags}


def decide(row, wd14_rec):
    reasons = []

    tags = wd14_rec.get("tags", {})
    solo = tags.get("solo", 0.0)
    multi_max = max(
        tags.get("2girls", 0.0), tags.get("2boys", 0.0),
        tags.get("multiple_girls", 0.0), tags.get("multiple_boys", 0.0),
        min(tags.get("1girl", 0.0), tags.get("1boy", 0.0)),
    )
    if solo < SOLO_THR or multi_max >= MULTI_THR:
        reasons.append("drop_not_solo")

    if CAPTION_MULTI_RE.search(row.get("caption_scrubbed", "")):
        reasons.append("drop_caption_multi")

    rating = wd14_rec.get("rating", {})
    if rating.get("explicit", 0.0) >= NSFW_THR or rating.get("questionable", 0.0) >= NSFW_THR:
        reasons.append("drop_nsfw")

    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N manifest rows (smoke test)")
    args = ap.parse_args()

    print(f"[48_clean_manifest] reading {MANIFEST_IN}", flush=True)
    rows = load_manifest(MANIFEST_IN, limit=args.limit)
    n_total = len(rows)
    print(f"[48_clean_manifest] {n_total} rows", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_cache(CACHE_PATH)
    print(f"[48_clean_manifest] wd14 cache: {len(cache)} uids already computed", flush=True)

    cache_f = open(CACHE_PATH, "a")

    reason_counts = Counter()
    dropped_detail = []
    kept_rows = []
    style_full = Counter()
    style_kept = Counter()
    wd14_errors = []

    n_computed = 0
    t_start = time.time()

    for i, row in enumerate(rows):
        uid = row["uid"]
        caption = row.get("caption_scrubbed", "")
        for m in STYLE_RE.finditer(caption):
            style_full[m.group(1).lower()] += 1

        if uid in cache:
            wd14_rec = cache[uid]
        else:
            try:
                wd14_rec = compute_wd14(row["target"])
            except Exception as e:  # noqa: BLE001 - defensive, image/tagger errors
                print(f"[wd14 ERROR] uid={uid} target={row['target']}: {e}", flush=True)
                wd14_errors.append({"uid": uid, "target": row["target"], "error": str(e)})
                # conservative: can't verify this row's content -> exclude it,
                # but do NOT cache the failure so a rerun retries it
                dropped_detail.append({"uid": uid, "reasons": ["drop_wd14_error"]})
                reason_counts["drop_wd14_error"] += 1
                continue
            wd14_rec["uid"] = uid
            cache[uid] = wd14_rec
            cache_f.write(json.dumps(wd14_rec, ensure_ascii=False) + "\n")
            cache_f.flush()
            os.fsync(cache_f.fileno())
            n_computed += 1

        reasons = decide(row, wd14_rec)
        if reasons:
            for r in reasons:
                reason_counts[r] += 1
            dropped_detail.append({"uid": uid, "reasons": reasons})
        else:
            kept_rows.append(row)
            for m in STYLE_RE.finditer(caption):
                style_kept[m.group(1).lower()] += 1

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            rate = n_computed / elapsed if elapsed > 0 and n_computed else 0.0
            remaining = n_total - (i + 1)
            eta_computed = (elapsed / n_computed * remaining) if n_computed else 0.0
            print(f"[progress] {i+1}/{n_total} rows, {n_computed} wd14-computed "
                  f"this run ({rate:.2f}/s), kept={len(kept_rows)}, "
                  f"eta~{eta_computed/60:.1f}min", flush=True)

    cache_f.close()

    os.makedirs(DATA, exist_ok=True)
    with open(MANIFEST_OUT, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[48_clean_manifest] wrote {MANIFEST_OUT} ({len(kept_rows)} rows)", flush=True)

    report = {
        "manifest_in": MANIFEST_IN,
        "manifest_out": MANIFEST_OUT,
        "total": n_total,
        "kept": len(kept_rows),
        "dropped": n_total - len(kept_rows),
        "reason_counts_overlapping": dict(reason_counts),
        "notes": [
            "reason_counts_overlapping counts each condition independently; "
            "a dropped row can hit more than one reason, so these do not sum "
            "to `dropped`.",
            "drop_nsfw was ACTIVE (wd14 rating head was available on this "
            "box) -- not skipped.",
            f"wd14: model={WD14_MODEL}, general_threshold={WD14_GENERAL_THRESHOLD}, "
            f"solo_thr={SOLO_THR}, multi_thr={MULTI_THR}, nsfw_thr={NSFW_THR}.",
        ],
        "wd14_errors": wd14_errors,
        "dropped_detail": dropped_detail,
        "style_distribution": {
            "full_set": dict(style_full.most_common()),
            "kept_set": dict(style_kept.most_common()),
        },
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"[48_clean_manifest] wrote {REPORT_OUT}", flush=True)

    # --- terminal summary ---
    print("\n=== SUMMARY ===")
    print(f"total={n_total}  kept={len(kept_rows)}  dropped={n_total - len(kept_rows)}")
    print("reason counts (overlapping):")
    for r, c in reason_counts.most_common():
        print(f"  {r}: {c}")
    print("\nstyle distribution (full set):")
    for style, c in style_full.most_common():
        print(f"  {style}: {c}")
    print("\nstyle distribution (kept set):")
    for style, c in style_kept.most_common():
        print(f"  {style}: {c}")
    if wd14_errors:
        print(f"\n{len(wd14_errors)} wd14 tagging errors -- see report.wd14_errors "
              f"(these rows were dropped as drop_wd14_error and NOT cached, "
              f"so a rerun will retry them)")


if __name__ == "__main__":
    main()
