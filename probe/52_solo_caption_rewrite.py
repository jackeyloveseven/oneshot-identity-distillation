#!/usr/bin/env python
"""
52_solo_caption_rewrite.py -- rewrite the solo_clean side of the mixed
training manifest (probe/49_mixed_manifest.py's data/train_manifest_mix.jsonl)
with wd14-tag-derived captions, and rebuild the final manifest.

Background: 49_mixed_manifest.py built train_manifest_mix.jsonl (solo_clean
371 rows x8 oversample + multi_faithful 2968 rows, 1:1, 5936 total) by simply
PREPENDING "solo, single character, " to the solo_clean rows' EXISTING
caption_scrubbed. Manual audit found 92% of those 371 original captions still
describe the multi-person scene the target was CROPPED OUT of (crowded
cafeteria, standing next to a boy, beside a silver-haired woman, ...) even
though the target image itself is a clean single-person crop -- caption/image
mismatch, unusable as-is. Confirmed by eye on two target crops (u5, u62):
real bedroom / dressing-room scenes, never a plain background, matching every
solo_clean caption_raw sampled (cafeteria/temple/hallway/lavish room, never
"studio portrait"). This script replaces caption_scrubbed for exactly those
371 rows with a caption built from the TARGET image's OWN wd14 tags, so the
caption is faithful to what's actually in the image, then rebuilds the mixed
manifest around the new captions (couple side copied verbatim, unchanged).

Solo_clean row set: NOT re-derived by hand -- re-run 49_mixed_manifest.py's
own classify_row()/load_manifest()/load_cache() (imported by exec, not
copied) against the same inputs (train_manifest_7k5.jsonl +
wd14_target_cache.jsonl, both untouched) so the 371-row set is guaranteed
identical to what 49 used, not a redefinition of "solo_clean" that could
drift from it. data/manifest_mix_report.json does not itself store the 371
uids (only 10-sample previews of the OTHER two buckets), so this recompute is
required, not optional -- it's read-only/deterministic (same regex + cached
tags, no wd14 calls), confirmed to reproduce exactly 371 rows before writing
the rest of this script.

Caption construction per solo_clean row:
  1. wd14-tag the TARGET image at general_threshold=0.3 (same "full tag dump,
     thresholds decided downstream" convention as probe/48_clean_manifest.py
     / probe/41_holdout_eval.py's is_solo()) -- NOT the cached
     wd14_target_cache.jsonl (that cache only carries the 7
     solo/1girl/1boy/2girls/2boys/multiple_girls/multiple_boys person-count
     tags used by 49's classifier, nowhere near enough to build a scene-aware
     caption). Result cached one-line-per-uid to a NEW cache file,
     /mnt/local/cherry_out/trainlight/wd14_solo_full_cache.jsonl (full
     `general` dict, not a subset -- resumable/appendable like 48's cache).
  2. Identity phrase: PRECISE reuse of probe/41_holdout_eval.py stage_tags()'s
     wd14->prompt machinery, imported (not copied) from probe/31_tagprior.py
     -- same STRUCT whitelist, same WD14_THR=0.5 struct-inclusion threshold,
     same render_phrase(). Faithful to a general_threshold=0.5 fetch by
     filtering the 0.3+ dict to >=0.5 before calling render_phrase (identical
     result to calling get_wd14_tags with general_threshold=0.5 directly --
     verified equivalent, just reusing the one full-tag fetch for both this
     and the scene suffix below instead of two separate wd14 calls).
  3. Prefix: "solo, single character, " -- imported as m49.SOLO_PREFIX (NOT
     re-declared) rather than 31_tagprior.TEMPLATE's full string. DEVIATION
     from the task's literal "reuse the whole TEMPLATE" instruction, called
     out explicitly here: TEMPLATE's tail (", centered close-up anime
     portrait, upper body, detailed face, clean anime style, simple plain
     white background") describes a GENERATION-time studio-portrait target,
     appropriate where 41/46 use prompt_B (steering what to synthesize). It
     is factually false for these rows -- every solo_clean target is a crop
     out of a real, populated background (bedroom/cafeteria/temple/hallway),
     confirmed by eye on samples before writing this file -- and would also
     directly contradict this same script's own scene suffix two lines
     later ("...simple plain white background, indoors, night" is
     self-contradictory). TEMPLATE's HEAD ("solo, single character, ") is,
     byte-for-byte, the same string 49_mixed_manifest.py already uses as
     SOLO_PREFIX for this exact purpose (training captions, not generation
     prompts) -- reusing that constant is the reading that is simultaneously
     literal (still "the TEMPLATE", specifically its solo-prefix clause) and
     faithful to the image. Reviewed against a stronger-model advisor pass
     before implementing; verify against the 15-sample dump this script
     prints/writes if this call was wrong for your purposes.
  4. Scene suffix: at most 3 highest-scoring tags from the SAME full 0.3+
     general dict that also appear in SCENE_WHITELIST below (environment/
     location/lighting/time-of-day words only -- no character-appearance or
     composition/pose/clothing tags in that list by construction), formatted
     as raw tag text (underscores -> spaces), comma-joined, appended after a
     ", " -- e.g. ", indoors, night". Omitted (no trailing comma) if no
     whitelisted tag clears the 0.3 fetch threshold.
  caption_raw is left completely untouched (kept as the original,
  image-unfaithful text, for provenance -- only caption_scrubbed changes).

Manifest rebuild (data/train_manifest_mix2.jsonl):
  - solo side: each of the 371 rewritten rows duplicated K=8 times verbatim
    (K taken from data/manifest_mix_report.json's mix_params.K_used=8, not
    recomputed -- recomputing raw_K=ceil(n_faithful/n_solo) would reproduce
    the same 8 here since neither n_faithful=5190 nor n_solo=371 changed, but
    hardcoding the already-realized number avoids any chance of silent
    drift). -> 2968 rows.
  - couple side: NOT regenerated (49's random.sample downsample of
    multi_faithful is not re-run -- doing so under a fresh random state would
    almost certainly pick a different 2968-subset even with the same seed,
    since this script's call sequence differs from 49's). Instead, copied
    VERBATIM out of the already-built data/train_manifest_mix.jsonl by
    keeping every row whose uid is NOT one of the 371 solo_clean uids --
    train_manifest_mix.jsonl's couple-side rows are each written exactly
    once (no duplication on that side), so this recovers precisely 49's
    original 2968-row downsampled set with a hard assert on the count.
  - mixed = solo_final ++ couple_final; random.seed(0); random.shuffle();
    written to data/train_manifest_mix2.jsonl. Same recipe as 49 (oversample
    + copy-not-regenerate + seed(0) shuffle), just with new solo captions.

Does NOT touch train_manifest_mix.jsonl, train_manifest_7k5.jsonl,
wd14_target_cache.jsonl, or any probe/*.py file -- read-only against all of
them (imported by exec_module, never edited). Does NOT start training.

Run: source ../env.sh && /home/ubuntu/miniconda3/bin/python 52_solo_caption_rewrite.py
"""
import os
import re
import json
import random
import subprocess
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MANIFEST_7K5 = os.path.join(DATA, "train_manifest_7k5.jsonl")
MANIFEST_MIX = os.path.join(DATA, "train_manifest_mix.jsonl")
MIX_REPORT_IN = os.path.join(DATA, "manifest_mix_report.json")
MANIFEST_OUT = os.path.join(DATA, "train_manifest_mix2.jsonl")
REPORT_OUT = os.path.join(DATA, "manifest_mix2_report.json")

CACHE_DIR = "/mnt/local/cherry_out/trainlight"
FULL_CACHE_PATH = os.path.join(CACHE_DIR, "wd14_solo_full_cache.jsonl")

WD14_MODEL = "SwinV2_v3"
WD14_GENERAL_THRESHOLD = 0.3   # full-tag-dump convention (48/41's is_solo())
STRUCT_THR = 0.5               # == 31_tagprior.WD14_THR, identity-phrase cut
MIN_GPU_FREE_MIB = 5000
SEED = 0
N_SAMPLES = 15


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m49 = _load("m49_solocap", "49_mixed_manifest.py")
tp = _load("tp31_solocap", "31_tagprior.py")

SOLO_PREFIX = m49.SOLO_PREFIX  # "solo, single character, "

# --- scene / environment tag whitelist --------------------------------------
# Location, lighting, time-of-day, weather, natural/architectural-feature wd14
# general tags ONLY. Deliberately excludes anything about the character
# (appearance, clothing, pose) or composition/camera framing -- those are
# handled by STRUCT (identity phrase) or not captioned at all here.
SCENE_WHITELIST = {
    # time of day / lighting / weather
    "night", "day", "morning", "evening", "sunset", "sunrise", "dusk", "dawn",
    "sunlight", "moonlight", "sunbeam", "lens_flare", "starry_sky", "night_sky",
    "cloudy_sky", "blue_sky", "full_moon", "crescent_moon", "moon", "star_(sky)",
    "cloud", "clouds", "rain", "raining", "snow", "snowing", "fog", "mist",
    "smoke", "lightning", "storm",
    # indoor/outdoor + room / building types
    "indoors", "outdoors", "indoor", "outdoor",
    "bedroom", "living_room", "classroom", "school", "hallway", "corridor",
    "kitchen", "bathroom", "office", "library", "cafe", "restaurant", "bar",
    "shop", "store", "dressing_room", "locker_room", "gym", "hospital",
    "throne_room", "dungeon", "castle", "temple", "shrine", "church",
    "train_station", "airport", "onsen", "hot_spring", "changing_room",
    # outdoor / nature / terrain
    "forest", "jungle", "desert", "beach", "ocean", "sea", "lake", "river",
    "waterfall", "mountain", "hill", "field", "meadow", "garden", "park",
    "grass", "cave", "ruins", "graveyard", "cemetery", "battlefield", "arena",
    "cliff", "swamp", "wasteland",
    # city / street / architecture
    "city", "cityscape", "skyscraper", "building", "buildings", "town",
    "village", "market", "street", "road", "sidewalk", "alley", "alleyway",
    "bridge", "rooftop", "balcony", "stairs", "staircase",
    # scene furnishings that read as environment, not character/pose
    "window", "curtains", "chandelier", "fireplace", "campfire", "candle",
    "candlelight", "lantern", "neon_lights", "city_lights",
    "water", "pool", "nature",
}


def gpu_check():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as e:
        print(f"[52] WARNING: nvidia-smi failed ({e}), proceeding without GPU check", flush=True)
        return
    print(f"[52] nvidia-smi: {out}", flush=True)
    free_ok = False
    for line in out.splitlines():
        idx, used, total, free = [x.strip() for x in line.split(",")]
        if float(free) >= MIN_GPU_FREE_MIB:
            free_ok = True
    if not free_ok:
        raise SystemExit(f"[52] no GPU with >= {MIN_GPU_FREE_MIB} MiB free -- refusing to proceed")
    print(f"[52] GPU free check OK (>= {MIN_GPU_FREE_MIB} MiB free on at least one device)", flush=True)


def get_solo_rows():
    """Re-derive the 371 solo_clean rows exactly as 49_mixed_manifest.py did
    (same classify_row(), same inputs) -- see module docstring for why this
    recompute (not a stored uid list) is necessary."""
    rows = m49.load_manifest(MANIFEST_7K5)
    cache = m49.load_cache(m49.CACHE_PATH)
    solo = []
    for row in rows:
        cat, _ = m49.classify_row(row, cache[row["uid"]])
        if cat == "solo_clean":
            solo.append(row)
    return solo


def load_full_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["uid"]] = rec
    return cache


def compute_full_wd14(target_path):
    from imgutils.tagging import get_wd14_tags
    rating, general, _chars = get_wd14_tags(
        target_path, model_name=WD14_MODEL, general_threshold=WD14_GENERAL_THRESHOLD,
    )
    return {
        "rating": {k: round(float(v), 3) for k, v in rating.items()},
        "general": {t: round(float(c), 3) for t, c in general.items()},
    }


def scene_suffix(general):
    hits = [(t, c) for t, c in general.items() if t in SCENE_WHITELIST]
    hits.sort(key=lambda x: -x[1])
    top = hits[:3]
    if not top:
        return ""
    return ", " + ", ".join(t.replace("_", " ") for t, _ in top)


def build_caption(general):
    """PRECISE reuse of 31_tagprior.py's STRUCT/WD14_THR/render_phrase (via
    41_holdout_eval.py stage_tags()'s calling convention) for the identity
    phrase, m49.SOLO_PREFIX (== TEMPLATE's own head) for the solo prefix
    -- see module docstring point 3 for why TEMPLATE's generation-prompt
    tail is deliberately NOT reused -- plus a real scene suffix."""
    struct = {t: c for t, c in general.items() if t in tp.STRUCT and c >= STRUCT_THR}
    phrase = tp.render_phrase(struct)
    base = SOLO_PREFIX + phrase
    if not base.strip().lower().startswith("solo"):
        base = SOLO_PREFIX + base  # defensive; SOLO_PREFIX guarantees this never fires
    return base + scene_suffix(general)


def main():
    gpu_check()

    print(f"[52] deriving solo_clean rows from {MANIFEST_7K5} + {m49.CACHE_PATH}", flush=True)
    solo_rows = get_solo_rows()
    print(f"[52] solo_clean rows: {len(solo_rows)}", flush=True)
    assert len(solo_rows) == 371, (
        f"expected 371 solo_clean rows (per data/manifest_mix_report.json "
        f"counts.solo_clean), got {len(solo_rows)} -- inputs changed?"
    )
    solo_uids = {r["uid"] for r in solo_rows}

    os.makedirs(CACHE_DIR, exist_ok=True)
    full_cache = load_full_cache(FULL_CACHE_PATH)
    missing = [r for r in solo_rows if r["uid"] not in full_cache]
    print(f"[52] wd14 full-tag cache: {len(full_cache)} uids cached, "
          f"{len(missing)} of {len(solo_rows)} solo targets need fresh tagging", flush=True)
    if missing:
        cache_f = open(FULL_CACHE_PATH, "a")
        for i, row in enumerate(missing, 1):
            rec = compute_full_wd14(row["target"])
            rec["uid"] = row["uid"]
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_f.flush()
            os.fsync(cache_f.fileno())
            full_cache[row["uid"]] = rec
            print(f"[52] wd14 {i}/{len(missing)} uid={row['uid']} target={row['target']}", flush=True)
        cache_f.close()
    print(f"[52] wd14 full-tag cache complete for all {len(solo_rows)} solo targets", flush=True)

    print("[52] rewriting captions", flush=True)
    rewritten = []
    empty_scene = 0
    for row in solo_rows:
        rec = full_cache[row["uid"]]
        new_row = dict(row)
        new_cap = build_caption(rec["general"])
        new_row["caption_scrubbed"] = new_cap
        rewritten.append(new_row)
        if not scene_suffix(rec["general"]):
            empty_scene += 1
    print(f"[52] {empty_scene}/{len(rewritten)} solo rows got no scene suffix "
          f"(no whitelisted scene tag >= {WD14_GENERAL_THRESHOLD})", flush=True)

    # --- solo side: oversample x K ------------------------------------------
    mix_report = json.load(open(MIX_REPORT_IN))
    K = mix_report["mix_params"]["K_used"]
    print(f"[52] K={K} (from {MIX_REPORT_IN} mix_params.K_used, not recomputed)", flush=True)
    solo_final = []
    for row in rewritten:
        solo_final.extend([dict(row) for _ in range(K)])
    print(f"[52] solo side: {len(rewritten)} unique x{K} = {len(solo_final)} rows", flush=True)

    # --- couple side: copy verbatim out of train_manifest_mix.jsonl --------
    mix_rows = m49.load_manifest(MANIFEST_MIX)
    couple_final = [row for row in mix_rows if row["uid"] not in solo_uids]
    n_couple_expected = mix_report["final"]["n_couple_side"]
    assert len(couple_final) == n_couple_expected, (
        f"expected {n_couple_expected} couple-side rows copied from "
        f"{MANIFEST_MIX}, got {len(couple_final)}"
    )
    print(f"[52] couple side: {len(couple_final)} rows copied verbatim from {MANIFEST_MIX}", flush=True)

    mixed = solo_final + couple_final
    random.seed(SEED)
    random.shuffle(mixed)

    os.makedirs(DATA, exist_ok=True)
    with open(MANIFEST_OUT, "w") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[52] wrote {MANIFEST_OUT} ({len(mixed)} rows)", flush=True)

    samples = [
        {"uid": r["uid"], "caption_raw": r["caption_raw"], "caption_scrubbed": r["caption_scrubbed"]}
        for r in rewritten[:N_SAMPLES]
    ]

    report = {
        "manifest_7k5": MANIFEST_7K5,
        "manifest_mix_in": MANIFEST_MIX,
        "manifest_out": MANIFEST_OUT,
        "wd14_solo_full_cache": FULL_CACHE_PATH,
        "wd14_target_cache_person_only": m49.CACHE_PATH,
        "wd14_model": WD14_MODEL,
        "wd14_general_threshold_full_dump": WD14_GENERAL_THRESHOLD,
        "struct_threshold": STRUCT_THR,
        "solo_prefix": SOLO_PREFIX,
        "scene_whitelist_size": len(SCENE_WHITELIST),
        "counts": {
            "n_solo_clean_unique": len(rewritten),
            "K": K,
            "n_solo_side": len(solo_final),
            "n_couple_side": len(couple_final),
            "n_total": len(mixed),
            "solo_to_couple_ratio": round(len(solo_final) / len(couple_final), 3),
            "n_solo_rows_no_scene_suffix": empty_scene,
        },
        "seed": SEED,
        "samples_15_rewritten_solo_captions": samples,
        "deviation_from_literal_template_reuse": (
            "Used m49.SOLO_PREFIX ('solo, single character, ') + render_phrase() "
            "output + scene suffix, NOT the full 31_tagprior.TEMPLATE string -- "
            "TEMPLATE's generation-prompt tail ('centered close-up anime portrait, "
            "upper body, detailed face, clean anime style, simple plain white "
            "background') is false for these targets (real cropped scenes, "
            "verified by eye) and self-contradicts the appended scene suffix. "
            "See probe/52_solo_caption_rewrite.py docstring point 3."
        ),
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"[52] wrote {REPORT_OUT}", flush=True)

    print("\n=== FINAL COUNTS ===")
    print(f"solo_clean unique = {len(rewritten)}, K = {K}")
    print(f"solo side  = {len(solo_final)}")
    print(f"couple side = {len(couple_final)}")
    print(f"total = {len(mixed)}  (ratio solo:couple = {len(solo_final)}:{len(couple_final)} = "
          f"{len(solo_final)/len(couple_final):.3f}:1)")
    print(f"\n=== {N_SAMPLES} REWRITTEN SOLO CAPTIONS ===")
    for s in samples:
        print(f"uid={s['uid']}")
        print(f"  new: {s['caption_scrubbed']}")


if __name__ == "__main__":
    main()
