#!/usr/bin/env python
"""
53_ref_caption_wd14.py -- M1-contingency playbook (2) (docs/design/
2026-07-11_traininglight_design.md S8 (2) "ref caption 换描述性标签"): wd14-tag
every ref image the training/dev-probe pipeline touches, and build the
cap_ref text that will replace probe/45_train_lora.py's hardcoded
CAP_REF_TXT = "anime illustration of a character" constant.

Why: the [ref_lat, x_t] training sequence's text channel currently gives the
ref slot a caption that is a fixed literal string, identical for every uid --
zero binding between the text and which image is actually sitting in the ref
slot. dev-probe failure notes cluster on signature accessories/marks
(sunglasses, tattoos, lip piercings, scars) getting dropped -- exactly the
kind of attribute wd14 tags cleanly. This script gives every ref image its
own wd14-tag-derived phrase so cap_ref becomes a real "what to read out of
ref" index instead of a no-op placeholder.

Inputs (read-only):
  - data/train_manifest_mix2.jsonl   -- 5936 rows / 3339 unique uids, each
    row's "ref" field already points at REF_IMG_DIR/u{uid}_ref.png (verified
    byte-identical against the template before writing this script).
  - data/devprobe12_tags.json        -- 12 dev-probe uids (top-level keys not
    starting with "_"); ref path is the same REF_IMG_DIR template.

Outputs:
  - /mnt/local/cherry_out/trainlight/wd14_ref_cache.jsonl  -- one line per
    uid, {"uid","rating","general"} at general_threshold=0.3 (the "fetch
    wide, filter downstream" convention probe/48_clean_manifest.py /
    probe/52_solo_caption_rewrite.py already use), fsync'd per line so a
    killed run resumes without re-tagging.
  - data/train_manifest_mix3.jsonl   -- mix2, verbatim, plus one new
    "ref_caption" field per row. Built by STRING SURGERY on each mix2 line
    (append `, "ref_caption": "..."` before the trailing "}"), not by
    json.loads->json.dumps round-tripping the whole row, so every other
    byte of every row is provably untouched (float formatting / key order /
    unicode escaping of caption_raw/caption_scrubbed can silently drift
    across a dumps() round-trip otherwise). Verified after writing (see
    verify_byte_exact() below): row count, row order, and the pre-existing
    fields of every single row are checked to be identical to mix2, and the
    new bytes are checked to be a strict suffix-insertion (mix3 line startswith
    mix2 line's content minus its closing brace).
  - data/devprobe12_refcaps.json     -- {uid: ref_caption} for the 12
    dev-probe uids, built from the SAME build_ref_caption() call as mix3 (one
    function, two call sites) so a uid's caption cannot silently diverge
    between the training manifest and the dev-probe sampler that will
    eventually consume it via --cap_ref.
  - data/manifest_mix3_report.json   -- stats, the 4 user-named signature-item
    diagnostics (u7495/u1138/u2842/u3737), and 15 sample captions (accessory/
    scar/piercing-tagged rows prioritized).

Caption format (per task spec): "an anime illustration of a character:
{phrase}" where phrase = probe/31_tagprior.py's render_phrase() over the
STRUCT whitelist at WD14_THR=0.5 -- imported, not redefined, so any future
change to the whitelist/threshold there is picked up here automatically.
Deliberately NO scene suffix (unlike 52's solo-target captions): ref images
are portrait/bust crops, scene words there are noise, and cap_ref's job is
identity indexing, not scene description.

Does NOT touch train_manifest_mix2.jsonl, devprobe12_tags.json, or any
probe/*.py file (all read-only / imported by exec, never edited). Does NOT
start training.

Run: source ../env.sh && /home/ubuntu/miniconda3/bin/python 53_ref_caption_wd14.py
"""
import os
import sys
import json
import time
import random
import subprocess
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MANIFEST_IN = os.path.join(DATA, "train_manifest_mix2.jsonl")
MANIFEST_OUT = os.path.join(DATA, "train_manifest_mix3.jsonl")
DEVPROBE_TAGS_IN = os.path.join(DATA, "devprobe12_tags.json")
DEVPROBE_REFCAPS_OUT = os.path.join(DATA, "devprobe12_refcaps.json")
REPORT_OUT = os.path.join(DATA, "manifest_mix3_report.json")

CACHE_DIR = "/mnt/local/cherry_out/trainlight"
CACHE_PATH = os.path.join(CACHE_DIR, "wd14_ref_cache.jsonl")
REF_IMG_DIR = "/mnt/local/smoke_anime/anime_260623/img"  # == 45_train_lora.REF_IMG_DIR
                                                          # == 46_devprobe.REF_IMG_DIR default

WD14_MODEL = "SwinV2_v3"
WD14_GENERAL_THRESHOLD = 0.3   # full-tag-dump convention (48/52); STRUCT filter applied downstream
MIN_GPU_FREE_MIB = 5000
SEED = 0
N_SAMPLES = 15
TIME_PROBE_N = 20              # first N fresh-tagged images -> report rate/ETA before continuing
PROGRESS_EVERY = 200

HIGHLIGHT_UIDS = {
    "7495": "sunglasses (用户点名)",
    "1138": "耳饰/earrings (用户点名)",
    "2842": "疤痕/scar (用户点名)",
    "3737": "唇钉/lip piercing (用户点名)",
}


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tp = _load("tp31_refcap", "31_tagprior.py")
STRUCT_THR = tp.WD14_THR  # 0.5, imported not redefined


def gpu_check():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as e:
        print(f"[53] WARNING: nvidia-smi failed ({e}), proceeding without GPU check", flush=True)
        return
    print(f"[53] nvidia-smi: {out}", flush=True)
    free_ok = False
    for line in out.splitlines():
        idx, used, total, free = [x.strip() for x in line.split(",")]
        if float(free) >= MIN_GPU_FREE_MIB:
            free_ok = True
    if not free_ok:
        raise SystemExit(f"[53] no GPU with >= {MIN_GPU_FREE_MIB} MiB free -- refusing to proceed")
    print(f"[53] GPU free check OK (>= {MIN_GPU_FREE_MIB} MiB free on at least one device)", flush=True)


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


def compute_full_wd14(ref_path):
    from imgutils.tagging import get_wd14_tags
    rating, general, _chars = get_wd14_tags(
        ref_path, model_name=WD14_MODEL, general_threshold=WD14_GENERAL_THRESHOLD,
    )
    return {
        "rating": {k: round(float(v), 3) for k, v in rating.items()},
        "general": {t: round(float(c), 3) for t, c in general.items()},
    }


def collect_uid_refpaths():
    """uid -> ref image path, plus the manifest-uid and devprobe-uid sets
    (disjoint, verified below). Manifest ref paths are taken directly from
    the manifest rows (not re-templated) so a template/actual mismatch would
    surface as a missing file, not a silent wrong-image tag."""
    manifest_uid2path = {}
    with open(MANIFEST_IN) as f:
        for line in f:
            d = json.loads(line)
            u = d["uid"]
            if u in manifest_uid2path:
                assert manifest_uid2path[u] == d["ref"], f"uid {u} has inconsistent ref path across rows"
            else:
                manifest_uid2path[u] = d["ref"]

    dp_tags = json.load(open(DEVPROBE_TAGS_IN))
    devprobe_uids = [k for k in dp_tags.keys() if not k.startswith("_")]
    devprobe_uid2path = {u: f"{REF_IMG_DIR}/u{u}_ref.png" for u in devprobe_uids}

    overlap = set(manifest_uid2path) & set(devprobe_uid2path)
    uid2path = dict(manifest_uid2path)
    uid2path.update(devprobe_uid2path)
    return uid2path, set(manifest_uid2path), set(devprobe_uid2path), overlap


def build_ref_caption(general):
    """SINGLE function used for both mix3's ref_caption field and
    devprobe12_refcaps.json -- guarantees a uid's caption is identical
    wherever it is consumed. `general` = the cached 0.3+ wd14 dict."""
    struct = {t: c for t, c in general.items() if t in tp.STRUCT and c >= STRUCT_THR}
    phrase = tp.render_phrase(struct)
    return f"an anime illustration of a character: {phrase}", struct


def verify_byte_exact(path_in, path_out):
    with open(path_in) as f:
        lines_in = f.readlines()
    with open(path_out) as f:
        lines_out = f.readlines()
    assert len(lines_in) == len(lines_out), (
        f"row count drift: mix2={len(lines_in)} mix3={len(lines_out)}")
    for i, (li, lo) in enumerate(zip(lines_in, lines_out)):
        di = json.loads(li)
        do = json.loads(lo)
        cap = do.pop("ref_caption")
        assert do == di, f"row {i} (uid={di.get('uid')}): fields changed beyond ref_caption"
        stripped_in = li.rstrip("\n")
        assert stripped_in.endswith("}"), f"row {i}: unexpected mix2 line ending"
        prefix = stripped_in[:-1]
        assert lo.rstrip("\n").startswith(prefix), (
            f"row {i} (uid={di.get('uid')}): mix3 line is not a strict suffix-insertion over mix2 line")
        assert isinstance(cap, str) and cap, f"row {i}: empty/non-string ref_caption"
    print(f"[53] verify_byte_exact OK: {len(lines_in)} rows, all pre-existing fields + byte "
          f"prefix identical mix2->mix3, only a trailing ref_caption field appended", flush=True)


def main():
    gpu_check()

    uid2path, manifest_uids, devprobe_uids, overlap = collect_uid_refpaths()
    print(f"[53] manifest uids={len(manifest_uids)} devprobe uids={len(devprobe_uids)} "
          f"overlap={len(overlap)} union={len(uid2path)}", flush=True)

    missing_files = [u for u, p in uid2path.items() if not os.path.exists(p)]
    if missing_files:
        raise SystemExit(f"[53] {len(missing_files)} ref images missing on disk, e.g. "
                          f"{missing_files[:5]} -- aborting")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_full_cache(CACHE_PATH)
    todo = [u for u in uid2path if u not in cache]
    print(f"[53] wd14 cache {CACHE_PATH}: {len(cache)} uids already cached "
          f"({len(cache) - len(set(cache) - set(uid2path))} relevant), {len(todo)} of "
          f"{len(uid2path)} need fresh tagging", flush=True)

    if todo:
        cache_f = open(CACHE_PATH, "a")
        t_run0 = time.time()
        probe_n = min(TIME_PROBE_N, len(todo))
        for i, u in enumerate(todo, 1):
            rec = compute_full_wd14(uid2path[u])
            rec["uid"] = u
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_f.flush()
            os.fsync(cache_f.fileno())
            cache[u] = rec
            if i == probe_n:
                dt = time.time() - t_run0
                rate = dt / probe_n
                remaining = len(todo) - probe_n
                eta_min = remaining * rate / 60.0
                print(f"[53] timing probe: {probe_n} images in {dt:.1f}s ({rate:.3f}s/img) "
                      f"-> ETA for remaining {remaining}: {eta_min:.1f} min", flush=True)
            if i % PROGRESS_EVERY == 0 or i == len(todo):
                dt = time.time() - t_run0
                print(f"[53] wd14 progress {i}/{len(todo)} ({i/dt:.2f} img/s, {dt/60:.1f} min elapsed)",
                      flush=True)
        cache_f.close()
        dt_total = time.time() - t_run0
        print(f"[53] wd14 tagging done: {len(todo)} fresh tags in {dt_total/60:.1f} min "
              f"({len(todo)/max(dt_total,1e-9):.2f} img/s)", flush=True)
    else:
        print("[53] wd14 cache already complete for every uid in scope, skipping tagging", flush=True)

    # --- build ref_caption for every uid, one function, two call sites downstream ---
    ref_captions = {}
    struct_hits = {}
    for u in uid2path:
        cap, struct = build_ref_caption(cache[u]["general"])
        ref_captions[u] = cap
        struct_hits[u] = struct

    # --- mix3: byte-exact string surgery over mix2 ---
    n_written = 0
    with open(MANIFEST_IN) as fin, open(MANIFEST_OUT, "w") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            d = json.loads(raw)  # parsed only to read uid; NOT re-dumped
            uid = d["uid"]
            cap = ref_captions[uid]
            assert raw.endswith("}"), f"unexpected mix2 line ending for uid={uid}"
            fout.write(raw[:-1] + ', "ref_caption": ' + json.dumps(cap, ensure_ascii=False) + "}\n")
            n_written += 1
    print(f"[53] wrote {MANIFEST_OUT} ({n_written} rows)", flush=True)
    verify_byte_exact(MANIFEST_IN, MANIFEST_OUT)

    # --- devprobe12_refcaps.json ---
    devcaps = {u: ref_captions[u] for u in devprobe_uids}
    with open(DEVPROBE_REFCAPS_OUT, "w") as f:
        json.dump(devcaps, f, indent=1, ensure_ascii=False)
    print(f"[53] wrote {DEVPROBE_REFCAPS_OUT} ({len(devcaps)} uids)", flush=True)
    print("[53] devprobe12_refcaps.json contents:", flush=True)
    for u in sorted(devcaps, key=int):
        print(f"  u{u}: {devcaps[u]}", flush=True)

    # --- sanity cross-check vs devprobe12_tags.json's own phrase (same wd14
    #     model/threshold/whitelist, same image path, computed independently
    #     by 46_devprobe.py -- should reproduce identically) ---
    dp_tags = json.load(open(DEVPROBE_TAGS_IN))
    mismatches = []
    for u in devprobe_uids:
        old_phrase = dp_tags[u]["phrase"]
        our_phrase = ref_captions[u][len("an anime illustration of a character: "):]
        if our_phrase != old_phrase:
            mismatches.append({"uid": u, "devprobe12_tags_phrase": old_phrase, "our_phrase": our_phrase})
    if mismatches:
        print(f"[53] WARNING: {len(mismatches)}/{len(devprobe_uids)} devprobe uids' phrase differs "
              f"from devprobe12_tags.json's independently-computed phrase (expected to match -- "
              f"same wd14 model/threshold/whitelist/image path):", flush=True)
        for m in mismatches:
            print(f"    u{m['uid']}: old={m['devprobe12_tags_phrase']!r} new={m['our_phrase']!r}", flush=True)
    else:
        print(f"[53] cross-check OK: all {len(devprobe_uids)} devprobe phrases match "
              f"devprobe12_tags.json's independently-computed phrase", flush=True)

    # --- signature-item diagnostics: the 4 user-named uids ---
    keywords = ["glass", "sunglass", "ear", "scar", "lip", "pierc", "tattoo", "mark"]
    diagnostics = {}
    for u, label in HIGHLIGHT_UIDS.items():
        gen = cache[u]["general"]
        hits = {t: c for t, c in gen.items() if any(k in t for k in keywords)}
        rows = []
        for t, c in sorted(hits.items(), key=lambda x: -x[1]):
            in_struct = t in tp.STRUCT
            if in_struct and c >= STRUCT_THR:
                status = "IN_CAPTION (score>=0.5, tag in STRUCT whitelist)"
            elif in_struct and c < STRUCT_THR:
                status = f"DETECTED_BELOW_THRESHOLD (score={c}<0.5, tag in STRUCT whitelist)"
            else:
                status = f"DETECTED_NOT_IN_STRUCT_WHITELIST (score={c}, tag={t!r} has no STRUCT entry)"
            rows.append({"tag": t, "score": c, "status": status})
        diagnostics[u] = {
            "label": label,
            "related_tag_hits_0.3plus": rows if rows else "NO_RELATED_TAG_FIRED_AT_0.3+",
            "ref_caption": ref_captions[u],
        }
    print("\n[53] === signature-item diagnostics (4 user-named uids) ===", flush=True)
    for u, d in diagnostics.items():
        print(f"  u{u} ({d['label']}):", flush=True)
        print(f"    ref_caption: {d['ref_caption']}", flush=True)
        print(f"    related raw wd14 tags (>=0.3): {d['related_tag_hits_0.3plus']}", flush=True)

    # --- 15 samples, prioritizing rows whose STRUCT hits include an
    #     accessory/facial-mark tag (WEAR | FACE | ACCESSORY) ---
    mark_keys = set(tp.WEAR) | set(tp.FACE) | set(tp.ACCESSORY)

    def has_mark(u):
        return bool(set(struct_hits[u]) & mark_keys)

    manifest_uid_list = sorted(manifest_uids, key=int)
    marked = [u for u in manifest_uid_list if has_mark(u)]
    unmarked = [u for u in manifest_uid_list if not has_mark(u)]
    rng = random.Random(SEED)
    rng.shuffle(marked)
    rng.shuffle(unmarked)
    chosen = (marked + unmarked)[:N_SAMPLES]
    samples = [
        {"uid": u, "struct_hits": struct_hits[u], "has_accessory_or_mark_tag": has_mark(u),
         "ref_caption": ref_captions[u]}
        for u in chosen
    ]

    n_marked_total = len(marked)
    report = {
        "manifest_in": MANIFEST_IN,
        "manifest_out": MANIFEST_OUT,
        "devprobe12_refcaps_out": DEVPROBE_REFCAPS_OUT,
        "wd14_cache": CACHE_PATH,
        "wd14_model": WD14_MODEL,
        "wd14_general_threshold_full_dump": WD14_GENERAL_THRESHOLD,
        "struct_threshold": STRUCT_THR,
        "caption_format": "an anime illustration of a character: {phrase}",
        "no_scene_suffix": True,
        "counts": {
            "n_manifest_uids": len(manifest_uids),
            "n_devprobe_uids": len(devprobe_uids),
            "n_overlap": len(overlap),
            "n_union_tagged": len(uid2path),
            "n_manifest_uids_with_accessory_or_mark_tag": n_marked_total,
            "n_manifest_uids_without": len(unmarked),
            "n_rows_written_mix3": n_written,
        },
        "devprobe_phrase_crosscheck_mismatches": mismatches,
        "signature_item_diagnostics": diagnostics,
        "samples_15": samples,
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"[53] wrote {REPORT_OUT}", flush=True)

    print("\n=== FINAL COUNTS ===", flush=True)
    print(f"manifest uids={len(manifest_uids)} devprobe uids={len(devprobe_uids)} "
          f"union tagged={len(uid2path)}", flush=True)
    print(f"mix3 rows written={n_written}", flush=True)
    print(f"manifest uids with accessory/mark STRUCT tag={n_marked_total}/{len(manifest_uid_list)}",
          flush=True)
    print(f"\n=== {N_SAMPLES} SAMPLE ref_caption (accessory/mark-tagged rows prioritized) ===", flush=True)
    for s in samples:
        print(f"uid={s['uid']} marked={s['has_accessory_or_mark_tag']}", flush=True)
        print(f"  {s['ref_caption']}", flush=True)


if __name__ == "__main__":
    main()
