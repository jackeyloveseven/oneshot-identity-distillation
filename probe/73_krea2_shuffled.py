#!/usr/bin/env python
"""
73_krea2_shuffled.py — shuffled-ref FLOOR for the krea2-identity-edit baseline
(probe/43_baseline_krea2.py), symmetric with cherry's own shuffled-ref control
(docs/design/2026-07-11_traininglight_design.md §11.1: "对照·shuffled-ref
（喂错主体 ref）" 5% clean-yes, gemini_judge_shuffledref_holdout30.json).

The paper claims "we're identity-on-par with krea2" (base 6k clean-yes 38% vs
krea2 33%, CI overlap -> par per §11.6 discipline: never "beat"). For that
claim to be airtight to a reviewer, krea2's 33% number itself must be shown to
actually depend on being fed the RIGHT reference — i.e. krea2 must ALSO
collapse when fed a WRONG subject's ref, exactly like cherry does. This script
generates that floor: same 30 held-out subjects (data/holdout30_tags.json,
same set as probe/41/42/43), each target uid's own prompt_B, but the target's
ref image swapped for a DIFFERENT (deranged, no fixed points) subject's ref.

Mirrors probe/43_baseline_krea2.py's worker-calling code verbatim (same
WORKER, same payload shape, same STEPS/CFG/GROUNDING_PX -- read from 43, not
re-derived) -- only the ref-image selection and output naming differ.

Calls the ALREADY-RUNNING worker over HTTP at localhost:8803 -- does not load
any model in this process.

Derangement: random.Random(42) over the 30 uids (sorted by int for a fixed
canonical order regardless of json key order), rejection-sampled until no
uid maps to itself. Deterministic and reproducible across runs since the
RNG's shuffle-and-reject trace is itself deterministic for a fixed seed.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img $PY 73_krea2_shuffled.py --stage gen
     $PY 38_gemini_judge.py --set krea2_shuffled   (separate call)
"""
import os, json, base64, argparse, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
# env.sh's default REF (revref_30k) is DEAD (dir cleared); hardcode the live
# anime_260623 source and do NOT trust $REF -- same fix probe/72 applies.
REF = "/mnt/local/smoke_anime/anime_260623/img"
OUT = os.environ.get("KREA2_SHUF_OUT", "/mnt/local/cherry_out/baseline_krea2_shuffled")
TAGS_JSON = os.path.join(DATA, "holdout30_tags.json")

WORKER = "http://localhost:8803/api/edit"
SEEDS = [0, 1]
# read verbatim from probe/43_baseline_krea2.py -- do not change independently
# of that file, the whole point is a symmetric floor for the SAME recipe.
STEPS, CFG, GROUNDING_PX = 8, 0.0, 768
WORKERS = 4  # worker.py serializes generation behind a lock anyway; this just
             # overlaps network/encode/decode across requests

DERANGE_SEED = 42


def build_mapping(uids):
    """Deterministic derangement: target uid -> a DIFFERENT source uid whose
    ref image gets fed instead. Rejection-sample random.Random(42).shuffle()
    until no fixed points; asserts this before returning so a bug here fails
    loudly rather than silently degrading into a partial-identity control."""
    ordered = sorted(uids, key=int)
    n = len(ordered)
    rng = random.Random(DERANGE_SEED)
    idx = list(range(n))
    tries = 0
    while True:
        perm = idx[:]
        rng.shuffle(perm)
        tries += 1
        if all(perm[i] != idx[i] for i in range(n)):
            break
    mapping = {ordered[i]: ordered[perm[i]] for i in range(n)}
    for target, source in mapping.items():
        assert source != target, f"derangement has a fixed point: u{target}"
    print(f"[derange] {n} uids, seed={DERANGE_SEED}, {tries} shuffle attempt(s) "
          f"to reject fixed points", flush=True)
    print(f"[derange] mapping (target -> source): "
          f"{json.dumps(mapping, indent=1)}", flush=True)
    return mapping


def img_path(uid, seed):
    return f"{OUT}/u{uid}_krea2shuf_s{seed}.png"


def gen_one(target_uid, source_uid, seed, prompt):
    out_p = img_path(target_uid, seed)
    if os.path.exists(out_p):
        return target_uid, seed, "skip"
    ref_b64 = base64.b64encode(open(f"{REF}/u{source_uid}_ref.png", "rb").read()).decode()
    payload = {
        "image_b64": ref_b64,
        "prompt": prompt,
        "steps": STEPS,
        "cfg": CFG,
        "seed": seed,
        "grounding_px": GROUNDING_PX,
    }
    r = requests.post(WORKER, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"u{target_uid} (src u{source_uid}) s{seed}: {data['error']}")
    b64 = data["image_b64"].split(",", 1)[1]
    with open(out_p, "wb") as f:
        f.write(base64.b64decode(b64))
    return target_uid, seed, f"ok {data.get('secs','?')}s"


def stage_gen():
    tags = json.load(open(TAGS_JSON))
    mapping = build_mapping(list(tags.keys()))
    todo = [(u, mapping[u], s, tags[u]["prompt_B"]) for u in tags for s in SEEDS
            if not os.path.exists(img_path(u, s))]
    print(f"[gen] {len(todo)} images remaining of {len(tags) * len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "_derangement_map.json"), "w") as f:
        json.dump(mapping, f, indent=1)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(gen_one, u, src, s, p): (u, s) for u, src, s, p in todo}
        for fut in as_completed(futs):
            u, s = futs[fut]
            try:
                uid, seed, status = fut.result()
                done += 1
                print(f"[gen] u{uid} seed={seed} {status} ({done}/{len(todo)})", flush=True)
            except Exception as e:
                print(f"[gen] u{u} seed={s} FAILED: {e}", flush=True)
    print(f"[gen] batch done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen"])
    a = ap.parse_args()
    {"gen": stage_gen}[a.stage]()
