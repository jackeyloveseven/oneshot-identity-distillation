#!/usr/bin/env python
"""
43_baseline_krea2.py — a second trained/fine-tuned baseline for the paper's
comparison table (draft_v0.md §4.4): conradlocke/krea2-identity-edit, a
community LoRA on krea/Krea-2-Turbo (12.9B MMDiT) that gives a strong T2I
base identity-preservation via a single reference image, already deployed
and serving at http://localhost:8803 (see project_krea2_edit_8800 memory /
/mnt/local/krea2_edit_serve/worker.py — a separate, unrelated project). This
is a direct purpose-trained reference for identity preservation on a modern
MMDiT backbone.

Uses the SAME 30 held-out subjects + same prompt_B captions + same SEEDS
as probe/41 (KV injection), with an identical text prompt and no per-method
prompt engineering).

krea2-identity-edit is an edit model (requires >=1 ref image, no plain
T2I mode) but we deliberately do NOT hand-write edit-style instructions for
it — we pass it the exact same prompt_B generic scene caption every other
method gets, for a controlled comparison. This is a description of how we
invoke the baseline, not a change to cherry's own hard constraint (cherry's
mechanism/captions elsewhere in this project remain instruction-verb-free
because that is intrinsic to how KV injection works, not an evaluation
choice).

Calls the ALREADY-RUNNING worker over HTTP — does not load any model in
this process. Turbo recipe (steps=8, cfg=0.0) per the worker's documented
defaults (matches krea2_benchmark_run.py's own generation recipe).

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img $PY 43_baseline_krea2.py --stage gen
     $PY 38_gemini_judge.py --set krea2   (separate call)
"""
import os, json, base64, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
OUT = os.environ.get("KREA2_OUT", "/mnt/local/cherry_out/baseline_krea2")
TAGS_JSON = os.path.join(DATA, "holdout30_tags.json")

WORKER = "http://localhost:8803/api/edit"
SEEDS = [0, 1]
STEPS, CFG, GROUNDING_PX = 8, 0.0, 768
WORKERS = 4  # worker.py serializes generation behind a lock anyway; this just
             # overlaps network/encode/decode across requests


def img_path(uid, seed):
    return f"{OUT}/u{uid}_krea2_s{seed}.png"


def gen_one(uid, seed, prompt):
    out_p = img_path(uid, seed)
    if os.path.exists(out_p):
        return uid, seed, "skip"
    ref_b64 = base64.b64encode(open(f"{REF}/u{uid}_ref.png", "rb").read()).decode()
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
        raise RuntimeError(f"u{uid} s{seed}: {data['error']}")
    b64 = data["image_b64"].split(",", 1)[1]
    with open(out_p, "wb") as f:
        f.write(base64.b64decode(b64))
    return uid, seed, f"ok {data.get('secs','?')}s"


def stage_gen():
    tags = json.load(open(TAGS_JSON))
    todo = [(u, s, tags[u]["prompt_B"]) for u in tags for s in SEEDS
            if not os.path.exists(img_path(u, s))]
    print(f"[gen] {len(todo)} images remaining of {len(tags) * len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(gen_one, u, s, p): (u, s) for u, s, p in todo}
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
