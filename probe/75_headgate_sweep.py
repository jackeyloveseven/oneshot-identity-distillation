#!/usr/bin/env python
"""
75_headgate_sweep.py -- lit-scan idea #1: head-level ref-injection gate.

Imports probe/30_kv_inject.py as a module (loads the frozen base once) and, on
the strip production band (21-25), generates a few subjects under several
head-gate configs to eyeball whether restricting ref K/V injection to a subset
of attention heads (by per-head ref-affinity) cuts the collage/PiP defect while
keeping identity. Training-free; red-line compliant.

Configs: baseline (all heads) / top-half / top-quarter / bottom-half heads.
Output montage-ready under /mnt/local/cherry_out/headgate/.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img \
       $PY 75_headgate_sweep.py
"""
import os, sys, json, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)          # loads the pure-T2I base + installs InjectProcessor

# detect head count
H = None
for attr in ("num_attention_heads", "num_heads", "n_heads", "num_heads_kv"):
    H = getattr(getattr(kv.tf, "config", object()), attr, None)
    if H:
        break
if not H:
    H = 24
print(f"[headgate] detected H={H}", flush=True)

REFPOS    = os.environ.get("HG_REFPOS", "strip")
BAND      = os.environ.get("HG_BAND", "21-25")
LOWFREQ_K = int(os.environ.get("HG_LOWFREQ_K", "8"))
SIZE      = int(os.environ.get("HG_SIZE", "512"))
TAG       = os.environ.get("HG_TAG", "")       # output filename suffix, e.g. "_lf1024"
kv.INJ["band"] = kv.parse_band(BAND)
print(f"[headgate] refpos={REFPOS} band={BAND} size={SIZE} tag={TAG!r}", flush=True)
OUT = "/mnt/local/cherry_out/headgate"
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, "..", "data")
tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))

SUBJECTS = os.environ.get("HG_SUBJECTS", "192,1317,2636,2542").split(",")
CONFIGS = [
    ("baseline", dict(head_gate="off")),
    ("tophalf",  dict(head_gate="top",    head_keep=H // 2)),
    ("topqtr",   dict(head_gate="top",    head_keep=max(1, H // 4))),
    ("bothalf",  dict(head_gate="bottom", head_keep=H // 2)),
    # matched-count controls (all keep=7): does the SELECTION (top ref-affinity)
    # carry identity, or just the COUNT (weaker injection)? -> advisor's decisive test
    ("random7",  dict(head_gate="random", head_keep=max(1, H // 4), head_seed=0)),
    ("bottom7",  dict(head_gate="bottom", head_keep=max(1, H // 4))),
    # idea #2 (outlier ref-token clipping): full injection, drop only high-norm
    # ref tokens (sink/watermark candidates). clip_rand = same-count random drop
    # control -> is it the norm-SELECTION or just dropping some tokens (count)?
    ("clip2",    dict(head_gate="off", clip_ref=2.0, clip_mode="norm")),
    ("cliprand", dict(head_gate="off", clip_ref=2.0, clip_mode="random", head_seed=0)),
]
_want = os.environ.get("HG_CONFIGS")
if _want:
    _w = set(_want.split(","))
    CONFIGS = [(n, c) for n, c in CONFIGS if n in _w]
print(f"[headgate] configs: {[(n, c) for n, c in CONFIGS]}", flush=True)

for uid in SUBJECTS:
    prompt = tags[uid]["prompt_B"]
    for name, cfg in CONFIGS:
        outp = f"{OUT}/u{uid}_{name}{TAG}_s0.png"
        if os.path.exists(outp):
            print(f"[headgate] skip {outp}", flush=True)
            continue
        gkw = dict(refpos=REFPOS, sched_mode="taper", seed=0, prompt=prompt,
                   size=SIZE, refsize=512)
        if REFPOS == "lowfreq":
            gkw["lowfreq_k"] = LOWFREQ_K
        im = kv.generate(uid, 4.0, **gkw, **cfg)
        im.save(outp)
        print(f"[headgate] u{uid} {name} DONE -> {outp}", flush=True)
print("[headgate] all done", flush=True)
