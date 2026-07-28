#!/usr/bin/env python
"""
76_timegate_sweep.py -- temporal-gate hypothesis (eyeball-first).

Idea: all three tried defect-fixes were SPATIAL localization (layer band, head
subset, token clip) and all failed -> identity is distributed in space. But the
production schedule is `taper`, which injects STRONGEST at the early high-noise
composition-setting steps. If the collage/PiP defect is the ref dragging its own
composition into that early layout pass, then gating injection to LATE (low-noise
detail) steps -- after the base has locked a clean single-subject composition --
could stamp identity WITHOUT the collage. That would mean the defect is localized
in DENOISING TIME even though identity is distributed across layers/heads.

This is a cheap multimodal eyeball run (per user's eyeball-first rule): a few
subjects x {taper(production ref), const, late50, late30, late50@w6} on the strip
production band. Look before quantifying.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img \
       HG_SUBJECTS=192,1317,2636,2542,4073 $PY 76_timegate_sweep.py
"""
import os, sys, json, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)          # loads pure-T2I base + installs InjectProcessor

REFPOS = os.environ.get("HG_REFPOS", "strip")
BAND   = os.environ.get("HG_BAND", "21-25")
SIZE   = int(os.environ.get("HG_SIZE", "512"))
kv.INJ["band"] = kv.parse_band(BAND)
print(f"[timegate] refpos={REFPOS} band={BAND} size={SIZE}", flush=True)

OUT = "/mnt/local/cherry_out/timegate"
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, "..", "data")
tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))

SUBJECTS = os.environ.get("HG_SUBJECTS", "192,1317,2636,2542,4073").split(",")
# (name, weight, generate-kwargs)
CONFIGS = [
    ("taper_w4",  4.0, dict(sched_mode="taper")),                 # production reference (current defect)
    ("const_w4",  4.0, dict(sched_mode="const")),                 # inject every step
    ("late50_w4", 4.0, dict(sched_mode="late", sched_frac=0.5)),  # inject last 50% only
    ("late30_w4", 4.0, dict(sched_mode="late", sched_frac=0.3)),  # inject last 30% only
    ("late50_w6", 6.0, dict(sched_mode="late", sched_frac=0.5)),  # stronger late (compensate low-noise weakness)
    ("late30_w8", 8.0, dict(sched_mode="late", sched_frac=0.3)),  # strong + short late window
]

for uid in SUBJECTS:
    uid = uid.strip()
    if uid not in tags:
        print(f"[timegate] skip unknown uid {uid}", flush=True); continue
    prompt = tags[uid]["prompt_B"]
    for name, w, cfg in CONFIGS:
        outp = f"{OUT}/u{uid}_{name}_s0.png"
        if os.path.exists(outp):
            print(f"[timegate] skip {outp}", flush=True); continue
        gkw = dict(refpos=REFPOS, seed=0, prompt=prompt, size=SIZE, refsize=512)
        if REFPOS == "lowfreq":
            gkw["lowfreq_k"] = int(os.environ.get("HG_LOWFREQ_K", "8"))
        try:
            im = kv.generate(uid, w, **gkw, **cfg)
            im.save(outp)
            print(f"[timegate] u{uid} {name} DONE -> {outp}", flush=True)
        except Exception as e:
            print(f"[timegate] u{uid} {name} FAIL: {e}", flush=True)
print("[timegate] all done", flush=True)
