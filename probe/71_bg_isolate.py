#!/usr/bin/env python
"""
71_bg_isolate.py -- background isolation preprocessing for the content-leakage
hypothesis (2026-07-19, ICEdit deep-read -> Diptych Prompting 2411.15466's
"content leakage" -> our PiP/collage defect).

Produces matted copies of the holdout30 reference images (anime subject
segmented via imgutils isnet-anime, composited onto a WHITE background),
keeping the exact `u{uid}_ref.png` filename so the existing pipeline
(30_kv_inject.py / 41_holdout_eval.py, which read `{REF}/u{uid}_ref.png`) can
be run verbatim with REF pointed at the matted dir -- a perfectly paired
original-vs-matted A/B with ZERO pipeline-code change.

Red-line compliant: pure inference-time input preprocessing. No base-model
change, no text-branch edit, no per-subject training.

Emits per-uid foreground fraction so bad mattes (isnet halo / amputation /
near-total removal) can be auto-flagged before spending GPU on generation --
the advisor's "sanity-check the mattes first" guardrail, quantified.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img \
       $PY 71_bg_isolate.py
"""
import os, json
import numpy as np
from PIL import Image
from imgutils.segment import segment_rgba_with_isnetis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
OUT_DIR = "/mnt/local/cherry_out/holdout30_matted_refs"
os.makedirs(OUT_DIR, exist_ok=True)

uids = list(json.load(open(os.path.join(DATA, "holdout30_tags.json"))).keys())
print(f"[bg_isolate] matting {len(uids)} holdout refs -> {OUT_DIR}", flush=True)

report = {}
for uid in uids:
    src = f"{REF}/u{uid}_ref.png"
    dst = f"{OUT_DIR}/u{uid}_ref.png"
    img = Image.open(src).convert("RGB")
    # isnet-anime -> RGBA (alpha = subject mask)
    _, rgba = segment_rgba_with_isnetis(img)
    alpha = np.array(rgba.split()[3])
    fg_frac = float((alpha > 127).mean())
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.split()[3])
    white.save(dst)
    report[uid] = round(fg_frac, 4)
    flag = "  <-- LOW fg (check!)" if fg_frac < 0.03 else ("  <-- high fg (weak bg?)" if fg_frac > 0.85 else "")
    print(f"  u{uid}: fg={fg_frac:.3f}{flag}", flush=True)

json.dump(report, open(os.path.join(DATA, "holdout30_matte_fgfrac.json"), "w"), indent=2)
lo = [u for u, f in report.items() if f < 0.03]
hi = [u for u, f in report.items() if f > 0.85]
print(f"\n[bg_isolate] done. {len(uids)} matted. LOW-fg(<3%): {lo or 'none'}  HIGH-fg(>85%): {hi or 'none'}", flush=True)
print(f"[bg_isolate] fgfrac report -> data/holdout30_matte_fgfrac.json", flush=True)
