#!/usr/bin/env python
"""
91_zimage_causal_driver.py -- decisive cross-model investigation driver
(Z-Image side, the "identity transfers" reference point).

Imports probe/30_kv_inject.py as a module (single model load) and runs the
matching measurement configs using its instrumented InjectProcessor
(measure / ref_bias, both default-off in 30 itself). Does NOT modify 30's
existing behavior.

NOTE: env.sh's REF (revref_30k) is currently EMPTY on disk (images were
cleaned up). This driver overrides REF to the anime_260623 pool used by the
Lumina side (same uid convention, e.g. u2_ref.png) so both models are probed
against the IDENTICAL reference photo -- a cleaner paired comparison than
mismatched subject pools.

Outputs -> /home/ubuntu/chengyanli/items/cherry/out/zimage_causal/
"""
import os
os.environ["REF"] = "/mnt/local/smoke_anime/anime_260623/img"  # override BEFORE importing 30 (module-level read)

import importlib.util, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = "/home/ubuntu/chengyanli/items/cherry/out/zimage_causal"
os.makedirs(OUTD, exist_ok=True)

spec = importlib.util.spec_from_file_location("zimg30", os.path.join(HERE, "30_kv_inject.py"))
Z = importlib.util.module_from_spec(spec)
sys.argv = ["30_kv_inject.py"]
spec.loader.exec_module(Z)

SUBJECT = "2"
log = {}


def run(tag, **kw):
    t0 = time.time()
    defaults = dict(uid=SUBJECT, steps=30, cfg=5.0, seed=0)
    defaults.update(kw)
    im = Z.generate(**defaults)
    im.save(f"{OUTD}/{tag}.png")
    dt = time.time() - t0
    summary = None
    if kw.get("measure"):
        summary = {lid: {
            "mean_ref_mass": sum(r["ref_mass"] for r in recs) / len(recs),
            "mean_k_ref_norm": sum(r["k_ref_norm"] for r in recs) / len(recs),
            "mean_k_ctx_norm": sum(r["k_ctx_norm"] for r in recs) / len(recs),
        } for lid, recs in Z.ATTN_LOG.items()}
    print(f"[{tag}] done in {dt:.1f}s" + (f"  layers_measured={len(summary)}" if summary else ""), flush=True)
    return im, summary


print("=== Z-Image ref-attention quantification (item 1, the working-identity reference point) ===", flush=True)

# Z's own known-good identity band (16-25), default refpos=strip (position-free
# pre-RoPE key) -- the SAME representation Lumina fails to attend to.
Z.INJ["band"] = set(range(16, 26))
_, s_strip_1625 = run("Z_strip_b16-25_w4", w=4.0, refpos="strip", measure=True)
log["Z_strip_b16-25_w4"] = s_strip_1625

# Positive control mirroring Lumina's strip-vs-rope comparison: "keep" = raw
# post-RoPE key (true position), same band/weight. Tests whether Z's attention
# to the ref slot is ITSELF phase-dependent (symmetric determinant check).
Z.INJ["band"] = set(range(16, 26))
_, s_keep_1625 = run("Z_keep_b16-25_w4", w=4.0, refpos="keep", measure=True)
log["Z_keep_b16-25_w4"] = s_keep_1625

# Full-depth (0-29) strip, matching Lumina's full-band measurement condition
# for a direct record (Z is known to degrade-but-not-collapse at wide bands --
# see out/u29998_e4_band10-29.png -- unlike Lumina's total collapse).
Z.INJ["band"] = set(range(30))
_, s_strip_full = run("Z_strip_bfull_w4", w=4.0, refpos="strip", measure=True)
log["Z_strip_bfull_w4"] = s_strip_full

with open(f"{OUTD}/summary.json", "w") as f:
    json.dump(log, f, indent=1)
print("saved summary.json", flush=True)
print("ALL DONE", flush=True)
