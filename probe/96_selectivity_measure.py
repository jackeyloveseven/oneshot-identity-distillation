#!/usr/bin/env python
"""
96_selectivity_measure.py -- item-2 strengthener (pure measurement, no rescue).

Measures whether each target-image query's attention over the injected
position-free (strip) ref-key columns is PEAKED (content-addressed) or DIFFUSE
(color-bleed), for BOTH models at matched band/weight. Normalized entropy in
[0,1]: 0 = one ref token dominates (content-selective), 1 = uniform over all ref
tokens (no content addressing). This is the cleanest determinant test -- it
involves no forced bias and no generation degradation confound.

Runs each model in a SEPARATE subprocess (cannot import both pipelines into one
process). Aggregates into out/lumina/causal/selectivity.json.
"""
import os, sys, json, importlib.util

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
OUTD = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
os.makedirs(OUTD, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))


def agg(atl, band):
    out = {}
    for lid, recs in atl.items():
        if lid not in band:
            continue
        n = len(recs)
        out[lid] = {
            "ref_mass": sum(r["ref_mass"] for r in recs) / n,
            "ref_entropy_norm": sum(r["ref_entropy_norm"] for r in recs) / n,
            "ref_peak": sum(r["ref_peak"] for r in recs) / n,
        }
    layers = sorted(out)
    return {
        "per_layer": {str(l): out[l] for l in layers},
        "mean_entropy_norm": sum(out[l]["ref_entropy_norm"] for l in layers) / len(layers),
        "mean_peak": sum(out[l]["ref_peak"] for l in layers) / len(layers),
        "mean_ref_mass": sum(out[l]["ref_mass"] for l in layers) / len(layers),
    }


def run_lumina():
    spec = importlib.util.spec_from_file_location("l80", os.path.join(HERE, "80_lumina_inject.py"))
    L = importlib.util.module_from_spec(spec); sys.argv = ["80"]; spec.loader.exec_module(L)
    ref = L.resolve_ref("2")
    band = set(range(16, 24))
    L.generate(prompt="solo, single character, centered close-up anime portrait, clean anime style, "
               "simple plain white background", neg="", w=1.0, size=1024, refsize=512, steps=20, cfg=4.0,
               seed=0, ref_path=ref, band=band, refpos="strip", measure=True, ref_bias=0.0)
    return agg(L.ATTN_LOG, band)


def run_zimage():
    os.environ["REF"] = "/mnt/local/smoke_anime/anime_260623/img"
    spec = importlib.util.spec_from_file_location("z30", os.path.join(HERE, "30_kv_inject.py"))
    Z = importlib.util.module_from_spec(spec); sys.argv = ["30"]; spec.loader.exec_module(Z)
    band = set(range(16, 26))
    Z.INJ["band"] = band
    Z.generate(uid="2", w=4.0, refpos="strip", steps=20, cfg=5.0, seed=0, measure=True)
    return agg(Z.ATTN_LOG, band)


if MODE == "lumina":
    json.dump(run_lumina(), open(f"{OUTD}/_sel_lumina.json", "w"), indent=1); print("lumina sel done")
elif MODE == "zimage":
    json.dump(run_zimage(), open(f"{OUTD}/_sel_zimage.json", "w"), indent=1); print("zimage sel done")
else:
    import subprocess
    for m in ["lumina", "zimage"]:
        subprocess.run([sys.executable, __file__, m], check=True)
    z = json.load(open(f"{OUTD}/_sel_zimage.json"))
    l = json.load(open(f"{OUTD}/_sel_lumina.json"))
    combined = {"zimage_band16-25_strip": z, "lumina_band16-23_strip": l}
    json.dump(combined, open(f"{OUTD}/selectivity.json", "w"), indent=1)
    print("\n=== CONTENT-SELECTIVITY (position-free strip ref-attention) ===")
    print(f"Z-Image : mean_entropy_norm={z['mean_entropy_norm']:.4f}  mean_peak={z['mean_peak']:.5f}  mean_mass={z['mean_ref_mass']:.4f}")
    print(f"Lumina  : mean_entropy_norm={l['mean_entropy_norm']:.4f}  mean_peak={l['mean_peak']:.5f}  mean_mass={l['mean_ref_mass']:.4f}")
    print("(lower entropy / higher peak = more content-selective / less diffuse)")
    print("saved:", f"{OUTD}/selectivity.json")
