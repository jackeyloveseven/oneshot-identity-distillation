#!/usr/bin/env python
"""
92_attn_quant_report.py -- reads the JSON dumps from 90_lumina_causal_driver.py
and 91_zimage_causal_driver.py and renders the comparison figure for the
decisive cross-model investigation (Z-Image vs Lumina-2.0 ref-KV attention
routing). Pure post-processing, no GPU needed.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZDIR = "/home/ubuntu/chengyanli/items/cherry/out/zimage_causal"
LDIR = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
OUT = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal/attn_quant_comparison.png"

z = json.load(open(f"{ZDIR}/summary.json"))
lA = json.load(open(f"{LDIR}/partA_summary.json"))
lB = json.load(open(f"{LDIR}/partB_summary.json"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- panel 1: per-layer ref-mass, Z (band16-25 strip) vs Lumina (band16-23 strip vs rope) ---
ax = axes[0]
def series(d, key="mean_ref_mass"):
    layers = sorted(int(k) for k in d.keys())
    return layers, [d[str(l)][key] for l in layers]

xl, yl = series(z["Z_strip_b16-25_w4"])
ax.plot(xl, yl, "o-", label="Z-Image strip (band16-25, w4)", color="tab:blue")
xl2, yl2 = series(z["Z_keep_b16-25_w4"])
ax.plot(xl2, yl2, "s--", label="Z-Image keep/post-RoPE (band16-25, w4)", color="tab:cyan")

xl3, yl3 = series(lA["A1_strip_b16-23_w2"])
ax.plot(xl3, yl3, "o-", label="Lumina strip (band16-23, w2)", color="tab:red")
xl4, yl4 = series(lA["A1_rope_b16-23_w2"])
ax.plot(xl4, yl4, "s--", label="Lumina rope/post-RoPE (band16-23, w2)", color="tab:orange")

ax.set_xlabel("layer id")
ax.set_ylabel("mean softmax attention mass\n(target image query -> ref-key columns)")
ax.set_title("Per-layer ref-attention: Z-Image vs Lumina-2.0")
ax.legend(fontsize=8)
ax.set_yscale("log")

# --- panel 2: causal ref_bias sweep on Lumina (band 8-15, strip) ---
ax = axes[1]
tags = sorted(lB.keys(), key=lambda t: lB[t]["ref_bias"])
biases = [lB[t]["ref_bias"] for t in tags]
masses = [sum(v["mean_ref_mass"] for v in lB[t]["per_layer"].values()) / len(lB[t]["per_layer"]) for t in tags]
ax.plot(biases, masses, "o-", color="tab:purple")
ax.set_xlabel("ref_bias (additive pre-softmax log-bias, all heads)")
ax.set_ylabel("mean ref-attention mass (band 8-15 avg)")
ax.set_title("Causal knob: forcing Lumina to attend to ref-key slot")
ax.axhline(1.0, ls=":", color="gray", lw=1)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print("saved:", OUT)

# --- print numeric summary tables ---
print("\n=== Z-Image (band16-25, w4) ===")
for l in xl:
    print(f"  L{l:2d}  strip={z['Z_strip_b16-25_w4'][str(l)]['mean_ref_mass']:.5f}  "
          f"keep={z['Z_keep_b16-25_w4'][str(l)]['mean_ref_mass']:.5f}")

print("\n=== Lumina (band16-23, w2) ===")
for l in xl3:
    print(f"  L{l:2d}  strip={lA['A1_strip_b16-23_w2'][str(l)]['mean_ref_mass']:.5f}  "
          f"rope={lA['A1_rope_b16-23_w2'][str(l)]['mean_ref_mass']:.5f}")

print("\n=== Lumina full-band (0-25) strip, w4 [known-collapse condition] ===")
for l, v in sorted(lA["A2_strip_bfull_w4"].items(), key=lambda kv: int(kv[0])):
    print(f"  L{l:>2}  ref_mass={v['mean_ref_mass']:.5f}")

print("\n=== Z-Image full-band (0-29) strip, w4 ===")
for l, v in sorted(z["Z_strip_bfull_w4"].items(), key=lambda kv: int(kv[0])):
    print(f"  L{l:>2}  ref_mass={v['mean_ref_mass']:.5f}")

print("\n=== Causal ref_bias sweep (Lumina band8-15, strip, w4) ===")
for t in tags:
    print(f"  ref_bias={lB[t]['ref_bias']:>5g}  mean_mass={sum(v['mean_ref_mass'] for v in lB[t]['per_layer'].values())/len(lB[t]['per_layer']):.5f}")
