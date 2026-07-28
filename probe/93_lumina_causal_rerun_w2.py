#!/usr/bin/env python
"""
93_lumina_causal_rerun_w2.py -- corrected causal ref_bias sweep.

90_lumina_causal_driver.py's Part B used w=4.0 at band 8-15, but the
confirmed-coherent baseline for that band (out/lumina/u2_depth_w2_b8-15.png)
was at w=2.0 -- w=4 already over-injects (rb=0 baseline came out nearly blank,
not the clean portrait), confounding the ref_bias sweep. This reruns Part B at
w=2.0 with a finer low-end bias grid to see whether a gentler forced-attention
push (starting from a genuinely coherent baseline) can route identity through,
vs. just washing to noise as at w=4.
"""
import importlib.util, os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
os.makedirs(OUTD, exist_ok=True)

spec = importlib.util.spec_from_file_location("lumina80", os.path.join(HERE, "80_lumina_inject.py"))
L = importlib.util.module_from_spec(spec)
sys.argv = ["80_lumina_inject.py"]
spec.loader.exec_module(L)

PROMPT = ("solo, single character, centered close-up anime portrait, upper body, "
          "detailed face, clean anime style, simple plain white background")
NEG = ""
SUBJECT = "2"
ref_path = L.resolve_ref(SUBJECT)


def run(tag, **kw):
    t0 = time.time()
    defaults = dict(prompt=PROMPT, neg=NEG, size=1024, refsize=512, steps=30, cfg=4.0, seed=0,
                     ref_path=ref_path)
    defaults.update(kw)
    im = L.generate(**defaults)
    im.save(f"{OUTD}/{tag}.png")
    dt = time.time() - t0
    summary = None
    if kw.get("measure"):
        summary = {lid: {
            "mean_ref_mass": sum(r["ref_mass"] for r in recs) / len(recs),
        } for lid, recs in L.ATTN_LOG.items()}
        avg = sum(v["mean_ref_mass"] for v in summary.values()) / len(summary)
        print(f"[{tag}] done in {dt:.1f}s  avg_ref_mass={avg:.4f}", flush=True)
    else:
        print(f"[{tag}] done in {dt:.1f}s", flush=True)
    return im, summary


band_b = set(range(8, 16))
biases = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
partB2 = {}
imgs = []
for rb in biases:
    tag = f"C_w2_biassweep_rb{rb:g}"
    im, s = run(tag, w=2.0, band=band_b, refpos="strip", measure=True, ref_bias=rb)
    partB2[tag] = {"ref_bias": rb, "per_layer": s}
    imgs.append((f"w2 rb={rb:g}", im))

with open(f"{OUTD}/partC_w2_summary.json", "w") as f:
    json.dump(partB2, f, indent=1)
print("saved partC_w2_summary.json", flush=True)

from PIL import Image, ImageDraw
def label(im, txt, cell):
    im = im.resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, min(cell, 340), 30], fill="black")
    d.text((6, 7), txt, fill="white")
    return im

cell = 512
rimg = Image.open(ref_path).convert("RGB")
im_off, _ = run("C_off", w="off", ref_path=None, band=band_b, refpos="strip")
panels = [("ref (pink hair)", rimg), ("off (no inject)", im_off)] + imgs
canvas = Image.new("RGB", (cell * len(panels), cell), "white")
for c, (txt, im) in enumerate(panels):
    canvas.paste(label(im, txt, cell), (c * cell, 0))
canvas.save(f"{OUTD}/montage_partC_w2_biassweep.png")
print("saved montage:", f"{OUTD}/montage_partC_w2_biassweep.png", flush=True)
print("ALL DONE", flush=True)
