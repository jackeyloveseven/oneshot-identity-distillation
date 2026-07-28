#!/usr/bin/env python
"""E3 corrective: vnorm's real operating point is low w (norm already matched to
target, so w is a multiplier ON TOP). Test vnorm at w in {0.5,1,2}, const sched."""
import importlib.util
from PIL import Image
spec = importlib.util.spec_from_file_location("kv", "/home/ubuntu/chengyanli/items/cherry/probe/30_kv_inject.py")
kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
OUT, REF = kv.OUT, kv.REF
for uid in ["29998", "9172"]:
    rimg = Image.open(f"{REF}/u{uid}_ref.png").convert("RGB")
    row = [kv.label(rimg, f"ref {uid}", 640)]
    for wv in [0.5, 1.0, 2.0]:
        print(f"[e3vn] u{uid} vnorm w={wv}", flush=True)
        im = kv.generate(uid, wv, route="off", sched_mode="const", vnorm=True, steps=50)
        im.save(f"{OUT}/u{uid}_e3_vn_const_w{wv:g}.png")
        row.append(kv.label(im, f"vn w{wv:g}", 640))
    canvas = Image.new("RGB", (len(row) * 640, 640), "white")
    for c, im in enumerate(row):
        canvas.paste(im, (c * 640, 0))
    canvas.save(f"{OUT}/montage_e3vn_{uid}.png")
    print("saved", f"{OUT}/montage_e3vn_{uid}.png", flush=True)
