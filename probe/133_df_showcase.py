#!/usr/bin/env python3
"""Qualitative showcase: run the locked Krea2 TSD pipeline on high-quality
datafactory character references. NOT part of the 30-identity evaluation; the
numbers/tables are untouched. Output goes to out/df_showcase_pilot/ only.

Method params and prompt templates are copied verbatim from the locked eval
(probe/120 + probe/121) so the distilled outputs are faithful to the paper's
method. Identity comes from the reference image; the short attribute phrase is
only a hint (the teacher prompt defers identity to the reference)."""
from __future__ import annotations

import concurrent.futures
import subprocess
import time
from pathlib import Path

from imgutils.detect import detect_heads
from PIL import Image

ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
PY = Path("/mnt/local/krea2_edit_serve/venv/bin/python")
PROBE = ROOT / "probe" / "115_krea2_identity_tsd.py"
LORA = Path("models/lora/krea2_identity_edit_v1_2.safetensors")
WH = Path("/home/ubuntu/datafactory_banked/pipe_single_image_notext")
OUT = ROOT / "out" / "df_showcase_pilot"
REFS = OUT / "refs"
LOG = OUT / "run.log"
SEED = 2027
PAD, SIZE = 0.35, 768

# Locked method params (from data/krea2_eval30_protocol.json).
M = dict(steps=12, inner_iters=3, lr=0.05, pseudo_step_size=1.0, teacher_weight=1.0,
         active_start=0.0, active_end=1.0, trust_region_rms=0.10,
         local_trust_region_rms=0.085, adaptive_trust_ratio=3.0,
         normalize="rms_token", residual_mode="raw", ref_boost=4.0, outside_weight=1.0)

SCENES = {
    "portrait": (640, 640,
        "in a centered close-up head-and-shoulders portrait inside a modern glass "
        "atrium, soft daylight, detailed face, looking toward the camera, coherent "
        "anatomy, no other people"),
    "rain": (512, 768,
        "standing on a rain-soaked neon city street at night, three-quarter body, "
        "holding a transparent umbrella, cinematic reflections, detailed visible "
        "face, coherent anatomy, no other people"),
    "kitchen": (768, 512,
        "preparing tea at a bright contemporary kitchen counter, waist-up, hands "
        "visible, turning toward the camera, warm morning light, detailed visible "
        "face, coherent anatomy, no other people"),
}

# best-of-n selection pool: (id, ref file, class_word, object_pron, possessive, attributes)
IDENTS = [
    ("tanjiro", WH / "ranked_smoke500_case_0192/refs/bot_01.png", "man", "him", "his",
     "dark red hair, hanafuda earrings, forehead scar, green checkered haori, red eyes"),
    ("zenitsu", WH / "ranked_smoke500_case_0202/refs/bot_01.png", "man", "him", "his",
     "blond hair, yellow haori with white triangle pattern, brown eyes"),
    ("rengoku", WH / "ranked_smoke500_case_0209/refs/bot_01.png", "man", "him", "his",
     "blond hair with red flame-like tips, red eyes, white flame-patterned cloak"),
    ("saber", WH / "ranked_smoke500_case_0210/refs/bot_01.png", "woman", "her", "her",
     "blond hair in a bun, green eyes, blue and silver armored dress"),
    ("scarlet", WH / "ranked_smoke500_case_0220/refs/user.png", "woman", "her", "her",
     "dark red and black hair, red eyes, black and grey gothic dress"),
]


def sq_crop(image: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    l, t, r, b = round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)
    canvas = Image.new("RGB", (r - l, b - t), "white")
    sb = (max(0, l), max(0, t), min(image.width, r), min(image.height, b))
    canvas.paste(image.crop(sb), (sb[0] - l, sb[1] - t))
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def prep_ref(ident) -> Path:
    name, src, *_ = ident
    dets = detect_heads(str(src))
    big = max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
    crop = sq_crop(Image.open(src).convert("RGB"), big[0])
    out = REFS / f"{name}.png"
    crop.save(out)
    return out


def prompts(ident, scene_desc):
    _, _, cw, obj, poss, attrs = ident
    scene_p = f"single anime {cw} {scene_desc}"
    caption_p = (f"Create a high-quality anime illustration of one {cw} as the only "
                 f"subject. Visible attributes: {attrs}. Depict {obj} {scene_desc}. "
                 "No collage and no split panel.")
    teacher_p = (f"Create a high-quality anime illustration of the exact {cw} shown in "
                 f"the reference image as the only subject. Automatically extracted "
                 f"visible attributes: {attrs}. Use the reference image for identity "
                 f"details beyond this short attribute list. Preserve {poss} identity "
                 f"without inventing a different person. Depict {obj} {scene_desc}. "
                 "Do not copy the reference background, framing, or pose. No collage "
                 "and no split panel.")
    return scene_p, caption_p, teacher_p


def run_job(ident, scene, refmap):
    name = ident[0]
    w, h, desc = SCENES[scene]
    scene_p, caption_p, teacher_p = prompts(ident, desc)
    neg = refmap[name]  # negative reference (a different identity)
    neg_ident = next(i for i in IDENTS if i[0] == neg)
    _, neg_caption, neg_teacher = prompts(neg_ident, desc)
    cmd = [str(PY), str(PROBE), "--uid", f"{name}_{scene}",
           "--ref", str(REFS / f"{name}.png"), "--negative-ref", str(REFS / f"{neg}.png"),
           "--lora-path", str(LORA), "--out-dir", str(OUT), "--seed", str(SEED),
           "--width", str(w), "--height", str(h),
           "--steps", str(M["steps"]), "--inner-iters", str(M["inner_iters"]),
           "--lr", str(M["lr"]), "--pseudo-step-size", str(M["pseudo_step_size"]),
           "--teacher-weight", str(M["teacher_weight"]), "--active-start", str(M["active_start"]),
           "--active-end", str(M["active_end"]), "--trust-region-rms", str(M["trust_region_rms"]),
           "--local-trust-region-rms", str(M["local_trust_region_rms"]),
           "--adaptive-trust-ratio", str(M["adaptive_trust_ratio"]),
           "--normalize", M["normalize"], "--residual-mode", M["residual_mode"],
           "--ref-boost", str(M["ref_boost"]), "--outside-weight", str(M["outside_weight"]),
           "--scene-prompt", scene_p, "--caption-prompt", caption_p,
           "--teacher-prompt", teacher_p, "--negative-teacher-prompt", neg_teacher]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    with LOG.open("a") as h:
        h.write(f"[{'OK' if ok else 'FAIL'}] {name}/{scene} {dt:.0f}s\n")
        if not ok:
            h.write(r.stderr[-2000:] + "\n")
    return name, scene, ok, dt


def main():
    REFS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    for ident in IDENTS:
        p = prep_ref(ident)
        with LOG.open("a") as h:
            h.write(f"cropped ref {ident[0]} -> {p}\n")
    names = [i[0] for i in IDENTS]
    refmap = {n: names[(k + 1) % len(names)] for k, n in enumerate(names)}  # wrong-ref = next
    jobs = [(ident, scene) for ident in IDENTS for scene in SCENES]
    with LOG.open("a") as h:
        h.write(f"launching {len(jobs)} jobs (2 workers)\n")
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_job, ident, scene, refmap) for ident, scene in jobs]
        for f in concurrent.futures.as_completed(futs):
            name, scene, ok, dt = f.result()
            done += 1
            with LOG.open("a") as h:
                h.write(f"progress {done}/{len(jobs)} ({name}/{scene} {dt:.0f}s)\n")
    with LOG.open("a") as h:
        h.write(f"ALL DONE in {(time.time()-t0)/60:.1f} min\n")


if __name__ == "__main__":
    main()
