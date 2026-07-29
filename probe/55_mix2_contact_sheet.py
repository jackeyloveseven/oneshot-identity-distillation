#!/usr/bin/env python
"""
55_mix2_contact_sheet.py -- mix2 vs mix1 three-way dev-probe visual review:
side-by-side contact sheets comparing ref / strip-baseline /
mix1_step_{2000,4000,6000} / mix2_step_{2000,4000,6000} for each of the 12
dev-probe subjects (data/devprobe12_tags.json, see probe/46_devprobe.py's
docstring for how this 12-uid set was built), one sheet per seed. Built for
eyeballing whether mix2 (the newer training mix) holds up on accessories /
marks / species axes against mix1 (the earlier mix) at matching checkpoint
steps. Read-only: this script only reads probe/46_devprobe's and the
mix1/mix2 training runs' already-generated output under
out/devprobe/ and writes two new PNGs; it never touches
47_m1_contact_sheet.py or any other script.

This is a straight column-swap of probe/47_m1_contact_sheet.py: same
tiling/label/missing-placeholder/font-loading logic, only COLUMNS and the
output filenames differ (5 columns -> 8 columns: ref, strip, mix1 x3
steps, mix2 x3 steps).

Layout per sheet: 12 rows (one per uid) x 8 image columns (ref, strip,
mix1_2k, mix1_4k, mix1_6k, mix2_2k, mix2_4k, mix2_6k), each cell a 256x256
white square with the source image scaled to fit (aspect preserved, no
crop/stretch) and centered. A narrow left label column carries the uid
text per row; a header row above carries the column names. A missing
source image renders as a gray cell with "MISSING" text instead of
crashing, and every missing path is collected into a report printed at
the end.

Run:
  /home/ubuntu/miniconda3/bin/python 55_mix2_contact_sheet.py

Output:
  out/trainlight/mix2_compare_s0.png
  out/trainlight/mix2_compare_s1.png
"""
import os
import json

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

TAGS_JSON = os.path.join(DATA, "devprobe12_tags.json")
REF_IMG_DIR = "/mnt/local/smoke_anime/anime_260623/img"
DEVPROBE_ROOT = "out/devprobe"
OUT_DIR = "out/trainlight"

SEEDS = [0, 1]
CELL = 256          # each image cell is CELL x CELL, white background
LABEL_COL_W = 110    # narrow left column for the uid text
HEADER_H = 44        # top row for column names

# (column display name, path-builder(uid, seed)) -- order = left-to-right
COLUMNS = [
    ("ref", lambda uid, seed: f"{REF_IMG_DIR}/u{uid}_ref.png"),
    ("strip", lambda uid, seed: f"{DEVPROBE_ROOT}/strip/u{uid}_strip_s{seed}.png"),
    ("mix1_2k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix1_step2000/u{uid}_lora_s{seed}.png"),
    ("mix1_4k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix1_step4000/u{uid}_lora_s{seed}.png"),
    ("mix1_6k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix1_step6000/u{uid}_lora_s{seed}.png"),
    ("mix2_2k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix2_step2000/u{uid}_lora_s{seed}.png"),
    ("mix2_4k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix2_step4000/u{uid}_lora_s{seed}.png"),
    ("mix2_6k", lambda uid, seed: f"{DEVPROBE_ROOT}/mix2_step6000/u{uid}_lora_s{seed}.png"),
]


def load_uids():
    """devprobe12_tags.json's own keys, skipping metadata keys that start
    with '_' (e.g. '_amendment', which documents the human-review swap that
    produced this 12-uid set -- see 46_devprobe.py's load_tags())."""
    tags = json.load(open(TAGS_JSON))
    uids = [k for k in tags if not k.startswith("_")]
    return uids


def draw_centered_text(draw, xy_box, text, font, fill):
    """xy_box = (x0, y0, x1, y1); centers text inside that box."""
    x0, y0, x1, y1 = xy_box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = x0 + (x1 - x0 - tw) / 2 - bbox[0]
    cy = y0 + (y1 - y0 - th) / 2 - bbox[1]
    draw.text((cx, cy), text, font=font, fill=fill)


def make_cell(path, missing_log, col_name, uid, seed):
    """Returns a CELL x CELL RGB image: the source scaled to fit (aspect
    preserved, centered on white), or a gray MISSING placeholder if the
    path doesn't exist / fails to open."""
    cell = Image.new("RGB", (CELL, CELL), "white")
    if not os.path.exists(path):
        missing_log.append((uid, seed, col_name, path))
        d = ImageDraw.Draw(cell)
        d.rectangle([0, 0, CELL, CELL], fill=(180, 180, 180))
        font = ImageFont.load_default(size=16)
        draw_centered_text(d, (0, 0, CELL, CELL), "MISSING", font, fill="black")
        return cell
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        missing_log.append((uid, seed, col_name, f"{path} (open failed: {e})"))
        d = ImageDraw.Draw(cell)
        d.rectangle([0, 0, CELL, CELL], fill=(180, 180, 180))
        font = ImageFont.load_default(size=16)
        draw_centered_text(d, (0, 0, CELL, CELL), "MISSING", font, fill="black")
        return cell
    scale = min(CELL / im.width, CELL / im.height)
    new_w, new_h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    off = ((CELL - new_w) // 2, (CELL - new_h) // 2)
    cell.paste(im, off)
    return cell


def build_sheet(uids, seed, missing_log):
    n_rows = len(uids)
    n_cols = len(COLUMNS)
    W = LABEL_COL_W + n_cols * CELL
    H = HEADER_H + n_rows * CELL
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=18)

    # header row: column names, corner cell above the label column left blank
    for c, (col_name, _) in enumerate(COLUMNS):
        x0 = LABEL_COL_W + c * CELL
        draw_centered_text(draw, (x0, 0, x0 + CELL, HEADER_H), col_name, header_font, fill="black")
    draw.line([(0, HEADER_H), (W, HEADER_H)], fill="black", width=2)
    draw.line([(LABEL_COL_W, 0), (LABEL_COL_W, H)], fill="black", width=2)

    for r, uid in enumerate(uids):
        y0 = HEADER_H + r * CELL
        draw_centered_text(draw, (0, y0, LABEL_COL_W, y0 + CELL), f"u{uid}", label_font, fill="black")
        for c, (col_name, path_fn) in enumerate(COLUMNS):
            path = path_fn(uid, seed)
            cell = make_cell(path, missing_log, col_name, uid, seed)
            x0 = LABEL_COL_W + c * CELL
            canvas.paste(cell, (x0, y0))
        draw.line([(0, y0), (W, y0)], fill=(200, 200, 200), width=1)

    return canvas


def main():
    uids = load_uids()
    print(f"[55_mix2_contact_sheet] {len(uids)} uids from {TAGS_JSON}: {uids}", flush=True)
    assert len(uids) == 12, f"expected 12 uids, got {len(uids)}"

    os.makedirs(OUT_DIR, exist_ok=True)
    missing_log = []
    outputs = []
    for seed in SEEDS:
        canvas = build_sheet(uids, seed, missing_log)
        out_path = os.path.join(OUT_DIR, f"mix2_compare_s{seed}.png")
        canvas.save(out_path)
        outputs.append((out_path, canvas.size))
        print(f"[55_mix2_contact_sheet] wrote {out_path}  size={canvas.size}", flush=True)

    print(f"[55_mix2_contact_sheet] missing images: {len(missing_log)}", flush=True)
    for uid, seed, col_name, path in missing_log:
        print(f"[55_mix2_contact_sheet]   MISSING u{uid} seed={seed} col={col_name}: {path}", flush=True)

    return outputs, missing_log


if __name__ == "__main__":
    main()
