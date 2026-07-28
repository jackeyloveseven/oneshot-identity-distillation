#!/usr/bin/env python
"""
63_spectral_flatness_predictor.py -- cheap, VLM-free candidate predictor for
the tiled/collaged composition-defect category this project's Gemini judge
flags as `visible_defect`. Motivated by a literature lead (SEGA, arXiv
2605.22668) that hits the same downstream symptom (RoPE positional-offset
mismatch -> attention dilution -> repetitive/tiling DiT artifacts) via a
different trigger (resolution extrapolation, not reference-KV injection):
their proxy is spectral flatness (Wiener entropy) of the generated
latent -- periodic/tiled content concentrates FFT energy in narrow bands
(low flatness); natural, non-repetitive content spreads energy broadly
(high flatness).

Adapted here to PIXEL space (grayscale luminance) rather than the latent,
since these images were only saved as PNG, not with latents preserved --
a reasonable portability adaptation, and arguably more direct since tiling
is fundamentally a pixel-level visual phenomenon.

SF(image) = geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum)
          in (0, 1]. Low SF -> energy concentrated in few frequencies
          (periodic/tiled). High SF -> broadly spread energy (natural).

Correlated against existing Gemini-judged visible_defect labels on TWO
SEPARATE, internally-consistent subsets -- NOT pooled, per advisor
guidance, since they're different instruments/prompts/sets:
  (1) widerband_decompose (probe/62): 72 images, 3 arms, joint
      ref-comparison HOLDOUT_PROMPT judge.
  (2) lowfreq10/center10 narrow-band diagnostic (probe/61): 40 images,
      2 arms, defect-only single-image judge.

This is exploratory and may simply not transfer from SEGA's trigger
(resolution extrapolation) to this project's trigger (reference-KV
injection) -- reported as-is; a null result here is a legitimate,
reportable outcome, not a failure to fix.

Run: $PY 63_spectral_flatness_predictor.py
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def spectral_flatness(img_path, resize=512):
    im = Image.open(img_path).convert("L").resize((resize, resize), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float64)
    arr = arr - arr.mean()
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P = P.flatten()
    P = P[P > 1e-12]  # drop exact zeros (DC-adjacent bins after mean removal) to keep log finite
    log_gm = np.mean(np.log(P))
    gm = np.exp(log_gm)
    am = np.mean(P)
    return float(gm / am)


def subset_widerband():
    verdicts = json.load(open(os.path.join(DATA, "gemini_judge_widerband_decompose.json")))
    tags = json.load(open(os.path.join(DATA, "devprobe12_tags.json")))
    uids = [u for u in tags if not u.startswith("_")]
    arms = ["strip_narrow", "lowfreq_wide", "strip_wide"]
    rows = []
    for uid in uids:
        for arm in arms:
            for s in [0, 1]:
                pid = f"{uid}_{arm}_s{s}"
                v = verdicts.get(pid)
                if not v or v.get("visible_defect") not in ("yes", "no"):
                    continue
                p = f"/mnt/local/cherry_out/widerband_decompose/u{uid}_wbd_{arm}_s{s}.png"
                if os.path.exists(p):
                    rows.append((pid, arm, v["visible_defect"], spectral_flatness(p)))
    return rows


def subset_narrowdiag():
    verdicts = json.load(open(os.path.join(DATA, "defect_only_diagnostic_band_width.json")))
    tmpl = {"lowfreq_narrow_21-25": "/mnt/local/cherry_out/lowfreq10/u{uid}_lf8_s{s}.png",
            "center_narrow_21-25": "/mnt/local/cherry_out/center10/u{uid}_center_s{s}.png"}
    rows = []
    for pid, v in verdicts.items():
        if v.get("visible_defect") not in ("yes", "no"):
            continue
        arm, uid_s = pid.split("__")  # uid_s = "u{uid}_s{s}"
        uid = uid_s.split("_s")[0][1:]  # strip leading 'u'
        s = uid_s.split("_s")[1]
        p = tmpl[arm].format(uid=uid, s=s)
        if os.path.exists(p):
            rows.append((pid, arm, v["visible_defect"], spectral_flatness(p)))
    return rows


def report(name, rows):
    print(f"\n=== {name} (n={len(rows)}) ===")
    yes_sf = [r[3] for r in rows if r[2] == "yes"]
    no_sf = [r[3] for r in rows if r[2] == "no"]
    print(f"  visible_defect=yes: n={len(yes_sf)}  SF mean={np.mean(yes_sf):.4f}  median={np.median(yes_sf):.4f}  std={np.std(yes_sf):.4f}")
    print(f"  visible_defect=no:  n={len(no_sf)}  SF mean={np.mean(no_sf):.4f}  median={np.median(no_sf):.4f}  std={np.std(no_sf):.4f}")
    if len(yes_sf) >= 2 and len(no_sf) >= 2:
        # point-biserial correlation (defect=1/0 vs continuous SF) -- equivalent to Pearson r here
        labels = np.array([1 if r[2] == "yes" else 0 for r in rows])
        sfs = np.array([r[3] for r in rows])
        r = np.corrcoef(labels, sfs)[0, 1]
        print(f"  point-biserial r(defect, SF) = {r:.3f}  (negative = lower SF predicts defect, as SEGA's story predicts)")
        # simple best-threshold separability check (not a claimed classifier, just descriptive)
        thr_candidates = np.percentile(sfs, np.arange(5, 100, 5))
        best_acc, best_thr = 0, None
        for t in thr_candidates:
            pred = (sfs < t).astype(int)  # predict defect if SF below threshold
            acc = (pred == labels).mean()
            if acc > best_acc:
                best_acc, best_thr = acc, t
        print(f"  best-threshold separability (descriptive, not cross-validated): acc={best_acc:.2f} @ SF<{best_thr:.4f}")


def main():
    wb = subset_widerband()
    nd = subset_narrowdiag()
    report("widerband_decompose (72 imgs, joint-prompt judge)", wb)
    report("narrowband diagnostic (40 imgs, defect-only-prompt judge)", nd)
    out = {"widerband_decompose": [{"pid": p, "arm": a, "label": l, "sf": sf} for p, a, l, sf in wb],
           "narrowband_diagnostic": [{"pid": p, "arm": a, "label": l, "sf": sf} for p, a, l, sf in nd]}
    outp = os.path.join(DATA, "spectral_flatness_predictor_check.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
