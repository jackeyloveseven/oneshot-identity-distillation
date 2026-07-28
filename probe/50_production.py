#!/usr/bin/env python
"""
50_production.py — production entrypoint for the recipe this project
actually recommends: `strip`, band 21-25, `sched=taper`, `w=4.0`, plus the
WD14 tag-prior text channel (probe/31_tagprior.py). This is the ONLY recipe
this file exposes. `lowfreq`/band10-25 is NOT wired in here — the held-out
validation (docs/paper/draft_v0.md Section 4.3) showed it has an 82%
composition-defect rate (vs 20% for `strip`) and is not production-ready.

This module is a thin wrapper around probe/30_kv_inject.py's `generate()`
and probe/31_tagprior.py's tag-to-caption rendering. It intentionally does
NOT expose any of the experimental knobs those probe scripts carry (route,
vnorm, lowfreq_k, center_cut, cfg_id, sched_mode, band width) — those were
tuning surface for research, not production configuration, and the whole
point of this file is that a caller cannot accidentally select the
regressed recipe.

Measured production characteristics (docs/paper/draft_v0.md Section 6):
  - ~27-29s/image on an idle H200, single image at a time (no batching).
  - ~24-25GB GPU memory required (20.7GB to load the pipeline, more at
    generation time). `enable_model_cpu_offload()` is NOT wired in here
    since it is ~35-40x slower (verified) and not viable for anything but
    a single occasional generation on a memory-constrained shared GPU --
    a caller who needs that tradeoff should call `30_kv_inject.generate()`
    directly and apply the offload themselves.
  - Clean-usable output rate (identity=yes-or-partial AND no visible
    composition defect), measured on 30 held-out subjects never seen during
    recipe tuning, Gemini 3.1 Pro judge with chain-of-thought reasoning:
    53% [41%,65%] (Wilson 95% CI). This is not "solved" -- see
    docs/paper/draft_v0.md Section 5 for residual failure modes
    (geometric attributes, ~32% flat identity mismatch).

Run:
  source ../env.sh && $PY 50_production.py --ref /path/to/u12345_ref.png --out out.png
  $PY 50_production.py --selftest   # loads the pipeline and generates one
                                     # image against a REF-directory sample,
                                     # to verify the deployment is functional
"""
import os, sys, argparse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

RECIPE = dict(refpos="strip", band="21-25", sched_mode="taper", w=4.0,
              steps=50, cfg=5.0, refsize=512, size=1024)


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IdentityGenerator:
    """Loads the pipeline once; call .generate() per image.

    Kept as a class (rather than a bare function) because loading the
    pipeline costs ~20.7GB GPU memory and non-trivial wall-clock time
    (Section 6) -- a real deployment should construct one instance and
    reuse it across many generate() calls, not reload per request.
    """

    def __init__(self):
        self._kv = _load("kv_inject", "30_kv_inject.py")
        self._kv.INJ["band"] = self._kv.parse_band(RECIPE["band"])
        self._tp = _load("tagprior", "31_tagprior.py")

    def caption_for(self, ref_path):
        """WD14-tag-derived structural caption for the reference image
        (Section 3.5's text channel) -- generic T2I phrasing only, no edit
        verbs, per this project's hard constraint."""
        from imgutils.tagging import get_wd14_tags
        _, general, _ = get_wd14_tags(ref_path, model_name="SwinV2_v3",
                                       general_threshold=self._tp.WD14_THR)
        struct = {t: c for t, c in general.items() if t in self._tp.STRUCT}
        phrase = self._tp.render_phrase(struct)
        return self._tp.TEMPLATE.format(ident=phrase)

    def generate(self, ref_path, seed=0):
        """Returns a PIL.Image. `ref_path` must point at a single-subject
        (`solo`) reference portrait -- multi-character references are out of
        scope (project minefield #11; caller is responsible for filtering,
        e.g. via the is_solo() pattern in probe/36 / probe/41)."""
        uid = os.path.basename(ref_path)
        if uid.endswith("_ref.png"):
            uid = uid[: -len("_ref.png")]
        if uid.startswith("u"):
            uid = uid[1:]
        ref_dir = os.path.dirname(ref_path) or "."
        os.environ["REF"] = ref_dir
        self._kv.REF = ref_dir
        prompt = self.caption_for(ref_path)
        return self._kv.generate(
            uid, RECIPE["w"], refpos=RECIPE["refpos"], size=RECIPE["size"],
            refsize=RECIPE["refsize"], steps=RECIPE["steps"], cfg=RECIPE["cfg"],
            seed=seed, sched_mode=RECIPE["sched_mode"], prompt=prompt,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="path to a *_ref.png reference portrait")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                     help="load the pipeline and generate one image from the "
                          "first *_ref.png found under $REF, to verify the "
                          "deployment is functional end-to-end")
    a = ap.parse_args()

    gen = IdentityGenerator()

    if a.selftest:
        ref_dir = os.environ["REF"]
        cands = sorted(f for f in os.listdir(ref_dir) if f.endswith("_ref.png"))
        if not cands:
            print(f"[selftest] FAIL: no *_ref.png found under {ref_dir}", flush=True)
            sys.exit(1)
        ref_path = os.path.join(ref_dir, cands[0])
        print(f"[selftest] generating from {ref_path} ...", flush=True)
        im = gen.generate(ref_path, seed=a.seed)
        im.save(a.out)
        print(f"[selftest] OK, wrote {a.out}", flush=True)
        return

    if not a.ref:
        ap.error("--ref is required unless --selftest is given")
    im = gen.generate(a.ref, seed=a.seed)
    im.save(a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
