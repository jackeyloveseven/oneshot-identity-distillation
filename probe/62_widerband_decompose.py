#!/usr/bin/env python
"""
62_widerband_decompose.py -- decisive single-variable dev-set decomposition
for the lowfreq recipe's 82% held-out defect collapse (draft_v0.md
Limitations: "why widening the band and frequency-gating together
destabilize composition is unexplained").

`lowfreq` (band 10-25, refpos=lowfreq k=8) changed TWO variables vs `strip`
(band 21-25, refpos=strip, no RoPE reapplied) at once: band width (5 layers
-> 16 layers) AND refpos (none -> low-freq-gated re-centered). A same-day
diagnostic (probe/61_defect_only_diagnostic.py, re-judging pre-existing
gen10 images) found a within-instrument contrast at the ORIGINAL narrow
band (21-25): lowfreq stays clean (5%, n=20) while `center` (full-precision
repositioned RoPE, no freq-gating) is already catastrophic (70%, n=20) --
i.e. AT NARROW BAND, the frequency-gating mechanism itself is not the
problem; full-precision repositioning is.

That result does NOT yet tell us why WIDENING lowfreq to band 10-25
collapses to 82%. Two live hypotheses:
  (a) low-freq position info is specifically dangerous once it reaches
      early/mid layout-forming layers (10-20) -- the frequency story
      matters, band depth is what activates it.
  (b) ANY reference-content injection into layers 10-20 is harmful,
      independent of position encoding -- E4 (README.md L132, 2 subjects,
      qualitative, pre-defect-judge) already flagged band 16-20 as
      "harmful (wrong appearance injected)" and 10-29 as "over-injected"
      under STRIP (position-free, full content) -- suggestive prior
      evidence AGAINST hypothesis (a) that was not connected to this
      question before now.

This script runs the ONE missing, decisive cell -- `strip` (position-free)
at the WIDE band (10-25), matching lowfreq's exact band width -- alongside
fresh regenerations of both reference arms, ALL THREE on the SAME dev
subjects/seeds/prompts, judged with the SAME instrument (HOLDOUT_PROMPT,
joint ref-comparison identity+defect), so the three-way comparison is
clean and single-instrument (not the cross-instrument, cross-N comparison
probe/61 had to make do with).

Three interpretable outcomes for strip_wide (band 10-25, no position):
  - clean defect rate + recovers identity gain (~40%)  -> hypothesis (a)
    confirmed, "widen strip" is a viable fix candidate, validate once on
    holdout30 as a third arm.
  - clean defect rate but identity stays ~strip's level -> the identity
    gain required position info; strip_wide is not a fix (nothing to
    validate), but this still isolates that the DEFECT is position-driven,
    a real mechanism finding.
  - also high defect rate -> hypothesis (b): content-into-layout-layers is
    itself the problem, independent of position. Clean negative for this
    entire fix direction; report honestly, no further GPU spend justified
    on this specific line.

Dev set: devprobe12_tags.json (12 subjects, working refs at
/mnt/local/smoke_anime/anime_260623/img -- gen10's original ref pool no
longer exists on disk, see probe/61's docstring). DIAGNOSTIC ONLY -- these
are dev-set numbers, not an acceptance result. Whatever this suggests gets
exactly ONE validation pass on holdout30 (a fresh third arm alongside the
existing old/new arms), never re-tuned against holdout30 itself.

Run: source ../env.sh && REF=/mnt/local/smoke_anime/anime_260623/img $PY 62_widerband_decompose.py --stage gen
     $PY 38_gemini_judge.py --set widerband_decompose   (separate call, also needs REF override)
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.environ.get("WIDERBAND_OUT", "/mnt/local/cherry_out/widerband_decompose")
TAGS_JSON = os.path.join(DATA, "devprobe12_tags.json")
SEEDS = [0, 1]

RECIPE = {
    "strip_narrow": dict(band="21-25", refpos="strip", lowfreq_k=4),   # = production 'old'
    "lowfreq_wide":  dict(band="10-25", refpos="lowfreq", lowfreq_k=8),  # = frozen 'new' (known 82% defect on holdout30)
    "strip_wide":    dict(band="10-25", refpos="strip", lowfreq_k=4),   # <-- the decisive new cell
}


def img_path(uid, arm, seed):
    return f"{OUT}/u{uid}_wbd_{arm}_s{seed}.png"


def stage_gen():
    tags = json.load(open(TAGS_JSON))
    uids = [u for u in tags if not u.startswith("_")]
    todo = [(u, a, s) for u in uids for a in RECIPE for s in SEEDS
            if not os.path.exists(img_path(u, a, s))]
    print(f"[gen] {len(todo)} images remaining of {len(uids)*len(RECIPE)*len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    for uid, arm, seed in todo:
        r = RECIPE[arm]
        kv.INJ["band"] = kv.parse_band(r["band"])
        prompt = tags[uid]["prompt_B"]
        print(f"[gen] u{uid} arm={arm} seed={seed}", flush=True)
        im = kv.generate(uid, 4.0, refpos=r["refpos"], lowfreq_k=r["lowfreq_k"],
                         sched_mode="taper", seed=seed, prompt=prompt)
        im.save(img_path(uid, arm, seed))
    print("[gen] batch done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen"])
    a = ap.parse_args()
    stage_gen()
