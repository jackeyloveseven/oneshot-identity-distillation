#!/usr/bin/env python
"""
54_its_sample.py -- ITS (Inference-Time Seed selection): a pre-registered-
playbook (5) seed early-screening sampler for a trained training-light LoRA
checkpoint.

Idea: sampling N seeds all the way to 50 steps to pick a good one is wasteful
-- most of what separates a good/bad seed (extra limbs, wrong composition,
identity drift) is already visible in a cheap K-step preview. This script
renders N=6 seeds (base seed --seed, actually --seed..--seed+N-1) at K=12
steps each, hard-filters out visibly broken previews (wd14 solo/multi-person
gate + a completeness/laplacian crash check), ranks the survivors by CCIP
identity similarity to the ref image, and only pays the full 50-step cost
ONCE, on the winning seed.

Reuse strategy (read-only, no bytes of probe/45_train_lora.py touched):
  The model (pure-T2I Z-Image transformer + VAE + text encoder, bf16) is
  loaded exactly once by importlib-loading probe/30_kv_inject.py (the same
  module probe/45_train_lora.py's own stage_sample() loads internally --
  see 45's `kv = _load("kv_inject_sample", "30_kv_inject.py")`). This file
  does NOT call 45.stage_sample() as a function: that function creates its
  OWN fresh `kv` (and therefore reloads the whole pipeline) on every call, so
  invoking it 6 times would reload ~7 times total -- exactly what this task's
  "model loaded once, 6 seeds share it" requirement rules out. 45's stage=
  sample CLI *does* already expose --steps externally (steps = args.steps or
  50), so a subprocess-per-seed with --steps 12 would technically work, but
  again at the cost of 6 reloads; that's why this file instead re-implements
  45's stage_sample() *inner* denoising loop (joint x=[[ref_lat,x_t]] /
  image_noise_mask=[[0,1]] / text-only CFG with the ref prefix kept in both
  branches / calculate_shift+retrieve_timesteps scheduler setup -- all copied
  verbatim from 45_train_lora.py:716-767, not reinvented) as `denoise()`
  below, parameterized by `steps`, so the SAME loaded tf/pipe/adapter and the
  SAME encoded prompt/ref-latent get reused across all N preview calls + the
  1 final call. This is the "45's components, own loop" fallback the task
  brief explicitly allows.

  This file does NOT importlib-load probe/45_train_lora.py at all -- only
  30_kv_inject.py (the exact module 45's own stage_sample() loads
  internally). There is nothing in 45 left to call once 30 is loaded
  directly: parse_band()/build_lora_config() are only reached on 45's fresh
  zero-init B=0 branch (irrelevant here, --adapter is required), and
  stage_sample() itself can't be called as a function without it creating
  its own fresh `kv` (a reload) every time. So "45's loaded components" is
  read as "the components 45 itself loads from 30_kv_inject.py", and the
  denoising loop below is copied verbatim from 45_train_lora.py:716-767 (not
  reinvented) rather than invoked through 45. The trained-adapter load
  matches 45's stage_sample --adapter branch exactly: `tf.load_lora_adapter(
  adapter, prefix=None, weight_name="pytorch_lora_weights.safetensors")`, NO
  cast_lora_fp32 call (that's only done on 45's fresh zero-init B=0 branch,
  never on a loaded checkpoint -- loaded LoRA weights already carry whatever
  dtype they were saved in and 45 leaves them alone).

Hard filter (elimination, applied to the K-step preview only):
  1. wd14 (imgutils, SwinV2_v3, general_threshold=0.3) solo<0.5 OR
     multi_max>=0.35 -- same model/thresholds/multi_max formula as
     probe/48_clean_manifest.py's drop_not_solo gate (SOLO_THR=0.5,
     MULTI_THR=0.35, multi_max = max(2girls,2boys,multiple_girls,
     multiple_boys, min(1girl,1boy))).
  2. completeness/laplacian crash check. NOTE (deviation from the task
     brief, see report): the brief says to reuse probe/18_metrics.py's
     completeness/laplacian functions. That file does not exist anywhere in
     this repo -- it lived at /mnt/local/sparkid/probe/18_metrics.py, a
     workspace wiped 2026-07-01 (see memory note reference_anime_native_
     metrics), and was never migrated into cherry/probe/ (cherry's own
     probe/32_metrics.py only carries the CCIP half forward). Reimplemented
     here directly against the same imgutils APIs the memory note documents
     (imgutils.validate.anime_completeness_score, imgutils.metrics.
     laplacian_score), adapted to be self-contained: the original
     breakdown_score() compared laplacian to a same-subject UNGUIDED
     baseline generation, which this ITS pipeline has no equivalent of (it
     only ever generates the actual ref-conditioned preview/final image), so
     this file uses absolute floors instead of a vs-base ratio --
     COMPLETENESS_POLISHED_FLOOR / LAPLACIAN_FLOOR below, both module-level
     constants for easy re-tuning, calibrated against this task's own GPU
     smoke test (see report for observed values).
  If ALL N seeds are eliminated by (1) or (2), fall back: ignore the hard
  filter entirely and rank all N seeds by CCIP alone.

Ranking (winner selection): CCIP identity difference (imgutils, same
ccip_extract_feature/ccip_difference usage as probe/32_metrics.py) between
each surviving preview and the ref image u{uid}_ref.png; LOWER diff = more
similar, so the winner is min(ccip_diff) among survivors (or among all N on
the all-eliminated fallback path).

Defect-aware quality gate (--select quality_ccip, new; --select ccip is the
default and reproduces the paragraph above byte-for-byte -- this is opt-in):
  Diagnosis on 4 known-defect rep30 subjects (u2407 malformed eye, u4224
  realism/style drift, u4573 face occluded by a held phone, u4096 extra-limb/
  garbled-hand anatomy) found the hard filter above has a blind spot: a
  preview can be "complete" (passes wd14 solo + completeness/laplacian) and
  still be wrong -- rendered in the wrong style, or with the identity-bearing
  face occluded/malformed -- and CCIP alone doesn't reliably catch this
  either (a phone-occluded face can still CCIP-match by hair/outfit/palette).
  quality_signals() below adds 3 more imgutils reads per preview, all
  independent of the hard filter and of each other: anime_real_score(img)
  ['real'] (photoreal-style-drift probability), len(detect_heads(img))
  (0 = no clear head = occluded/collapsed), len(detect_text(img)) (>0 =
  garbled text artifact). A 4th (detect_eyes count) is recorded but NOT
  gated on -- see QUALITY_USE_EYES below for why. --select quality_ccip
  narrows survivors to quality_clean = {real < REAL_THR AND n_heads >= 1 AND
  n_text == 0} before taking min(ccip_diff); if quality_clean is empty it
  falls back to ccip-only over all survivors (quality_fallback_used=True in
  the scores json) -- same "don't let an over-eager filter zero out the
  candidate pool" shape as the existing hard-filter fallback, just one layer
  in. See probe/54's task report for the offline 30-subject A/B and the
  per-subject diagnostic (which defects this does and doesn't rescue, and
  why -- it is NOT a fix for every failure mode in the 4 diagnosed subjects).

Run (GPU step -- checks nvidia-smi free>=30G first, polls every 120s if not):
  source ../env.sh
  export REF=/mnt/local/smoke_anime/anime_260623/img   # env.sh's own REF
                                                         # points at a
                                                         # destroyed dir; this
                                                         # script also
                                                         # defaults REF itself
                                                         # (see REF_IMG_DIR
                                                         # below) as a safety
                                                         # net, same pattern
                                                         # probe/46_devprobe.py
                                                         # uses.
  $PY 54_its_sample.py --ref 4416 \
      --adapter out/trainlight/run_mix1/step_4000 \
      --prompt "In an anime_2d style, solo, ..." \
      --cap_ref "an anime illustration of a character" \
      --select quality_ccip \
      --out out/devprobe/its_smoke
  # --select defaults to "ccip" (legacy, backward-compatible winner selection);
  # pass --select quality_ccip to enable the defect-aware quality gate.
"""
import os, sys, json, time, argparse, importlib.util, subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# REF: env.sh's own REF has pointed at a destroyed dir since the 2026-07-06
# data incident (see probe/41_holdout_eval.py's docstring for the same
# issue) -- default to the known-good snapshot the task brief specifies,
# same defensive os.environ.get(...) pattern probe/46_devprobe.py uses, and
# normalize os.environ["REF"] BEFORE importlib-loading 30_kv_inject.py so
# that module's own `REF = os.environ["REF"]` (read at import time) resolves
# to the same directory this script uses for u{uid}_ref.png.
REF_IMG_DIR = os.environ.get("REF", "/mnt/local/smoke_anime/anime_260623/img")
os.environ["REF"] = REF_IMG_DIR
print(f"[54_its_sample] using REF={REF_IMG_DIR}", flush=True)

# --- offline HF cache merge for imgutils quality/defect signals -----------
# env.sh's HF_HOME=/mnt/local/hf_home only holds the repos the ORIGINAL hard
# filter needs (anime_completeness, ccip_onnx, wd14_tagger_with_embeddings +
# SmilingWolf/wd-swinv2-tagger-v3, imgutils-models). The NEW defect-aware
# quality signals added below (anime_real_score, detect_heads, detect_text,
# detect_eyes) need repos (deepghs/anime_real_cls, anime_head_detection,
# anime_face_detection, text_detection, anime_eye_detection) that were found
# scattered across THREE OTHER pre-existing local HF caches on this shared
# machine -- no single existing HF_HOME has all of them. Rather than touch
# env.sh (out of scope: only this file may change for this task) or silently
# fall through to a network download on a shared H200 (breaks the task's
# HF_HUB_OFFLINE=1 requirement), this builds ONE merged local cache dir
# (idempotent symlink farm, created once then reused) with every repo BOTH
# the original hard filter and the new quality signals need, and points
# HF_HOME/HF_HUB_CACHE at it for the rest of this process.
#   Safe to override: 30_kv_inject.py's model loads (ZBASE) are 100%
# local-path (MODEL = os.environ["ZBASE"]; ZImagePipeline.from_pretrained on
# that local directory) and never consult HF_HOME/HF_HUB_CACHE, so
# redirecting these two env vars here cannot affect model loading -- only
# these imgutils calls, which is exactly the surface this task's quality
# gate touches.
_HF_CACHE_SOURCES = [
    os.environ.get("HF_HOME", "/mnt/local/hf_home"),   # env.sh's HF_HOME: ccip/wd14/completeness
    os.path.expanduser("~/.cache/huggingface"),          # default cache: real_cls/head/text/eyes
    "/mnt/local/hf",                                      # alt cache: head/hand/person detection
    "/mnt/local/huggingface_cache",                       # alt cache: face detection
]
_HF_CACHE_REQUIRED_REPOS = [
    "models--deepghs--anime_completeness", "models--deepghs--ccip_onnx",
    "models--deepghs--wd14_tagger_with_embeddings", "models--SmilingWolf--wd-swinv2-tagger-v3",
    "models--deepghs--imgutils-models",
    "models--deepghs--anime_real_cls", "models--deepghs--anime_head_detection",
    "models--deepghs--anime_face_detection", "models--deepghs--text_detection",
    "models--deepghs--anime_eye_detection",
]
_HF_CACHE_MERGED = "/mnt/local/cherry_out/hf_cache_merged"


def _setup_merged_hf_cache():
    hub = os.path.join(_HF_CACHE_MERGED, "hub")
    os.makedirs(hub, exist_ok=True)
    missing = []
    for repo in _HF_CACHE_REQUIRED_REPOS:
        target = os.path.join(hub, repo)
        if os.path.lexists(target):
            continue
        found = None
        for root in _HF_CACHE_SOURCES:
            cand = os.path.join(root, "hub", repo)
            if os.path.isdir(cand):
                found = cand
                break
        if found is None:
            missing.append(repo)
            continue
        os.symlink(found, target)
    if missing:
        raise RuntimeError(
            f"[54_its_sample] offline HF cache setup FAILED, repo(s) not found in any of "
            f"{_HF_CACHE_SOURCES}: {missing}. Populate one of these caches (run the failing "
            f"imgutils call once with HF_HUB_OFFLINE unset) or add its source root to "
            f"_HF_CACHE_SOURCES above.")
    os.environ["HF_HOME"] = _HF_CACHE_MERGED
    os.environ["HF_HUB_CACHE"] = hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(f"[54_its_sample] offline HF cache ready at {_HF_CACHE_MERGED} "
          f"({len(_HF_CACHE_REQUIRED_REPOS)} repos merged from {len(_HF_CACHE_SOURCES)} source caches)",
          flush=True)


_setup_merged_hf_cache()

DEV = "cuda"
N_DEFAULT = 6
K_DEFAULT = 12
FINAL_STEPS = 50          # production RECIPE steps (probe/45_train_lora.py stage_sample default)
CFG = 5.0                 # production RECIPE cfg
SIZE = 1024                # production RECIPE size
REFSIZE = 512              # production RECIPE refsize

MIN_FREE_GB = 30
GPU_POLL_S = 120

# --- wd14 solo/multi hard filter: verbatim thresholds/formula from
# probe/48_clean_manifest.py's drop_not_solo gate --------------------------
WD14_MODEL = "SwinV2_v3"
WD14_GENERAL_THRESHOLD = 0.3
SOLO_THR = 0.5
MULTI_THR = 0.35
WD14_TAGS_OF_INTEREST = ["solo", "1girl", "1boy", "2girls", "2boys",
                          "multiple_girls", "multiple_boys"]

# --- completeness/laplacian crash check (see module docstring for why this
# is a reimplementation, not a probe/18_metrics.py import) -----------------
COMPLETENESS_POLISHED_FLOOR = 0.15   # anime_completeness_score(img)['polished']
                                       # below this = classifier confidently
                                       # says NOT polished (rough/monochrome
                                       # collapse), eliminate. Deliberately
                                       # lenient: a 12-step preview is
                                       # legitimately less refined than a
                                       # 50-step final, this floor is only
                                       # meant to catch confident collapse,
                                       # not "could be sharper".
LAPLACIAN_FLOOR = 5.0                 # imgutils.metrics.laplacian_score below
                                       # this = near-flat/blank image (docstring
                                       # examples: sharp anime art scores
                                       # 150-1200+, even a blurry real photo
                                       # scores ~16 -- single digits means
                                       # "nothing rendered", not "soft").

CCIP_SAME_THR = 0.178  # imgutils anime-ArcFace SAME/DIFF threshold (probe/32_metrics.py)

# --- defect-aware quality gate (--select quality_ccip only; see module ----
# docstring "Defect-aware quality gate" section for the diagnosis this is
# built against) --------------------------------------------------------
REAL_THR = 0.5   # anime_real_score(img)['real'] (imgutils, deepghs/anime_real_cls) >= this
                   # = flag as photoreal/style-drift, eliminate from the quality-clean subset.
                   # 'anime'+'real' sum to 1.0, so 0.5 is the classifier's own decision
                   # boundary -- starting value per task brief, module constant for easy
                   # re-tuning. Deliberately NOT tuned against the rep30 eval set this task
                   # A/Bs on (would contaminate the A/B); see task report for why 0.5 already
                   # is a real (not just nominal) split on the diagnosed subjects -- and for
                   # its one known false-negative (u4224: a subject whose reference/prompt push
                   # it toward photoreal on ALL 6 seeds, so no seed selection threshold can
                   # rescue it -- the classifier's 'real' reading on that subject's cleanest-
                   # looking-by-CCIP seed undersells how photoreal it visually is).
QUALITY_USE_EYES = False   # optional detect_eyes(img) (imgutils, deepghs/anime_eye_detection)
                   # count gate -- OFF by default. Smoke evidence against the diagnosed u2407
                   # malformed-eye subject: its actual (bad) ITS winner and a visibly-cleaner
                   # sibling seed both score n_eyes=1 (hair-over-one-eye is this character's
                   # correct design, not a defect by itself) -- the malformation is in the
                   # rendered EYE TEXTURE, which a detection COUNT cannot see. Per the task
                   # brief's "宁缺毋滥" instruction: n_eyes is still computed and recorded in
                   # every per_seed entry for future use, just not gated on here.


def quality_signals(img_path):
    """Defect-aware quality signals for one preview seed -- computed and stored for EVERY
    preview regardless of --select (so a --select ccip run still leaves this data available
    for later analysis without a re-run). Independent of the existing hard filter/CCIP:
      real    -- anime_real_score(img)['real'], photoreal-style-drift probability.
      n_heads -- len(detect_heads(img)); 0 = no clear head detected (occluded/collapsed).
      n_text  -- len(detect_text(img)); >0 = garbled text artifact rendered into the image.
      n_eyes  -- len(detect_eyes(img)); recorded only, see QUALITY_USE_EYES above.
      quality_ok -- real < REAL_THR and n_heads >= 1 and n_text == 0 (and n_eyes >= 1 iff
                    QUALITY_USE_EYES). This is the per-seed predicate --select quality_ccip
                    narrows the survivor pool by before taking min(ccip_diff)."""
    import warnings
    from imgutils.validate import anime_real_score
    from imgutils.detect import detect_heads, detect_text, detect_eyes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*detect_text is deprecated.*")
        n_text = len(detect_text(img_path))
    real = float(anime_real_score(img_path).get("real", 0.0))
    n_heads = len(detect_heads(img_path))
    n_eyes = len(detect_eyes(img_path))
    quality_ok = (real < REAL_THR) and (n_heads >= 1) and (n_text == 0)
    if QUALITY_USE_EYES:
        quality_ok = quality_ok and (n_eyes >= 1)
    return {"real": round(real, 4), "n_heads": n_heads, "n_text": n_text, "n_eyes": n_eyes,
            "quality_ok": quality_ok}


def _load(modname, filename):
    """Import a probe/ script as a module by file path, read-only (identical
    pattern probe/45_train_lora.py / probe/46_devprobe.py / probe/50_
    production.py all already use)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu_wait(min_free_gb=MIN_FREE_GB, poll_s=GPU_POLL_S):
    """Project GPU etiquette: check nvidia-smi free memory before any GPU
    step, proceed once >=min_free_gb is free, else poll every poll_s
    seconds. Shared H200 -- another job's ~42G training footprint is
    expected and NOT waited out, only genuine insufficient headroom is."""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mb = min(int(x) for x in out.stdout.strip().splitlines())
        free_gb = free_mb / 1024.0
        if free_gb >= min_free_gb:
            print(f"[54_its_sample] GPU free={free_gb:.1f}G >= {min_free_gb}G, proceeding", flush=True)
            return
        print(f"[54_its_sample] GPU free={free_gb:.1f}G < {min_free_gb}G, "
              f"waiting {poll_s}s before re-checking...", flush=True)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# hard filter / ranking metrics
# ---------------------------------------------------------------------------

def wd14_filter(img_path):
    from imgutils.tagging import get_wd14_tags
    _rating, general, _chars = get_wd14_tags(
        img_path, model_name=WD14_MODEL, general_threshold=WD14_GENERAL_THRESHOLD,
    )
    tags = {t: round(float(general.get(t, 0.0)), 4) for t in WD14_TAGS_OF_INTEREST}
    solo = tags["solo"]
    multi_max = max(tags["2girls"], tags["2boys"], tags["multiple_girls"], tags["multiple_boys"],
                     min(tags["1girl"], tags["1boy"]))
    ok = solo >= SOLO_THR and multi_max < MULTI_THR
    return {"tags": tags, "solo": solo, "multi_max": round(multi_max, 4), "pass": ok}


def breakdown_filter(img_path):
    """completeness + laplacian crash detection -- see module docstring for
    the probe/18_metrics.py-doesn't-exist deviation note."""
    from imgutils.validate import anime_completeness_score
    from imgutils.metrics import laplacian_score
    comp = {k: round(float(v), 4) for k, v in anime_completeness_score(img_path).items()}
    lap = round(float(laplacian_score(img_path)), 2)
    completeness_pass = comp.get("polished", 0.0) >= COMPLETENESS_POLISHED_FLOOR
    laplacian_pass = lap >= LAPLACIAN_FLOOR
    return {"completeness": comp, "completeness_pass": completeness_pass,
            "laplacian": lap, "laplacian_pass": laplacian_pass,
            "pass": completeness_pass and laplacian_pass}


def ccip_diff(a_path, b_path):
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    return float(ccip_difference(ccip_extract_feature(a_path), ccip_extract_feature(b_path)))


# ---------------------------------------------------------------------------
# sampling (45_train_lora.py stage_sample()'s inner loop, parameterized by
# `steps` so it can be called K-step (preview) or 50-step (final) against the
# SAME already-loaded pipeline/adapter/encodes -- see module docstring)
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise(kv, tf, DT, pipe, ref_lat, cap_ref_enc, cap_pos_enc, cap_neg_enc,
            seed, steps, cfg=CFG, size=SIZE):
    sched = pipe.scheduler
    gen = torch.Generator(DEV).manual_seed(seed)
    hlat = 2 * (size // 16)
    latents = torch.randn((1, 16, hlat, hlat), generator=gen, device=DEV, dtype=torch.float32)

    isl = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = kv.calculate_shift(isl,
                             sched.config.get("base_image_seq_len", 256),
                             sched.config.get("max_image_seq_len", 4096),
                             sched.config.get("base_shift", 0.5),
                             sched.config.get("max_shift", 1.15))
    sched.sigma_min = 0.0
    try:
        timesteps, _ = kv.retrieve_timesteps(sched, steps, DEV, sigmas=None, mu=mu)
    except Exception:
        timesteps, _ = kv.retrieve_timesteps(sched, steps, DEV, sigmas=None)
    n_timesteps = len(timesteps)

    torch.cuda.synchronize()
    t_loop0 = time.time()
    for t in timesteps:
        ts = ((1000 - t.expand(latents.shape[0])) / 1000)
        x_t = latents.to(DT).unsqueeze(2)[0]        # (16,1,H,W)

        def joint_fwd(cap_txt_enc):
            # [0][0] = target-slot velocity (x0-noise direction), ref/cap
            # tokens already excluded by omni unpatchify -- same call shape
            # as 45_train_lora.py's forward_backward()/stage_sample().
            return tf(x=[[ref_lat, x_t]], t=ts, cap_feats=[[cap_ref_enc, cap_txt_enc]],
                      siglip_feats=[None], image_noise_mask=[[0, 1]],
                      return_dict=False)[0][0].float()

        cond = joint_fwd(cap_pos_enc)
        unc = joint_fwd(cap_neg_enc)
        pred = cond + cfg * (cond - unc)
        noise_pred = (-pred).unsqueeze(0).squeeze(2)  # (1,16,H,W)
        latents = sched.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
    torch.cuda.synchronize()
    loop_s = time.time() - t_loop0

    n_nan = int(torch.isnan(latents).sum().item())
    t_dec0 = time.time()
    lat = latents.to(pipe.vae.dtype)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img_t = pipe.vae.decode(lat, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img_t, output_type="pil")[0]
    torch.cuda.synchronize()
    decode_s = time.time() - t_dec0

    return img, {"n_timesteps": n_timesteps, "n_nan": n_nan, "loop_s": loop_s, "decode_s": decode_s}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="uid, resolved to $REF/u{uid}_ref.png")
    ap.add_argument("--prompt", required=True, help="cond text for the target slot")
    ap.add_argument("--adapter", required=True,
                     help="LoRA checkpoint dir (probe/45_train_lora.py stage=train output)")
    ap.add_argument("--cap_ref", required=True, help="ref-slot caption text")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="number of seeds to screen")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="preview denoise steps")
    ap.add_argument("--seed", type=int, default=0, help="base seed; actual seeds = seed..seed+n-1")
    ap.add_argument("--select", choices=["ccip", "quality_ccip"], default="ccip",
                     help="winner selection within hard-filter survivors: 'ccip' (default, "
                          "backward-compatible) = min(ccip_diff) over all survivors, unchanged "
                          "from pre-quality-gate behavior; 'quality_ccip' = min(ccip_diff) over "
                          "the quality-clean subset of survivors (real<REAL_THR & n_heads>=1 & "
                          "n_text==0), falling back to ccip-only over all survivors if that "
                          "subset is empty (quality_fallback_used=True in the scores json)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    uid = args.ref.lstrip("u")
    ref_path = os.path.join(REF_IMG_DIR, f"u{uid}_ref.png")
    if not os.path.exists(ref_path):
        print(f"[54_its_sample] FAIL: ref image not found at {ref_path}", flush=True)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    seeds = [args.seed + i for i in range(args.n)]
    t0_wall = time.time()

    gpu_wait()

    t_load0 = time.time()
    kv = _load("kv_inject_its", "30_kv_inject.py")   # loads pipeline ONCE (bf16), shared by all seeds
    tf, DT, pipe = kv.tf, kv.DT, kv.pipe

    # trained-adapter path, byte-identical to 45_train_lora.py stage_sample()'s
    # --adapter branch (asymmetric-default gotcha: weight_name must be given
    # explicitly). NO cast_lora_fp32 here -- that's only ever applied on 45's
    # fresh zero-init B=0 branch, never to a loaded checkpoint.
    tf.load_lora_adapter(args.adapter, prefix=None, weight_name="pytorch_lora_weights.safetensors")
    tf.eval()
    load_s = time.time() - t_load0
    print(f"[54_its_sample] pipeline+adapter loaded in {load_s:.1f}s (adapter={args.adapter})", flush=True)

    cap_ref_enc = kv.enc(args.cap_ref)
    cap_pos_enc = kv.enc(args.prompt)
    cap_neg_enc = kv.enc(kv.NEG)
    ref_lat = kv.ref_latent(ref_path, REFSIZE)
    print(f"[54_its_sample] u{uid} seeds={seeds} n={args.n} k={args.k} final_steps={FINAL_STEPS} "
          f"cap_ref={args.cap_ref[:60]!r} prompt={args.prompt[:80]!r}", flush=True)

    per_seed = {}
    for seed in seeds:
        t_seed0 = time.time()
        img, diag = denoise(kv, tf, DT, pipe, ref_lat, cap_ref_enc, cap_pos_enc, cap_neg_enc,
                             seed, args.k)
        preview_path = os.path.join(args.out, f"u{uid}_its_preview_s{seed}.png")
        img.save(preview_path)
        seed_wall = time.time() - t_seed0

        wd14 = wd14_filter(preview_path)
        brk = breakdown_filter(preview_path)
        diff = ccip_diff(ref_path, preview_path)
        qual = quality_signals(preview_path)   # computed always, gated only under --select quality_ccip

        reasons = []
        if not wd14["pass"]:
            reasons.append(f"wd14: solo={wd14['solo']:.3f}(<{SOLO_THR}) or "
                            f"multi_max={wd14['multi_max']:.3f}(>={MULTI_THR})")
        if not brk["completeness_pass"]:
            reasons.append(f"completeness: polished={brk['completeness'].get('polished', 0):.3f}"
                            f"(<{COMPLETENESS_POLISHED_FLOOR})")
        if not brk["laplacian_pass"]:
            reasons.append(f"laplacian: {brk['laplacian']:.1f}(<{LAPLACIAN_FLOOR})")
        hard_pass = wd14["pass"] and brk["pass"]

        per_seed[str(seed)] = {
            "seed": seed, "preview_path": preview_path,
            "n_timesteps": diag["n_timesteps"], "n_nan": diag["n_nan"],
            "loop_s": round(diag["loop_s"], 3), "decode_s": round(diag["decode_s"], 3),
            "wall_s": round(seed_wall, 3),
            "wd14": wd14,
            "completeness": brk["completeness"], "completeness_pass": brk["completeness_pass"],
            "laplacian": brk["laplacian"], "laplacian_pass": brk["laplacian_pass"],
            "hard_filter_pass": hard_pass, "eliminate_reasons": reasons,
            "ccip_diff": round(diff, 4), "ccip_same": diff < CCIP_SAME_THR,
            "quality": qual,
        }
        print(f"[54_its_sample] seed={seed} n_timesteps={diag['n_timesteps']} n_nan={diag['n_nan']} "
              f"wd14_pass={wd14['pass']}(solo={wd14['solo']:.3f} multi_max={wd14['multi_max']:.3f}) "
              f"completeness_pass={brk['completeness_pass']}(polished={brk['completeness'].get('polished', 0):.3f}) "
              f"laplacian_pass={brk['laplacian_pass']}({brk['laplacian']:.1f}) "
              f"ccip_diff={diff:.4f}{'(SAME)' if diff < CCIP_SAME_THR else ''} "
              f"quality_ok={qual['quality_ok']}(real={qual['real']:.3f} n_heads={qual['n_heads']} "
              f"n_text={qual['n_text']} n_eyes={qual['n_eyes']}) "
              f"loop_s={diag['loop_s']:.2f} wall_s={seed_wall:.2f}", flush=True)

    survivors = [s for s in seeds if per_seed[str(s)]["hard_filter_pass"]]
    fallback_used = len(survivors) == 0
    pool = survivors if survivors else seeds

    # quality_clean is always computed (informational, from data already gathered above) so a
    # --select ccip run still records it; it only DRIVES selection under --select quality_ccip.
    quality_clean = [s for s in pool if per_seed[str(s)]["quality"]["quality_ok"]]
    if args.select == "quality_ccip":
        quality_fallback_used = len(quality_clean) == 0
        select_pool = quality_clean if quality_clean else pool
    else:
        quality_fallback_used = False
        select_pool = pool
    winner = min(select_pool, key=lambda s: per_seed[str(s)]["ccip_diff"])
    print(f"[54_its_sample] survivors={survivors} fallback_used={fallback_used} "
          f"select={args.select} quality_clean={quality_clean} quality_fallback_used={quality_fallback_used} "
          f"winner=seed{winner} (ccip_diff={per_seed[str(winner)]['ccip_diff']:.4f})", flush=True)

    t_final0 = time.time()
    final_img, fdiag = denoise(kv, tf, DT, pipe, ref_lat, cap_ref_enc, cap_pos_enc, cap_neg_enc,
                                winner, FINAL_STEPS)
    final_path = os.path.join(args.out, f"u{uid}_its_s{winner}.png")
    final_img.save(final_path)
    final_wall = time.time() - t_final0
    print(f"[54_its_sample] final image (seed={winner}, steps={FINAL_STEPS}, "
          f"n_timesteps={fdiag['n_timesteps']}, n_nan={fdiag['n_nan']}) saved to {final_path} "
          f"in {final_wall:.2f}s (loop_s={fdiag['loop_s']:.2f} decode_s={fdiag['decode_s']:.2f})", flush=True)

    total_wall = time.time() - t0_wall
    preview_loop_mean = sum(per_seed[str(s)]["loop_s"] for s in seeds) / len(seeds)
    speedup = fdiag["loop_s"] / preview_loop_mean if preview_loop_mean > 0 else float("nan")

    result = {
        "uid": uid, "ref": ref_path, "prompt": args.prompt, "adapter": args.adapter,
        "cap_ref": args.cap_ref, "n": args.n, "k": args.k, "base_seed": args.seed,
        "seeds": seeds, "final_steps": FINAL_STEPS, "cfg": CFG, "size": SIZE, "refsize": REFSIZE,
        "thresholds": {"solo_thr": SOLO_THR, "multi_thr": MULTI_THR,
                        "completeness_polished_floor": COMPLETENESS_POLISHED_FLOOR,
                        "laplacian_floor": LAPLACIAN_FLOOR, "ccip_same_thr": CCIP_SAME_THR,
                        "real_thr": REAL_THR, "quality_use_eyes": QUALITY_USE_EYES},
        "per_seed": per_seed,
        "survivors": survivors, "fallback_used": fallback_used,
        "select": args.select, "quality_clean": quality_clean, "quality_fallback_used": quality_fallback_used,
        "winner_seed": winner, "winner_ccip_diff": per_seed[str(winner)]["ccip_diff"],
        "final_image": final_path,
        "timing": {
            "load_s": round(load_s, 2),
            "preview_loop_mean_s": round(preview_loop_mean, 3),
            "final_loop_s": round(fdiag["loop_s"], 3),
            "final_wall_s": round(final_wall, 3),
            "preview_vs_final_loop_speedup": round(speedup, 2),
            "expected_speedup_from_steps_alone": round(FINAL_STEPS / args.k, 2),
            "total_wall_s": round(total_wall, 2),
        },
    }
    scores_path = os.path.join(args.out, f"u{uid}_its_scores.json")
    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[54_its_sample] wrote {scores_path}", flush=True)
    print(f"[54_its_sample] DONE: winner=seed{winner} select={args.select} fallback_used={fallback_used} "
          f"quality_fallback_used={quality_fallback_used} "
          f"total_wall_s={total_wall:.1f} preview/final loop speedup={speedup:.2f}x "
          f"(steps alone predicts ~{FINAL_STEPS / args.k:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
