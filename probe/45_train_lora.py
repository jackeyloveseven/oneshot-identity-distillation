#!/usr/bin/env python
"""
45_train_lora.py — training-light adapter: peft LoRA on the frozen pure-T2I
Z-Image transformer, four stages (regression / profile / train / sample).

Design: docs/design/2026-07-11_traininglight_design.md (S2.7, S3 "3b" + the
recommendation section, S4, S6, S7 M0). This file does NOT re-derive the
design -- it implements the M0 sanity checks the design left unverified and
lays down the (not-yet-run) full training loop the design specifies.

CONSTRAINTS honored (project red lines, repeated here on purpose):
  1. Base model stays pure-T2I Z-Image, weights frozen the whole time. No
     Omni/siglip checkpoint is ever loaded (env.sh already deleted it).
  2. No per-subject training anywhere in this file -- LoRA is one adapter
     shared across all subjects.
  3. Only public diffusers/peft interfaces are used
     (`ZImageTransformer2DModel.add_adapter` / `.save_lora_adapter` /
     `.load_lora_adapter`, all from diffusers' `PeftAdapterMixin`, backed by
     `peft.inject_adapter_in_model`). No diffusers/peft source is edited.
  4. Training captions: stage=train reads `caption_scrubbed` from a manifest
     produced by a separate (parallel) task -- this file does not clean
     captions itself. stage=profile uses the raw `t2i_instruction` field
     from anime_30k_recap/metadata.jsonl, which the design doc (S2.2) already
     screened for zero identity-leak phrase hits; profile-stage caption
     *quality* is explicitly out of scope per the task brief.

Reuses (never edits) probe/30_kv_inject.py (pipeline loading, InjectProcessor,
ref_latent()/enc() helpers, RoPE/omni conventions) and probe/50_production.py
(IdentityGenerator / the frozen production recipe) via the same
importlib-spec `_load()` pattern 50_production.py itself uses to pull in
30_kv_inject.py. Both are loaded read-only; nothing in probe/ is modified.

Key derived facts used throughout (see docstrings inline for where each was
checked against diffusers source, not assumed):
  - The transformer's omni (list-of-list `x`) forward path already offsets
    each image slot's RoPE F-axis by its running caption+image cursor
    (`transformer_z_image.py::patchify_and_embed_omni`, `cap_end_pos`), and
    gives every image its own H/W grid starting at (0,0) -- i.e. ref and
    target already get *independent* F frame numbers "for free" through the
    public forward() call; no manual position hacking is needed for 3b
    (unlike the refpos="center"/"lowfreq" H,W re-gridding InjectProcessor
    does for the *old* strip/KV-injection mechanism -- that code path is
    untouched and irrelevant to 3b).
  - `unpatchify()` with `x_pos_offsets` set (always true in omni_mode)
    returns ONLY the *last* image slot's reconstruction per batch item
    (transformer_z_image.py:496-519, "Return only the last (target) image").
    Concretely: `tf(x=[[ref, x_t]], ...)[0]` already IS the target-only
    prediction -- ref-token and cap-token loss exclusion falls out of the
    public forward()/unpatchify() contract for free, no extra masking code
    needed on our side.
  - Velocity-target direction: `pipeline_z_image.py` feeds
    `noise_pred = -transformer_raw_output` into
    `FlowMatchEulerDiscreteScheduler.step()`, whose `step()` implies
    `model_output := noise - x0` (standard `x0 = sample - sigma*model_output`
    rearrangement of `scale_noise`'s `sample = sigma*noise + (1-sigma)*x0`).
    So the RAW transformer output being supervised here is
    `x0 - noise` (NOT noise-x0), and `t` (the `t=` kwarg to `tf()`) is
    `1 - sigma` (`ts=1` <=> clean, matches `t_clean` in the omni dual-embed
    path being pinned to ones). Both facts were cross-checked against
    scheduler_config.json (num_train_timesteps=1000) and
    transformer_z_image.py's TimestepEmbedder t_scale=1000, not assumed.
  - Gradient checkpointing (`ModelMixin.enable_gradient_checkpointing`)
    defaults to `use_reentrant=False`, which -- verified empirically with a
    standalone repro (frozen Linear + zero-init trainable LoRA branch,
    requires_grad=False input) -- DOES correctly backprop into LoRA params
    inside a checkpointed block even when the block's own input tensor has
    requires_grad=False (our `x_t` is a constructed tensor, not a leaf
    requiring grad). No `enable_input_require_grads()`-style workaround is
    needed here.

Run:
  source ../env.sh
  $PY 45_train_lora.py --stage regression
  $PY 45_train_lora.py --stage profile
  $PY 45_train_lora.py --stage train --manifest data/train_manifest_7k5.jsonl --steps 1000  # NOT run by us
  $PY 45_train_lora.py --stage sample --ref u1000 --seed 0 [--adapter <ckpt_dir>]
"""
import os, sys, json, time, random, argparse, importlib.util, glob

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REF_IMG_DIR = "/mnt/local/smoke_anime/anime_260623/img"
RECAP_META = "/mnt/local/smoke_anime/anime_30k_recap/metadata.jsonl"
OUT_DIR = "/mnt/local/cherry_out/trainlight"
DEV = "cuda"

# LoRA capacity: design doc S3 "推荐" section -- 3b main path, capacity
# inherited from 3a's band prior (band10-25 x full QKVO, r=8) rather than
# spraying LoRA over all 30 layers. 16 layers x 4 projections x 2*3840*8
# params = 3,932,160 ~= 3.9M, matching the design doc's estimate exactly.
DEFAULT_BAND = "10-25"
LORA_PROJS = ["to_q", "to_k", "to_v", "to_out.0"]
CAP_REF_TXT = "anime illustration of a character"  # generic placeholder caption
                                                    # for the ref image slot,
                                                    # verbatim-matches 30_kv_inject.py's
                                                    # CAP_REF for the same slot.
                                                    # FALLBACK ONLY as of the S8(2)
                                                    # ref-caption playbook: stage=train
                                                    # uses a manifest row's own
                                                    # "ref_caption" field (probe/
                                                    # 53_ref_caption_wd14.py) when
                                                    # present, and stage=sample uses
                                                    # --cap_ref when given; this
                                                    # constant is only reached for
                                                    # rows/invocations that opt out
                                                    # (old manifests without the
                                                    # field, or no --cap_ref).


def _load(modname, filename):
    """Import a probe/ script as a module by file path, without editing it
    (identical pattern to 50_production.py's own `_load()`)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_band(s):
    s = s.strip()
    if "-" in s:
        lo, hi = s.split("-")
        return range(int(lo), int(hi) + 1)
    return [int(s)]


def lora_target_modules(band):
    return [f"layers.{i}.attention.{p}" for i in band for p in LORA_PROJS]


def build_lora_config(band, r, alpha, dropout):
    from peft import LoraConfig
    return LoraConfig(
        r=r, lora_alpha=alpha, target_modules=lora_target_modules(band),
        lora_dropout=dropout, bias="none", init_lora_weights=True,
    )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def cast_lora_fp32(model):
    """Design S4.2: frozen backbone stays bf16, LoRA params are kept fp32
    (master weights) and the fwd/bwd compute happens under bf16 autocast.
    peft's lora.Linear.forward runs `result = base_layer(x)` BEFORE ever
    touching lora_A/lora_B, using the untouched original `x` -- so this cast
    cannot perturb the frozen path's numerics (verified in stage=regression)."""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            n += p.numel()
    return n


# --------------------------------------------------------------------------
# shared data helpers (profile + train)
# --------------------------------------------------------------------------

def scan_available_pairs():
    """uid set with both u{uid}_ref.png and u{uid}_target.png under
    REF_IMG_DIR. Mirrors design doc S2.4's counting logic (not the manifest
    -- that's train-stage-only and comes from the parallel caption task)."""
    files = os.listdir(REF_IMG_DIR)
    refs = {f[1:-len("_ref.png")] for f in files if f.endswith("_ref.png")}
    tgts = {f[1:-len("_target.png")] for f in files if f.endswith("_target.png")}
    return refs & tgts


def load_recap_captions():
    caps = {}
    with open(RECAP_META) as f:
        for line in f:
            d = json.loads(line)
            caps[str(d["uid"])] = d["t2i_instruction"]
    return caps


def load_profile_samples(n, seed=20260711):
    """Small pool of (ref_path, target_path, caption) for throughput
    profiling only -- NOT an eval set, no relation to the dev-probe (2025) /
    frozen-acceptance (2026) / holdout30 (2024) / gen10 (42) protocol seeds
    in the design doc S5. Caption quality is explicitly out of scope for
    stage=profile per the task brief."""
    caps = load_recap_captions()
    avail = scan_available_pairs()
    inter = sorted(avail & set(caps.keys()))
    if not inter:
        raise RuntimeError("no uid has ref+target+caption all present -- check REF_IMG_DIR/RECAP_META")
    rng = random.Random(seed)
    rng.shuffle(inter)
    chosen = inter[: max(n, 1)]
    return [
        (f"{REF_IMG_DIR}/u{uid}_ref.png", f"{REF_IMG_DIR}/u{uid}_target.png", caps[uid])
        for uid in chosen
    ]


def sample_ts(t_dist, rng):
    """`ts` here is the model's native "clean fraction" input (1=clean,
    0=pure noise; see module docstring for the sign derivation). Uniform is
    the S7 M0 default; logit-normal is implemented (not a stub) behind
    --t_dist for the S4.3-flagged future swap, using the SD3/Flux-style
    sigmoid(N(0,1)) recipe with fixed m=0,s=1 (no schedule tuning done)."""
    if t_dist == "uniform":
        return rng.random()
    elif t_dist == "logitnormal":
        z = rng.gauss(0.0, 1.0)
        return 1.0 / (1.0 + pow(2.718281828459045, -z))
    raise ValueError(t_dist)


def encode_pair(kv, DT, ref_path, tgt_path, cap_txt, cap_ref_enc, t_dist, rng, get_latent=None):
    """Build one training example's transformer inputs.

    Sequence/RoPE convention: reuses kv.ref_latent() (== VAE encode-to-latent,
    the exact helper 30_kv_inject.py's capture forward already uses for the
    ref slot) for BOTH ref (clean) and target (noised) images, then hands
    them to the transformer as `x=[[ref_lat, x_t]]` with
    `image_noise_mask=[[0, 1]]` -- this is precisely the joint-sequence
    shape 30_kv_inject.py's "capture" forward already constructs (S3 3b:
    "capture forward 其实已经在构造 [ref, x0] 联合序列的雏形"), just now run
    WITH grad instead of being thrown away after harvesting K/V.

    `get_latent(path, size)` defaults to kv.ref_latent (no caching); stage=
    train passes a LatentCache-backed getter instead so --cache_latents can
    skip re-encoding a uid seen in an earlier epoch, without this function
    needing to know caching exists.

    Returns: (x_pair, cap_pair, ts, target_vel[fp32])
      x_pair  = [ref_lat(DT), x_t(DT)]
      cap_pair= [cap_ref_enc, cap_tgt_enc]
      target_vel = x0 - noise, fp32 (see module docstring for the sign proof)
    """
    get_latent = get_latent or kv.ref_latent
    ref_lat = get_latent(ref_path, 512)             # (16,1,32,32) DT, clean, no grad path (kv.ref_latent is @no_grad)
    x0 = get_latent(tgt_path, 1024).float()          # (16,1,128,128) fp32, clean target latent
    noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32)
    ts = sample_ts(t_dist, rng)
    sigma = 1.0 - ts
    x_t = (sigma * noise + (1.0 - sigma) * x0).to(DT)
    target_vel = x0 - noise
    cap_tgt_enc = kv.enc(cap_txt)
    return [ref_lat, x_t], [cap_ref_enc, cap_tgt_enc], ts, target_vel


def forward_backward(tf, x_list, cap_list, ts_list, target_list, opt):
    """One fwd(+bwd if opt given) step over a micro-batch already built by
    encode_pair(). Loss is plain flow-matching MSE, masked to the target
    segment for free (see module docstring re: unpatchify x_pos_offsets --
    `out[i]` below IS ALREADY only the target tokens' reconstruction, ref
    and cap tokens never appear in `out` at all)."""
    bs = len(x_list)
    t_tensor = torch.tensor(ts_list, device=DEV, dtype=torch.float32)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(opt is not None), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = tf(x=x_list, t=t_tensor, cap_feats=cap_list, siglip_feats=[None] * bs,
                  image_noise_mask=[[0, 1]] * bs, return_dict=False)[0]
        loss = sum(F.mse_loss(o.float(), tgt.float()) for o, tgt in zip(out, target_list)) / bs
    if opt is not None:
        loss.backward()
        opt.step()
    return loss.item()


# --------------------------------------------------------------------------
# additive: gated auxiliary decode-space losses (--loss_perc / --loss_ccip)
#
# Both default to weight 0.0 (off). forward_backward() above is completely
# UNTOUCHED -- when both flags are 0.0 (the default), stage_train() below
# calls forward_backward() exactly as before and none of this section's code
# ever executes (aux nets are never even imported). This section only adds
# a new, separately-called function plus new lazy-loaded helpers.
#
# Recipe verified by a feasibility scout ahead of this implementation
# (session scratchpad: gate2_mitigated.py for --loss_perc's decode+LPIPS@256+
# DINOv2 path and its VAE grad-checkpoint+slicing mitigations,
# gate1and2_combined.py for --loss_ccip's CCIP-torch identity term,
# ccip_torch_port_test3.py for the onnx->torch CCIP port itself
# [onnxsim-fold at fixed batch size, then onnx2torch; matched onnxruntime to
# 1.3e-5 max abs diff, confirmed differentiable], gate2_grad_sanity.py for an
# end-to-end proof the decode+perceptual path is not silently detached
# (LoRA gradient differs between mse-only and mse+perceptual on an identical
# batch/model-state)). Ported here verbatim where possible; not re-derived.
# --------------------------------------------------------------------------

AUX_CACHE_DIR = os.path.join(OUT_DIR, "aux_models_cache")  # our own namespace;
                                                             # never reads/writes
                                                             # any published
                                                             # checkpoint/data
_AUX_NETS = {}      # lazy cache: "perc" -> (lpips_net, dino_net); ("ccip", bs) -> ccip_torch_model
_NORM_CONSTS = {}   # lazy cache: device -> (imagenet_mean, imagenet_std, clip_mean, clip_std)


def get_perc_nets():
    """Lazy-load + cache LPIPS(vgg) and DINOv2-small, frozen eval mode on DEV.
    Only reached when --loss_perc > 0."""
    if "perc" not in _AUX_NETS:
        import lpips
        lp = lpips.LPIPS(net="vgg").to(DEV).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
        from transformers import AutoModel
        dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
        for p in dino.parameters():
            p.requires_grad_(False)
        _AUX_NETS["perc"] = (lp, dino)
    return _AUX_NETS["perc"]


def get_ccip_torch(bs):
    """Lazy-build + cache a differentiable torch port of CCIP
    (ccip-caformer-24-randaug-pruned), batch-size-LOCKED to `bs`, via the
    scout-verified recipe: onnxsim-fold the stock onnx (deepghs/ccip_onnx) to
    a fixed (bs,3,384,384) input shape, then onnx2torch.convert(). Simplified
    onnx is cached under AUX_CACHE_DIR (our own namespace) so repeat
    runs/processes at the same bs skip the onnxsim step.

    IMPORTANT: the returned module's input shape is fixed at conversion time
    -- every forward call MUST pass exactly `bs` images. This is why
    forward_backward_aux() below always runs the CCIP forward over the FULL
    microbatch and masks the resulting per-item loss AFTERWARD, rather than
    gathering only the low-sigma items into a smaller sub-batch first (that
    would not match this locked shape and would crash). Only reached when
    --loss_ccip > 0."""
    key = ("ccip", bs)
    if key not in _AUX_NETS:
        os.makedirs(AUX_CACHE_DIR, exist_ok=True)
        onnx_path = os.path.join(AUX_CACHE_DIR, f"ccip_feat_simplified_bs{bs}.onnx")
        if not os.path.exists(onnx_path):
            from huggingface_hub import hf_hub_download
            raw = hf_hub_download("deepghs/ccip_onnx", "ccip-caformer-24-randaug-pruned/model_feat.onnx")
            import onnxsim
            import onnx as onnx_pkg
            simplified, ok = onnxsim.simplify(raw, overwrite_input_shapes={"input": [bs, 3, 384, 384]})
            if not ok:
                raise RuntimeError("onnxsim.simplify() check failed for CCIP onnx")
            onnx_pkg.save(simplified, onnx_path)
        from onnx2torch import convert
        model = convert(onnx_path).to(DEV).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _AUX_NETS[key] = model
    return _AUX_NETS[key]


def _norm_consts():
    if DEV not in _NORM_CONSTS:
        imn_mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        imn_std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)
        _NORM_CONSTS[DEV] = (imn_mean, imn_std, clip_mean, clip_std)
    return _NORM_CONSTS[DEV]


def dino_embed(dino, pixels_pm1):
    """pixels_pm1: (B,3,H,W) in [-1,1], DIFFERENTIABLE preprocessing standing
    in for AutoImageProcessor (PIL/numpy-based, would break the graph)."""
    imn_mean, imn_std, _, _ = _norm_consts()
    x01 = ((pixels_pm1 + 1.0) / 2.0).clamp(0, 1)
    x224 = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
    x224 = (x224 - imn_mean) / imn_std
    out = dino(pixel_values=x224.to(dino.dtype))
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]


def ccip_embed(ccip_model, pixels_pm1):
    """pixels_pm1: (B,3,H,W) in [-1,1] -> CCIP's own preprocessing (resize
    384, CLIP-norm) reimplemented in torch (differentiable), in place of
    imgutils' PIL-based _preprocess_image."""
    _, _, clip_mean, clip_std = _norm_consts()
    x01 = ((pixels_pm1 + 1.0) / 2.0).clamp(0, 1)
    x384 = F.interpolate(x01, size=(384, 384), mode="bilinear", align_corners=False)
    x384 = (x384 - clip_mean) / clip_std
    out = ccip_model(x384.float())
    return out[0] if isinstance(out, (tuple, list)) else out


def load_pixels(paths, size):
    """Real PNGs -> (B,3,size,size) tensor in [-1,1], matching VAE decode's
    output convention (LANCZOS resize, same as encode_pair()'s ref_latent()
    preprocessing). No grad path -- ground truth, always used under
    torch.no_grad()."""
    import numpy as np
    from PIL import Image
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
        arrs.append(arr)
    return torch.stack(arrs).to(DEV)


def reconstruct_x0(x_t, v_hat, sigma):
    """45's documented velocity convention (see module docstring):
    target_vel = x0 - noise, x_t = sigma*noise + (1-sigma)*x0
    => x0 = x_t + sigma*v_hat."""
    return x_t + sigma * v_hat


def decode_to_pixels(pipe, x0_hat_batched):
    lat = x0_hat_batched.to(pipe.vae.dtype)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    return img


def forward_backward_aux(tf, pipe, x_list, cap_list, ts_list, target_list, ref_paths, tgt_paths,
                          opt, w_perc, w_ccip):
    """forward_backward() + two additive, gated auxiliary loss terms computed
    in DECODED PIXEL SPACE. Only called when w_perc>0 or w_ccip>0.

      --loss_perc W: adds W*(LPIPS-VGG@256 + DINOv2-small cosine distance),
        decoded x0_hat vs the TARGET image pixels (fidelity to the specific
        training target this step supervises).
      --loss_ccip W: adds W*(1 - CCIP-torch cosine similarity), decoded
        x0_hat vs the REFERENCE image pixels (identity similarity to the
        conditioning ref, independent of target pose/outfit/scene).

    GATING (critical, per the feasibility scout): x0_hat = x_t + sigma*v_hat
    is only a meaningful clean-image estimate near the END of denoising (low
    sigma / high ts); at high sigma it is mostly reconstructed noise, and
    decoding+scoring it would inject garbage gradient into the LoRA. Both
    terms are therefore masked to ts>0.5 (sigma<0.5) samples ONLY, per-sample
    within the microbatch (t_dist samples ts independently per item, so a
    batch is typically a mix), via a masked mean (sum over masked items /
    count of masked items, NOT / bs -- keeps the per-included-item loss scale
    constant regardless of how many items in a given batch happen to
    qualify).

    The VAE decode + both aux nets always run over the FULL bs-sized
    microbatch (same shape every step, matching the feasibility scout's
    profiled memory numbers exactly) -- see get_ccip_torch()'s docstring for
    why (CCIP onnx graph is batch-size-locked; a smaller gathered sub-batch
    would crash it). The mask is applied AFTERWARD, only to the resulting
    per-item scalar losses. The one exception: if ZERO items in a batch
    qualify (possible but rare at bs=4 uniform ts: 0.5**4 ~= 6%), the entire
    decode+aux block is skipped for that step -- pure compute-saving (nothing
    to mask into a non-empty mean, and it avoids a 0/0)."""
    bs = len(x_list)
    t_tensor = torch.tensor(ts_list, device=DEV, dtype=torch.float32)
    opt.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = tf(x=x_list, t=t_tensor, cap_feats=cap_list, siglip_feats=[None] * bs,
                  image_noise_mask=[[0, 1]] * bs, return_dict=False)[0]
        mse = sum(F.mse_loss(o.float(), tgt.float()) for o, tgt in zip(out, target_list)) / bs

    mask = torch.tensor([1.0 if ts > 0.5 else 0.0 for ts in ts_list], device=DEV, dtype=torch.float32)
    n_gated = int(mask.sum().item())

    perc_val = torch.zeros((), device=DEV)
    ccip_val = torch.zeros((), device=DEV)

    if n_gated > 0 and (w_perc > 0 or w_ccip > 0):
        x0_hats = []
        for i in range(bs):
            v_hat = out[i].float()
            x_t_i = x_list[i][1].float()
            sigma_i = 1.0 - ts_list[i]
            x0_hats.append(reconstruct_x0(x_t_i, v_hat, sigma_i))
        x0_hat_batched = torch.stack(x0_hats, dim=0).squeeze(2)
        pred_pixels = decode_to_pixels(pipe, x0_hat_batched).clamp(-1, 1)  # (B,3,1024,1024)

        if w_perc > 0:
            loss_fn_lpips, dino = get_perc_nets()
            pred_lp = F.interpolate(pred_pixels, size=(256, 256), mode="bilinear", align_corners=False)
            with torch.no_grad():
                tgt_px = load_pixels(tgt_paths, pred_pixels.shape[-1])
                tgt_lp = F.interpolate(tgt_px, size=(256, 256), mode="bilinear", align_corners=False)
                tgt_dino_emb = dino_embed(dino, tgt_px)
            lpips_per_item = loss_fn_lpips(pred_lp, tgt_lp).view(bs)
            pred_dino_emb = dino_embed(dino, pred_pixels)
            dino_per_item = 1.0 - F.cosine_similarity(pred_dino_emb, tgt_dino_emb, dim=-1)
            perc_per_item = lpips_per_item + dino_per_item
            perc_val = (perc_per_item * mask).sum() / n_gated

        if w_ccip > 0:
            ccip_model = get_ccip_torch(bs)
            with torch.no_grad():
                ref_px = load_pixels(ref_paths, pred_pixels.shape[-1])
                ref_ccip_emb = ccip_embed(ccip_model, ref_px)
            pred_ccip_emb = ccip_embed(ccip_model, pred_pixels)
            ccip_per_item = 1.0 - F.cosine_similarity(pred_ccip_emb, ref_ccip_emb, dim=-1)
            ccip_val = (ccip_per_item * mask).sum() / n_gated

    loss = mse + w_perc * perc_val + w_ccip * ccip_val
    loss.backward()
    opt.step()
    return dict(loss=loss.item(), mse=mse.item(),
                perc=perc_val.item() if w_perc > 0 else None,
                ccip=ccip_val.item() if w_ccip > 0 else None,
                n_gated=n_gated, bs=bs)


# --------------------------------------------------------------------------
# stage: regression  (M0 judgement a)
# --------------------------------------------------------------------------

def stage_regression(args):
    prod = _load("prod50_regr", "50_production.py")
    gen = prod.IdentityGenerator()   # loads the pipeline ONCE, installs InjectProcessor
                                      # on all 30 layers exactly as 30_kv_inject.py's
                                      # module-level code does (that file's __main__
                                      # guard doesn't gate the load/install, only main())
    kv = gen._kv
    pipe = kv.pipe

    if args.ref:
        ref_path = args.ref
    else:
        cands = sorted(f for f in os.listdir(REF_IMG_DIR) if f.endswith("_ref.png"))
        if not cands:
            print(f"[regression] FAIL: no *_ref.png under {REF_IMG_DIR}", flush=True)
            sys.exit(1)
        ref_path = os.path.join(REF_IMG_DIR, cands[0])
    print(f"[regression] ref={ref_path} seed={args.seed} recipe={prod.RECIPE}", flush=True)

    # Intercept the pre-VAE-decode latent tensor (the "latent level" output
    # the task asks us to diff) by wrapping the bound method on this one
    # loaded vae instance -- no file is edited, this is the same kind of
    # runtime attribute swap InjectProcessor itself uses (set_processor()).
    captured = {}
    orig_decode = pipe.vae.decode

    def spy_decode(lat, *a, **kw):
        captured["lat"] = lat.detach().float().clone()
        return orig_decode(lat, *a, **kw)

    def run_once(seed):
        pipe.vae.decode = spy_decode
        try:
            img = gen.generate(ref_path, seed=seed)
        finally:
            pipe.vae.decode = orig_decode
        return captured["lat"].clone(), img

    kv.STORE.clear()
    lat0, img0 = run_once(args.seed)
    store_keys_before = sorted(kv.STORE.keys())
    print(f"[regression] no-LoRA baseline: latent shape={tuple(lat0.shape)}, "
          f"InjectProcessor STORE populated at layers={store_keys_before}", flush=True)

    lora_cfg = build_lora_config(parse_band(args.band), args.r, args.alpha, 0.0)
    kv.tf.add_adapter(lora_cfg)
    total, trainable = count_params(kv.tf)
    print(f"[regression] after add_adapter: total_params={total:,} trainable={trainable:,} "
          f"(expected ~3.9M for band={args.band} r={args.r})", flush=True)
    n_fp32 = cast_lora_fp32(kv.tf)
    print(f"[regression] cast {n_fp32:,} LoRA params to fp32", flush=True)

    # Sanity: is peft's Linear transparently callable from InjectProcessor's
    # `attn.to_q(hidden_states)`-style call sites?
    sample_layer = kv.tf.layers[parse_band(args.band)[0]].attention.to_q
    print(f"[regression] tf.layers[{parse_band(args.band)[0]}].attention.to_q -> "
          f"{type(sample_layer).__name__} (peft-wrapped: {'lora' in type(sample_layer).__module__})",
          flush=True)

    kv.STORE.clear()
    lat1, img1 = run_once(args.seed)
    store_keys_after = sorted(kv.STORE.keys())
    diff = (lat1 - lat0).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    bit_exact = max_diff == 0.0
    print(f"[regression] with-LoRA(B=0) run: InjectProcessor STORE populated at layers={store_keys_after}",
          flush=True)
    print(f"[regression] max_abs_diff={max_diff:.6e} mean_abs_diff={mean_diff:.6e} bit_exact={bit_exact}",
          flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    img0.save(f"{OUT_DIR}/regression_noLoRA.png")
    img1.save(f"{OUT_DIR}/regression_withLoRA_Bzero.png")
    result = dict(
        ref=ref_path, seed=args.seed, band=args.band, r=args.r, alpha=args.alpha,
        trainable_params=trainable, total_params=total,
        store_keys_before=store_keys_before, store_keys_after=store_keys_after,
        max_abs_diff=max_diff, mean_abs_diff=mean_diff, bit_exact=bit_exact,
        inject_processor_transparent=("lora" in type(sample_layer).__module__),
    )
    with open(f"{OUT_DIR}/regression_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    if not bit_exact:
        print("[regression] NOT bit-exact -- see design doc S6 plan-B (hand-written "
              "LoRA math inside InjectProcessor.__call__) if this is a real "
              "incompatibility and not a dtype-cast artifact.", flush=True)


# --------------------------------------------------------------------------
# stage: profile  (M0 judgement b)
# --------------------------------------------------------------------------

def stage_profile(args):
    kv = _load("kv_inject_prof", "30_kv_inject.py")
    tf, DT = kv.tf, kv.DT
    print(f"[profile] pipeline loaded, transformer dtype={DT}", flush=True)
    print(f"[profile] cuda mem right after load: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
          f"free/total = {[x/1e9 for x in torch.cuda.mem_get_info()]}", flush=True)

    lora_cfg = build_lora_config(parse_band(args.band), args.r, args.alpha, 0.0)
    tf.add_adapter(lora_cfg)
    total, trainable = count_params(tf)
    cast_lora_fp32(tf)
    print(f"[profile] trainable LoRA params: {trainable:,} (band={args.band} r={args.r})", flush=True)

    tf.enable_gradient_checkpointing()
    tf.train()
    opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=args.lr)

    samples = load_profile_samples(args.n_samples)
    print(f"[profile] {len(samples)} distinct (ref,target,caption) samples in the profiling pool", flush=True)
    cap_ref_enc = kv.enc(CAP_REF_TXT)  # frozen generic ref-slot caption, computed once & reused

    results = {}
    bs_list = [int(x) for x in args.bs_list.split(",")]
    for bs in bs_list:
        rng = random.Random(1000 + bs)
        n_iters = args.warmup + args.measure

        # Precompute all micro-batches for this bs BEFORE timing anything --
        # the task asks for fwd+bwd throughput specifically, not VAE-encode /
        # text-encode cost (those would dominate at this tiny LoRA param
        # count and are not what S4.3's 0.7-1s/sample estimate is about).
        batches = []
        for it in range(n_iters):
            x_list, cap_list, ts_list, tgt_list = [], [], [], []
            for b in range(bs):
                ref_path, tgt_path, cap_txt = samples[(it * bs + b) % len(samples)]
                xp, cp_, ts, tgt = encode_pair(kv, DT, ref_path, tgt_path, cap_txt, cap_ref_enc,
                                                args.t_dist, rng)
                x_list.append(xp); cap_list.append(cp_); ts_list.append(ts); tgt_list.append(tgt)
            batches.append((x_list, cap_list, ts_list, tgt_list))

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        for it in range(args.warmup):
            x_list, cap_list, ts_list, tgt_list = batches[it]
            forward_backward(tf, x_list, cap_list, ts_list, tgt_list, opt)
        torch.cuda.synchronize()

        step_times = []
        losses = []
        for it in range(args.warmup, n_iters):
            x_list, cap_list, ts_list, tgt_list = batches[it]
            torch.cuda.synchronize()
            t0 = time.time()
            loss = forward_backward(tf, x_list, cap_list, ts_list, tgt_list, opt)
            torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            losses.append(loss)

        mean_step = sum(step_times) / len(step_times)
        mean_sample = mean_step / bs
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        ratio_lo, ratio_hi = mean_sample / 1.0, mean_sample / 0.7  # vs S4.3's 0.7-1s/sample band
        print(f"[profile] bs={bs}: mean_step={mean_step:.3f}s mean_sample={mean_sample:.3f}s "
              f"peak_mem={peak_gb:.2f}GB loss_last={losses[-1]:.4f} "
              f"(design S4.3 estimate 0.7-1s/sample -> ratio {ratio_lo:.2f}x-{ratio_hi:.2f}x)",
              flush=True)
        results[bs] = dict(mean_step_s=mean_step, mean_sample_s=mean_sample, peak_mem_gb=peak_gb,
                            step_times=step_times, losses=losses)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/profile_result.json", "w") as f:
        json.dump(dict(band=args.band, r=args.r, alpha=args.alpha, trainable_params=trainable,
                        warmup=args.warmup, measure=args.measure, t_dist=args.t_dist,
                        results=results), f, indent=2)
    print(f"[profile] wrote {OUT_DIR}/profile_result.json", flush=True)


# --------------------------------------------------------------------------
# stage: train  (written per the manifest contract, NOT executed by us --
# manifest is produced by a parallel task and may not exist yet)
# --------------------------------------------------------------------------

def read_manifest(path):
    if not os.path.exists(path):
        print(f"[train] FAIL: manifest not found at {path} -- this file is written "
              f"against the contract {{uid, ref, target, caption_scrubbed}} per line, "
              f"produced by a parallel caption-scrubbing task; nothing to do until it "
              f"exists.", flush=True)
        sys.exit(1)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    for req in ("uid", "ref", "target", "caption_scrubbed"):
        if rows and req not in rows[0]:
            raise ValueError(f"manifest row missing required field {req!r}: {rows[0]}")
    return rows


class LatentCache:
    """Optional in-RAM cache keyed by (path, size); --cache_latents turns
    this on. At 7.5k-28.7k rows this is small (~5-20GB of bf16 tensors on
    CPU) so a plain dict is enough -- no need for an LRU or a disk-backed
    prefetch pipeline at this scale (design S4.4: 7.5k doesn't need
    precaching, this is purely an optional speed knob)."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.store = {}

    def get(self, kv, path, size):
        if not self.enabled:
            return kv.ref_latent(path, size)
        key = (path, size)
        if key not in self.store:
            self.store[key] = kv.ref_latent(path, size).cpu()
        return self.store[key].to(DEV)


class CapRefCache:
    """S8(2) ref-caption playbook: encoded-text cache for the ref slot's
    cap_ref, keyed by the caption STRING (not uid) -- two different uids that
    happen to render the same wd14 phrase, or the same uid's oversampled
    duplicate rows (train_manifest_mix3.jsonl's x8 solo rows), all hit one
    encode() call. Unconditionally on (no --flag, unlike --cache_latents):
    text embeddings here are a handful of ~100-token bf16 vectors per unique
    string, nowhere near LatentCache's multi-GB image-latent scale, so there
    is no memory tradeoff to gate. Independent of --cache_latents, which only
    ever caches (path,size)->image-latent tensors and never touches captions
    -- the two caches cannot collide or need to agree with each other."""

    def __init__(self):
        self.store = {}

    def get(self, kv, text):
        if text not in self.store:
            self.store[text] = kv.enc(text).cpu()
        return self.store[text].to(DEV)


def save_checkpoint(tf, opt, step, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    tf.save_lora_adapter(ckpt_dir)  # standard diffusers PeftAdapterMixin save (adapter-only)
    torch.save({"step": step, "optimizer": opt.state_dict()}, os.path.join(ckpt_dir, "trainer_state.pt"))
    print(f"[train] checkpoint saved: {ckpt_dir} (step={step})", flush=True)


def find_latest_ckpt(save_root):
    # only complete checkpoints: a crashed save leaves a partial step_* dir
    # (makedirs succeeds, weight/state files never land) that must be skipped
    cands = sorted((p for p in glob.glob(os.path.join(save_root, "step_*"))
                    if os.path.exists(os.path.join(p, "pytorch_lora_weights.safetensors"))
                    and os.path.exists(os.path.join(p, "trainer_state.pt"))),
                    key=lambda p: int(p.rsplit("_", 1)[-1]))
    return cands[-1] if cands else None


def stage_train(args):
    rows = read_manifest(args.manifest)
    print(f"[train] manifest {args.manifest}: {len(rows)} rows", flush=True)

    kv = _load("kv_inject_train", "30_kv_inject.py")
    tf, DT, pipe = kv.tf, kv.DT, kv.pipe
    os.makedirs(args.save_dir, exist_ok=True)

    if args.unope:
        # UnoPE-style diagonal H/W RoPE offset for the ref slot (probe/
        # 57_unope_patch.py) -- runtime monkeypatch on THIS tf instance only,
        # diffusers source untouched. Opt-in (--unope, default off) so every
        # existing/concurrent run of this file that doesn't pass --unope is
        # byte-for-byte unaffected. MUST be installed identically here and in
        # stage_sample() below, or train/inference RoPE geometry mismatches.
        _load("unope57_train", "57_unope_patch.py").install_unope_patch(tf)

    step0 = 0
    if args.resume:
        resume_dir = args.resume if os.path.isdir(args.resume) else find_latest_ckpt(args.save_dir)
        if resume_dir is None:
            print(f"[train] --resume given but no checkpoint found under {args.save_dir}", flush=True)
            sys.exit(1)
        print(f"[train] resuming from {resume_dir}", flush=True)
        # prefix=None: our checkpoints are saved directly off `tf` (not a
        # full pipeline), so keys have no "transformer." prefix to strip.
        # weight_name explicit: see stage_sample -- diffusers' load default
        # is the .bin name while save writes .safetensors.
        # adapter_name explicit: load's auto-name ("default_0") differs from
        # save_lora_adapter's default ("default") -- without this the first
        # post-resume checkpoint save raises "Adapter name default not found".
        tf.load_lora_adapter(resume_dir, prefix=None, adapter_name="default",
                              weight_name="pytorch_lora_weights.safetensors")
        cast_lora_fp32(tf)  # loaded weights inherit bf16 from base; keep LoRA fp32 like fresh path
        state = torch.load(os.path.join(resume_dir, "trainer_state.pt"), map_location=DEV)
        step0 = state["step"]
        opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=args.lr)
        opt.load_state_dict(state["optimizer"])
    else:
        lora_cfg = build_lora_config(parse_band(args.band), args.r, args.alpha, args.dropout)
        tf.add_adapter(lora_cfg)
        cast_lora_fp32(tf)
        opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=args.lr)

    total, trainable = count_params(tf)
    print(f"[train] trainable LoRA params: {trainable:,} (band={args.band} r={args.r} alpha={args.alpha})",
          flush=True)

    tf.enable_gradient_checkpointing()
    tf.train()

    need_aux = args.loss_perc > 0 or args.loss_ccip > 0
    if need_aux:
        # additive mitigations (feasibility scout, gate2_mitigated.py): both
        # stock diffusers AutoencoderKL toggles, zero new code, cut naive
        # decode+perceptual peak memory from +61.9GB to +6.4GB over baseline
        # in the scout's isolated measurement.
        pipe.vae.enable_gradient_checkpointing()
        pipe.vae.enable_slicing()
        print(f"[train] aux losses ON: loss_perc={args.loss_perc} loss_ccip={args.loss_ccip} "
              f"(gated to ts>0.5 samples only; pipe.vae grad-checkpoint+slicing enabled)", flush=True)

    cache = LatentCache(args.cache_latents)
    cap_ref_cache = CapRefCache()  # S8(2): per-row ref_caption encode cache
                                    # (see CapRefCache docstring re: no
                                    # interaction with --cache_latents)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    def epoch_iter():
        # Simple step-based (not epoch-based) cursor over a reshuffled
        # manifest, per the --steps N CLI contract in the task brief.
        rng = random.Random(0)
        order = list(range(len(rows)))
        while True:
            rng.shuffle(order)
            for i in order:
                yield rows[i]

    it = epoch_iter()
    # fast-forward the iterator to the resume point so re-running doesn't
    # replay already-seen rows identically every restart (best-effort --
    # exact resumption of the RNG stream isn't attempted, only step count).
    for _ in range(step0 * args.bs):
        next(it)

    get_latent = (lambda path, size: cache.get(kv, path, size))
    rng_t = random.Random(0)
    t0_wall = time.time()
    for step in range(step0, args.steps):
        x_list, cap_list, ts_list, tgt_list, ref_paths, tgt_paths = [], [], [], [], [], []
        for _ in range(args.bs):
            row = next(it)
            # S8(2): per-row ref_caption if the manifest has it, else the
            # frozen CAP_REF_TXT constant (old manifests / back-compat).
            ref_cap_txt = row.get("ref_caption") or CAP_REF_TXT
            cap_ref_enc = cap_ref_cache.get(kv, ref_cap_txt)
            if step < step0 + 3:  # cheap per-batch proof cap_ref tracks the row, not a constant
                print(f"[train] step={step} uid={row['uid']} ref_caption[:40]={ref_cap_txt[:40]!r}",
                      flush=True)
            xp, cp_, ts, tgt = encode_pair(kv, DT, row["ref"], row["target"], row["caption_scrubbed"],
                                            cap_ref_enc, args.t_dist, rng_t, get_latent=get_latent)
            x_list.append(xp); cap_list.append(cp_); ts_list.append(ts); tgt_list.append(tgt)
            ref_paths.append(row["ref"]); tgt_paths.append(row["target"])  # only read by
                                                                            # forward_backward_aux
                                                                            # when need_aux; cheap
                                                                            # python-side no-op
                                                                            # otherwise

        if need_aux:
            stats = forward_backward_aux(tf, pipe, x_list, cap_list, ts_list, tgt_list,
                                          ref_paths, tgt_paths, opt, args.loss_perc, args.loss_ccip)
            loss = stats["loss"]
        else:
            loss = forward_backward(tf, x_list, cap_list, ts_list, tgt_list, opt)
            stats = None
        log_record = {"step": step, "loss": loss, "lr": args.lr, "wall_s": time.time() - t0_wall}
        if stats is not None:
            log_record.update({"mse": stats["mse"], "perc": stats["perc"],
                                "ccip": stats["ccip"], "n_gated": stats["n_gated"]})
        log_f.write(json.dumps(log_record) + "\n")
        log_f.flush()
        if step % args.log_every == 0:
            if stats is not None:
                print(f"[train] step={step} loss={loss:.4f} mse={stats['mse']:.4f} "
                      f"perc={stats['perc']} ccip={stats['ccip']} n_gated={stats['n_gated']}/{args.bs}",
                      flush=True)
            else:
                print(f"[train] step={step} loss={loss:.4f}", flush=True)
        if step > step0 and step % args.save_every == 0:
            save_checkpoint(tf, opt, step, os.path.join(args.save_dir, f"step_{step}"))

    save_checkpoint(tf, opt, args.steps, os.path.join(args.save_dir, f"step_{args.steps}"))
    log_f.close()
    print(f"[train] done: {args.steps} steps, log at {log_path}", flush=True)


# --------------------------------------------------------------------------
# stage: sample  (3b inference path, needed by M0 judgement d + M1 dev-probe)
# --------------------------------------------------------------------------

def resolve_refs(spec):
    """--ref accepts either a single image path, or a comma-separated uid
    list resolved against REF_IMG_DIR (u{uid}_ref.png), matching the uid
    conventions of the other probe scripts."""
    if os.path.exists(spec):
        return [spec]
    uids = [u.strip().lstrip("u") for u in spec.split(",") if u.strip()]
    return [f"{REF_IMG_DIR}/u{u}_ref.png" for u in uids]


def uid_of(ref_path):
    b = os.path.basename(ref_path)
    if b.endswith("_ref.png"):
        b = b[: -len("_ref.png")]
    return b[1:] if b.startswith("u") else b


def stage_sample(args):
    """Full denoising sampler over the SAME joint-sequence convention as the
    training forward: every step is one `tf(x=[[ref_lat, x_t]], ...,
    image_noise_mask=[[0,1]])` per CFG branch -- ref slot clean, ref-slot
    caption = --cap_ref if given, else CAP_REF_TXT (S8(2) playbook: matches
    stage=train's per-row ref_caption/CAP_REF_TXT fallback -- pass the same
    text used for that uid's row, e.g. via data/devprobe12_refcaps.json, to
    keep train/inference consistent). With a trained --adapter loaded this IS
    the 3b sampler for M0(d)/M1 dev-probe; without --adapter a fresh
    zero-init (B=0) LoRA is attached, giving the deterministic "pure T2I +
    ref prefix present" baseline.

    NOTE this baseline is NOT the strip/KV-injection production output and no
    equivalence is expected -- 3b conditions through in-context tokens, the
    production recipe conditions through captured K/V; same base weights,
    different mechanism. The InjectProcessor stays installed but idle
    (INJ["mode"]=="off" throughout; module-level install in 30_kv_inject.py).

    CFG: text cond vs text uncond (kv.NEG) only; the ref prefix is kept in
    BOTH branches per the frozen decision -- if the ref prefix in the uncond
    branch turns out to shift baseline behavior, record it, don't add switch
    combinations here. cfg/steps/size/refsize default to 50_production.py's
    RECIPE values (5.0 / 50 / 1024 / 512); scheduler setup (calculate_shift
    mu, sigma_min=0.0, retrieve_timesteps) copied verbatim from
    30_kv_inject.generate()'s production path.

    Prompt resolution order: --prompt text > holdout30_tags.json prompt_B
    (when the uid is one of the 30 holdout subjects; keeps M1/终评 caption
    parity with the other arms per design S5.3) > kv.PROMPT generic.
    """
    kv = _load("kv_inject_sample", "30_kv_inject.py")
    tf, DT, pipe = kv.tf, kv.DT, kv.pipe

    if args.unope:
        # Same monkeypatch, same install call as stage_train() above (project
        # red line: train/inference geometry must match bit-for-bit). Opt-in,
        # default off.
        _load("unope57_sample", "57_unope_patch.py").install_unope_patch(tf)

    if args.adapter:
        # weight_name must be given explicitly: save_lora_adapter writes
        # pytorch_lora_weights.safetensors but load_lora_adapter's default
        # search path is the .bin name (asymmetric defaults in diffusers
        # 0.38.0.dev0, verified by traceback), so a bare directory fails.
        tf.load_lora_adapter(args.adapter, prefix=None,
                              weight_name="pytorch_lora_weights.safetensors")
        tag = "adpt-" + os.path.basename(os.path.normpath(args.adapter))
    else:
        tf.add_adapter(build_lora_config(parse_band(args.band), args.r, args.alpha, 0.0))
        cast_lora_fp32(tf)
        tag = "B0"
    _, trainable = count_params(tf)
    tf.eval()
    print(f"[sample] adapter={args.adapter or 'zero-init B=0'} trainable={trainable:,}", flush=True)

    hpath = os.path.join(HERE, "..", "data", "holdout30_tags.json")
    holdout = json.load(open(hpath)) if os.path.exists(hpath) else {}

    steps = args.steps if args.steps is not None else 50   # production RECIPE steps
    size, refsize, cfg = 1024, 512, args.cfg
    os.makedirs(OUT_DIR, exist_ok=True)

    cap_ref_txt = args.cap_ref if args.cap_ref is not None else CAP_REF_TXT

    for ref_path in resolve_refs(args.ref):
        uid = uid_of(ref_path)
        prompt = args.prompt or holdout.get(uid, {}).get("prompt_B") or kv.PROMPT
        print(f"[sample] u{uid} seed={args.seed} steps={steps} cfg={cfg} "
              f"cap_ref={cap_ref_txt[:60]!r} prompt={prompt[:80]!r}...", flush=True)
        with torch.no_grad():
            cap_ref = kv.enc(cap_ref_txt)
            cap_pos = kv.enc(prompt)
            cap_neg = kv.enc(kv.NEG)
            ref_lat = kv.ref_latent(ref_path, refsize)

            gen = torch.Generator(DEV).manual_seed(args.seed)
            hlat = 2 * (size // 16)
            latents = torch.randn((1, 16, hlat, hlat), generator=gen, device=DEV,
                                   dtype=torch.float32)

            sched = pipe.scheduler
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

            for t in timesteps:
                ts = ((1000 - t.expand(latents.shape[0])) / 1000)
                x_t = latents.to(DT).unsqueeze(2)[0]        # (16,1,H,W)

                def joint_fwd(cap_txt_enc):
                    # same call shape as the training forward_backward();
                    # [0][0] = target-slot velocity (x0-noise direction),
                    # ref/cap tokens already excluded by omni unpatchify.
                    return tf(x=[[ref_lat, x_t]], t=ts, cap_feats=[[cap_ref, cap_txt_enc]],
                              siglip_feats=[None], image_noise_mask=[[0, 1]],
                              return_dict=False)[0][0].float()

                cond = joint_fwd(cap_pos)
                unc = joint_fwd(cap_neg)
                pred = cond + cfg * (cond - unc)
                noise_pred = (-pred).unsqueeze(0).squeeze(2)  # (1,16,H,W)
                latents = sched.step(noise_pred.to(torch.float32), t, latents,
                                      return_dict=False)[0]

            n_nan = torch.isnan(latents).sum().item()
            lat = latents.to(pipe.vae.dtype)
            lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            img = pipe.vae.decode(lat, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img, output_type="pil")[0]

        out_path = f"{OUT_DIR}/sample_u{uid}_seed{args.seed}_{tag}.png"
        img.save(out_path)
        print(f"[sample] u{uid}: nan_latents={n_nan} wrote {out_path}", flush=True)
        if n_nan:
            print(f"[sample] WARNING: u{uid} produced {n_nan} NaN latent entries", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["regression", "profile", "train", "sample"])
    # shared LoRA hyperparams (design S3 recommendation / S7 M0)
    ap.add_argument("--band", default=DEFAULT_BAND, help="e.g. '10-25' or '21-25'")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--t_dist", default="uniform", choices=["uniform", "logitnormal"])
    ap.add_argument("--unope", action="store_true", help="install the UnoPE diagonal H/W ref-slot "
                                                          "RoPE-offset monkeypatch (probe/57_unope_patch.py) "
                                                          "before train/sample forward() calls; default off "
                                                          "(stock shared-origin geometry, unaffected)")
    # regression + sample
    ap.add_argument("--ref", default=None, help="regression: ref image path (default: first "
                                                 "*_ref.png); sample: image path or comma-separated uids")
    ap.add_argument("--seed", type=int, default=0)
    # sample
    ap.add_argument("--prompt", default=None, help="sample: cond text (default: holdout30 "
                                                    "prompt_B if uid matches, else generic PROMPT)")
    ap.add_argument("--adapter", default=None, help="sample: LoRA checkpoint dir (default: "
                                                     "fresh zero-init adapter = pure-T2I+ref-prefix baseline)")
    ap.add_argument("--cfg", type=float, default=5.0, help="sample: CFG scale (production value)")
    ap.add_argument("--cap_ref", default=None, help="sample: ref-slot caption text (S8(2) playbook; "
                                                     "default: CAP_REF_TXT constant, matching training "
                                                     "on manifests without a ref_caption field). Pass "
                                                     "the same string a uid's manifest row used, e.g. "
                                                     "data/devprobe12_refcaps.json[uid], to keep "
                                                     "train/inference consistent.")
    # profile
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--bs_list", default="1,4")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=20)
    # train
    ap.add_argument("--manifest", default="data/train_manifest_7k5.jsonl")
    ap.add_argument("--steps", type=int, default=None,
                     help="train: optimizer steps (default 2000); sample: denoise steps (default 50)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--save_dir", default=OUT_DIR)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--cache_latents", action="store_true")
    ap.add_argument("--loss_perc", type=float, default=0.0,
                     help="train: additive weight on decoded-x0_hat LPIPS-VGG@256 + DINOv2-small "
                          "loss vs TARGET pixels (default 0.0 = off, byte-identical to the "
                          "pre-existing mse-only path). Gated to ts>0.5 (low-sigma) samples only "
                          "-- see forward_backward_aux() docstring.")
    ap.add_argument("--loss_ccip", type=float, default=0.0,
                     help="train: additive weight on decoded-x0_hat CCIP-torch identity-embedding "
                          "distance vs REFERENCE pixels (default 0.0 = off). Same ts>0.5 gating as "
                          "--loss_perc. The CCIP onnx graph gets batch-size-locked to --bs on first "
                          "use within a process -- keep --bs constant across a run/resume.")
    ap.add_argument("--resume", default=None, help="checkpoint dir, or any truthy value to "
                                                     "auto-find the latest under --save_dir")
    a = ap.parse_args()

    if a.stage == "regression":
        stage_regression(a)
    elif a.stage == "profile":
        stage_profile(a)
    elif a.stage == "train":
        if a.steps is None:
            a.steps = 2000
        stage_train(a)
    elif a.stage == "sample":
        if not a.ref:
            ap.error("--ref is required for stage=sample")
        stage_sample(a)


if __name__ == "__main__":
    main()
