#!/usr/bin/env python
"""
80_lumina_inject.py -- port of probe/30_kv_inject.py's ref-KV injection +
head-gate mechanism from Z-Image onto the frozen Lumina-Image-2.0 base
(Lumina2Transformer2DModel, 24 query heads / 8 kv heads GQA n_rep=3 / 26
main joint-attention layers, 3D RoPE axes_dim_rope=[32,32,32], qk-norm).

Cross-model replication target: "frozen T2I DiT distributes identity across
attention heads; the number of heads receiving injected ref K/V is an
identity<->defect dial." This script is the MINIMAL port needed to run that
head-count sweep on a second architecture family -- no Z-Image-specific
extras (no lowfreq/center RoPE surgery, no spatial routing, no per-step
schedule). Just: capture -> concat-inject -> optional head-gate.

Mechanism (mirrors 30_kv_inject.py):
  * Custom InjectProcessor subclasses nothing but replicates
    Lumina2AttnProcessor2_0's qk-norm/RoPE/GQA/SDPA exactly, installed via
    block.attn.set_processor() on all 26 `transformer.layers` blocks. The
    diffusers library itself is never modified.
  * capture: a single standalone forward with the ref image's VAE latent as
    the "image" (hidden_states), text conditioning a generic caption, and
    timestep=1.0 (Lumina convention: t=1 is clean image, t=0 is noise -- see
    pipeline_lumina2.py L726-727). At each patched layer we store the ref
    image-token K (pre-RoPE, post qk-norm -- "position-free identity
    dictionary", matching 30_kv_inject.py's default refpos="strip") and V
    (kv_heads space, pre-GQA-repeat) from the last `ref_len` tokens of that
    forward's joint [cap; image] sequence.
  * inject: during real generation, concat the stored ref K (pre-RoPE) with
    the freshly-computed post-RoPE target K along the sequence axis BEFORE
    the GQA repeat (so repeat handles ref and target uniformly), and concat
    V with value scaled by injection weight `w` (identity strength knob --
    value has no norm, so scaling it is not neutralized the way scaling a
    normed key would be).
  * head_gate: additive -inf bias on the ref-column block of the SDPA mask,
    per QUERY head (post-GQA-repeat, 24-head space), for all heads not in
    the kept top/bottom/random subset. GQA repeat only tiles the head axis,
    it does not reorder the sequence axis, so this masks correctly
    regardless of which kv-head a query head was expanded from.

Run:
  source /home/ubuntu/chengyanli/items/cherry/env.sh
  $PY probe/80_lumina_inject.py --ref 2 --prompt "..." --w 4 --seed 0 \
      --head_gate off  --out /home/ubuntu/chengyanli/items/cherry/out/lumina/u2_full.png
  $PY probe/80_lumina_inject.py --ref 2 --prompt "..." --w 4 --seed 0 \
      --head_gate top --head_keep 6 --out .../u2_top6.png
"""
import os, math, argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL = "models/lumina2"
REFDIR = "/mnt/local/smoke_anime/anime_260623/img"
OUT = "/home/ubuntu/chengyanli/items/cherry/out/lumina"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

from diffusers import Lumina2Pipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.lumina2.pipeline_lumina2 import calculate_shift, retrieve_timesteps

# ------------------------- injection state -------------------------
INJ = {
    "mode": "off",            # off | capture | inject
    "w": 4.0,                 # value-scale identity-strength knob
    "ref_len": 0, "tgt_len": 0,
    "band": set(range(26)),   # which of the 26 main layers get ref-KV (default: all)
    "refpos": "strip",        # strip = pre-RoPE (position-free) ref key, matches 30's default
    "head_gate": "off",       # off | top | bottom | random
    "head_keep": None,
    "head_seed": 0,
    # --- decisive cross-model investigation instrumentation (default OFF, no
    # effect on existing behavior unless explicitly set) ---
    "measure": False,         # log per-layer target-image-query -> ref-key softmax attention mass
    "ref_bias": 0.0,          # additive pre-softmax log-bias on ref-key columns, ALL heads uniformly
                              # (causal knob: force attention onto the position-free ref-key slot,
                              # decoupled from qk-norm/RoPE content -- tests whether attention
                              # ROUTING is the sole barrier to identity transfer on Lumina)
    "cur_step": -1,
}
STORE = {}
ATTN_LOG = {}  # {layer_id: [{"step":.., "ref_mass":.., "k_ref_norm":.., "k_ctx_norm":..}, ...]}


def dump_attn_log(path):
    import json
    summary = {}
    for lid, recs in sorted(ATTN_LOG.items()):
        if not recs:
            continue
        n = len(recs)
        summary[str(lid)] = {
            "n_steps": n,
            "mean_ref_mass": sum(r["ref_mass"] for r in recs) / n,
            "mean_k_ref_norm": sum(r["k_ref_norm"] for r in recs) / n,
            "mean_k_ctx_norm": sum(r["k_ctx_norm"] for r in recs) / n,
        }
    with open(path, "w") as f:
        json.dump({"per_layer": summary, "raw": ATTN_LOG}, f, indent=1)
    print(f"[measure] dumped attn log -> {path}", flush=True)
    for lid, s in summary.items():
        print(f"  L{lid:>2}  ref_mass={s['mean_ref_mass']:.5f}  "
              f"k_ref_norm={s['mean_k_ref_norm']:.3f}  k_ctx_norm={s['mean_k_ctx_norm']:.3f}", flush=True)


class InjectProcessor:
    """Drop-in replacement for Lumina2AttnProcessor2_0 with capture/inject/head-gate."""

    def __init__(self, lid):
        self.lid = lid

    def __call__(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask=None, image_rotary_emb=None, base_sequence_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        k_pre = key  # post qk-norm, PRE-RoPE -- "strip" position-free identity dict

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key_rope = apply_rotary_emb(key, image_rotary_emb, use_real=False)
        else:
            key_rope = key
        query, key_rope = query.to(dtype), key_rope.to(dtype)

        active = self.lid in INJ["band"]
        inj_here = False
        n_ref = 0
        if active and INJ["mode"] == "capture":
            ref_len = INJ["ref_len"]
            if INJ["refpos"] == "strip":
                sk = k_pre[:, -ref_len:].detach().clone()
            elif INJ["refpos"] == "lowfreq" and image_rotary_emb is not None:
                # coarse-position-only key: keep the true per-token rotation for
                # the lowest `lowfreq_k` row/col RoPE freq pairs (rough layout --
                # "upper-left-ish"), null the rest to identity rotation (exact
                # pixel-locking removed). axis layout in image_rotary_emb is
                # [cap-marker(third), row(third), col(third)] concatenated.
                kp = INJ.get("lowfreq_k", 4)
                lf = image_rotary_emb[:, -ref_len:].clone()
                third = lf.shape[-1] // 3
                lf[..., third + kp: 2 * third] = 1.0 + 0j
                lf[..., 2 * third + kp: 3 * third] = 1.0 + 0j
                sk = apply_rotary_emb(k_pre[:, -ref_len:], lf, use_real=False).detach().clone()
            else:  # "rope": true post-RoPE key, ref grid at its real (row,col) position
                sk = key_rope[:, -ref_len:].detach().clone()
            sv = value[:, -ref_len:].detach().clone()
            STORE[self.lid] = (sk.to(dtype), sv.to(value.dtype))
            key_use, value_use = key_rope, value
        elif active and INJ["mode"] == "inject" and self.lid in STORE:
            rk, rv = STORE[self.lid]
            rk = rk.to(key_rope.dtype)
            rv = rv.to(value.dtype)
            key_use = torch.cat([key_rope, rk], dim=1)
            value_use = torch.cat([value, INJ["w"] * rv], dim=1)
            n_ref = rk.shape[1]
            inj_here = True
        else:
            key_use, value_use = key_rope, value

        n_rep = attn.heads // kv_heads
        if n_rep > 1:
            key_use = key_use.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value_use = value_use.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        qh = query.transpose(1, 2)     # (B,H,Sq,d)
        kh = key_use.transpose(1, 2)   # (B,H,Sk,d)
        vh = value_use.transpose(1, 2)

        attn_bias = None
        if attention_mask is not None:
            am = attention_mask.bool().view(batch_size, 1, 1, -1)
            if inj_here:
                pad = torch.ones(batch_size, 1, 1, n_ref, dtype=torch.bool, device=am.device)
                am = torch.cat([am, pad], dim=-1)
            attn_bias = torch.zeros(am.shape, device=qh.device, dtype=qh.dtype).masked_fill(~am, float("-inf"))

        # head-level ref-injection gate: inject ref K/V only into a subset of the
        # 24 QUERY heads (post-GQA-repeat space); mask ref columns for the rest.
        if inj_here and INJ.get("head_gate", "off") != "off":
            B, H, Ktot, d = kh.shape
            ref0 = Ktot - n_ref
            T = INJ["tgt_len"]
            q_img = qh[:, :, -T:, :]              # target IMAGE-token queries only
            k_ref = kh[:, :, ref0:, :]
            a_head = (torch.matmul(q_img, k_ref.transpose(-1, -2)) / (d ** 0.5)
                      ).float().mean(dim=(0, 2, 3))   # [H] mean ref-affinity per head
            keep = min(INJ.get("head_keep") or H, H)
            if INJ["head_gate"] == "random":
                g = torch.Generator(device=a_head.device).manual_seed(int(INJ.get("head_seed", 0)) + self.lid)
                order = torch.randperm(H, generator=g, device=a_head.device)
            else:
                order = torch.argsort(a_head, descending=(INJ["head_gate"] == "top"))
            drop = order[keep:].tolist()
            if drop:
                hbias = torch.zeros((B, H, 1, Ktot), device=qh.device, dtype=qh.dtype)
                for h in drop:
                    hbias[:, h, :, ref0:] = float("-inf")
                attn_bias = hbias if attn_bias is None else (attn_bias + hbias)

        # --- causal knob: force attention onto the ref-key columns via a uniform
        # additive pre-softmax bias (all heads, all query rows) -- decoupled from
        # qk-norm/RoPE content. Tests whether attention ROUTING is the sole
        # barrier to identity transfer (default ref_bias=0.0 -> no-op).
        if inj_here and INJ.get("ref_bias", 0.0) != 0.0:
            B, H, Ktot, _ = kh.shape
            ref0 = Ktot - n_ref
            rb = torch.zeros((1, 1, 1, Ktot), device=qh.device, dtype=qh.dtype)
            rb[..., ref0:] = INJ["ref_bias"]
            attn_bias = rb if attn_bias is None else (attn_bias + rb)

        # --- measurement instrumentation (default off, zero effect on generation):
        # exact softmax attention mass that target IMAGE-token queries assign to
        # the injected ref-key columns, per layer. Uses the SAME scale/bias the
        # real SDPA call below will use, so it reflects the true routing.
        if inj_here and INJ.get("measure", False):
            B, H, Ktot, d = kh.shape
            ref0 = Ktot - n_ref
            T = INJ["tgt_len"]
            q_img = qh[:, :, -T:, :].float()
            scores = torch.matmul(q_img, kh.float().transpose(-1, -2)) * softmax_scale
            if attn_bias is not None:
                scores = scores + attn_bias.float()
            probs = torch.softmax(scores, dim=-1)
            ref_mass = probs[..., ref0:].sum(dim=-1).mean().item()
            k_ref_norm = kh[:, :, ref0:, :].float().norm(dim=-1).mean().item()
            k_ctx_norm = kh[:, :, ref0 - T:ref0, :].float().norm(dim=-1).mean().item()
            # content-selectivity: renormalize each query's distribution over ONLY
            # the ref columns, measure how peaked it is. Low normalized entropy /
            # high peak = content-addressing (query prefers specific ref tokens);
            # entropy -> 1 = diffuse (attends all ref tokens equally = color-bleed).
            rp = probs[..., ref0:].float()
            rp = rp / (rp.sum(dim=-1, keepdim=True) + 1e-9)
            R = rp.shape[-1]
            ent = -(rp * (rp + 1e-9).log()).sum(dim=-1) / (torch.log(torch.tensor(float(R))))
            peak = rp.max(dim=-1).values
            ATTN_LOG.setdefault(self.lid, []).append({
                "step": INJ.get("cur_step", -1), "ref_mass": ref_mass,
                "k_ref_norm": k_ref_norm, "k_ctx_norm": k_ctx_norm,
                "ref_entropy_norm": ent.mean().item(), "ref_peak": peak.mean().item(),
            })
            del scores, probs, rp

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_bias, scale=softmax_scale)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ------------------------- load frozen base -------------------------
print("loading Lumina-Image-2.0 (frozen, bf16) ...", flush=True)
pipe = Lumina2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(DEV)
tf = pipe.transformer
DT = tf.dtype
for i, blk in enumerate(tf.layers):
    blk.attn.set_processor(InjectProcessor(i))
print(f"installed InjectProcessor on {len(tf.layers)} layers", flush=True)

NEG = ""  # Lumina2 default negative prompt


@torch.no_grad()
def enc_prompt(text, max_len=256):
    wrapped = pipe.system_prompt + " <Prompt Start> " + text
    return pipe._get_gemma_prompt_embeds(prompt=[wrapped], device=DEV, max_sequence_length=max_len)


@torch.no_grad()
def enc_raw(text, max_len=256):
    # negative_prompt path in Lumina2Pipeline.encode_prompt does NOT get the
    # system_prompt/<Prompt Start> wrapping -- replicate exactly.
    return pipe._get_gemma_prompt_embeds(prompt=[text], device=DEV, max_sequence_length=max_len)


@torch.no_grad()
def vae_encode_square(path, size):
    import numpy as np
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
    arr = arr.unsqueeze(0).to(DEV, torch.bfloat16)
    lat = pipe.vae.encode(arr).latent_dist.mode()
    lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return lat.to(DT)  # (1,16,H/8,W/8)


@torch.no_grad()
def prep_ref(ref_path, refsize, band, cap_text="anime illustration of a character, simple background"):
    """One-time (VAE encode + caption embed) prep for a ref capture. Returns
    (ref_lat, cap_embeds, cap_mask) to be fed into capture_step() either once
    (t=1.0 "clean" convention) or every denoising step (timestep-matched)."""
    ref_lat = vae_encode_square(ref_path, refsize)
    cap_embeds, cap_mask = enc_prompt(cap_text)
    INJ["band"] = band
    INJ["ref_len"] = (refsize // 16) ** 2
    return ref_lat, cap_embeds.to(DT), cap_mask


@torch.no_grad()
def capture_step(ref_lat, cap_embeds, cap_mask, t):
    """Run one capture forward at timestep `t`, refilling STORE for INJ['band']."""
    STORE.clear()
    INJ["mode"] = "capture"
    tf(hidden_states=ref_lat, timestep=t, encoder_hidden_states=cap_embeds,
       encoder_attention_mask=cap_mask, return_dict=False)
    INJ["mode"] = "off"
    assert len(STORE) == len(INJ["band"]), f"capture stored {len(STORE)} layers, expected {len(INJ['band'])}"


@torch.no_grad()
def capture_ref(ref_path, refsize, band, cap_text="anime illustration of a character, simple background"):
    """Legacy one-shot capture at t=1.0 (Lumina's 'clean image' convention)."""
    ref_lat, cap_embeds, cap_mask = prep_ref(ref_path, refsize, band, cap_text)
    t_clean = torch.tensor([1.0], device=DEV, dtype=torch.float32)
    capture_step(ref_lat, cap_embeds, cap_mask, t_clean)


@torch.no_grad()
def generate(prompt, neg=NEG, w="off", size=1024, refsize=512, steps=30, cfg=4.0, seed=0,
             ref_path=None, band=None, head_gate="off", head_keep=None, head_seed=0,
             refpos="strip", lowfreq_k=4, max_sequence_length=256, per_step=False,
             measure=False, ref_bias=0.0):
    band = band if band is not None else set(range(26))
    inject_on = (w != "off") and (ref_path is not None)
    INJ["head_gate"] = head_gate if inject_on else "off"
    INJ["head_keep"] = head_keep
    INJ["head_seed"] = head_seed
    INJ["band"] = band
    INJ["refpos"] = refpos
    INJ["lowfreq_k"] = lowfreq_k
    INJ["tgt_len"] = (size // 16) ** 2
    INJ["measure"] = bool(measure) and inject_on
    INJ["ref_bias"] = float(ref_bias) if inject_on else 0.0
    if measure:
        ATTN_LOG.clear()
    ref_prep = None
    if inject_on:
        INJ["w"] = float(w)
        if per_step:
            ref_prep = prep_ref(ref_path, refsize, band)
        else:
            capture_ref(ref_path, refsize, band)

    gen = torch.Generator(DEV).manual_seed(seed)
    cap_embeds, cap_mask = enc_prompt(prompt, max_sequence_length)
    neg_embeds, neg_mask = enc_raw(neg, max_sequence_length)
    cap_embeds, neg_embeds = cap_embeds.to(DT), neg_embeds.to(DT)

    latents = pipe.prepare_latents(1, tf.config.in_channels, size, size, DT, DEV, gen)

    sigmas_np = __import__("numpy").linspace(1.0, 1.0 / steps, steps)
    image_seq_len = latents.shape[-1] * latents.shape[-2] // (tf.config.patch_size ** 2)
    mu = calculate_shift(image_seq_len,
                          pipe.scheduler.config.get("base_image_seq_len", 256),
                          pipe.scheduler.config.get("max_image_seq_len", 4096),
                          pipe.scheduler.config.get("base_shift", 0.5),
                          pipe.scheduler.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, DEV, sigmas=sigmas_np, mu=mu)

    for i, t in enumerate(timesteps):
        current_t = (1 - t / pipe.scheduler.config.num_train_timesteps).expand(latents.shape[0])
        if inject_on and per_step:
            # timestep-matched ref capture: re-run the ref forward at THIS
            # step's temb (ref pixels stay clean; only the temb/AdaLN
            # modulation tracks the target's current noise level), mirroring
            # Z-Image's per-step ref-KV recompute as faithfully as Lumina2's
            # (no per-block noise-mask channel) architecture allows.
            capture_step(ref_prep[0], ref_prep[1], ref_prep[2], current_t)
        INJ["cur_step"] = i
        INJ["mode"] = "inject" if inject_on else "off"
        cond = tf(hidden_states=latents, timestep=current_t, encoder_hidden_states=cap_embeds,
                  encoder_attention_mask=cap_mask, return_dict=False)[0]
        INJ["mode"] = "off"  # uncond branch never sees injected ref K/V
        uncond = tf(hidden_states=latents, timestep=current_t, encoder_hidden_states=neg_embeds,
                    encoder_attention_mask=neg_mask, return_dict=False)[0]
        noise_pred = uncond + cfg * (cond - uncond)
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_pred = noise_pred * (cond_norm / noise_norm)
        noise_pred = -noise_pred
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    lat = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def resolve_ref(ref):
    if os.path.exists(ref):
        return ref
    return f"{REFDIR}/u{ref}_ref.png"


def label(im, txt, cell):
    im = im.resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, min(cell, 320), 30], fill="black")
    d.text((6, 7), txt, fill="white")
    return im


def parse_band(s):
    s = s.strip()
    if "-" in s:
        lo, hi = s.split("-")
        return set(range(int(lo), int(hi) + 1))
    return {int(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="uid (e.g. 2) or a path to a *_ref.png")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--neg", default=NEG)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--refsize", type=int, default=512)
    ap.add_argument("--band", default="0-25")
    ap.add_argument("--head_gate", default="off", choices=["off", "top", "bottom", "random"])
    ap.add_argument("--head_keep", type=int, default=None)
    ap.add_argument("--head_seed", type=int, default=0)
    ap.add_argument("--refpos", default="strip", choices=["strip", "rope", "lowfreq"],
                     help="strip=pre-RoPE position-free key (Z-Image default; did NOT transfer "
                          "identity in our port); rope=true post-RoPE key (transfers identity only "
                          "as position-locked copy if refsize==size); lowfreq=coarse-position-only "
                          "key (partial RoPE surgery, the mode aimed at genuine content-addressed "
                          "identity transfer under a NEW pose)")
    ap.add_argument("--lowfreq_k", type=int, default=4)
    ap.add_argument("--out", default=None, help="output path for the combined montage")
    ap.add_argument("--tag", default="cmp")
    ap.add_argument("--skip_off", action="store_true", help="skip the no-injection baseline panel")
    ap.add_argument("--measure", action="store_true",
                     help="log per-layer target-image-query -> ref-key softmax attention mass "
                          "during the 'full inject' panel; dumps to --measure_out")
    ap.add_argument("--ref_bias", type=float, default=0.0,
                     help="causal knob: additive pre-softmax log-bias on ref-key columns (all heads); "
                          "0.0 = no-op (default). Forces attention onto ref-key slot.")
    ap.add_argument("--measure_out", default=None, help="json path for --measure dump")
    a = ap.parse_args()

    ref_path = resolve_ref(a.ref)
    band = parse_band(a.band)
    uid_tag = os.path.basename(ref_path).replace("_ref.png", "")

    cell = 640
    panels = []

    rimg = Image.open(ref_path).convert("RGB")
    panels.append(("ref", rimg))

    if not a.skip_off:
        print(f"[gen] {uid_tag} off (no injection)", flush=True)
        im_off = generate(a.prompt, neg=a.neg, w="off", size=a.size, refsize=a.refsize,
                           steps=a.steps, cfg=a.cfg, seed=a.seed)
        im_off.save(f"{OUT}/{uid_tag}_{a.tag}_off.png")
        panels.append(("off (no inject)", im_off))

    print(f"[gen] {uid_tag} full inject w={a.w} band={sorted(band)[:3]}..{sorted(band)[-1:]} "
          f"n={len(band)} refpos={a.refpos}", flush=True)
    im_full = generate(a.prompt, neg=a.neg, w=a.w, size=a.size, refsize=a.refsize,
                        steps=a.steps, cfg=a.cfg, seed=a.seed, ref_path=ref_path, band=band,
                        head_gate="off", refpos=a.refpos, lowfreq_k=a.lowfreq_k,
                        measure=a.measure, ref_bias=a.ref_bias)
    im_full.save(f"{OUT}/{uid_tag}_{a.tag}_full_w{a.w:g}.png")
    panels.append((f"full inject w{a.w:g}", im_full))
    if a.measure:
        mo = a.measure_out or f"{OUT}/attn_quant_{uid_tag}_{a.tag}.json"
        dump_attn_log(mo)

    if a.head_gate != "off" or a.head_keep is not None:
        hg = a.head_gate if a.head_gate != "off" else "top"
        hk = a.head_keep if a.head_keep is not None else 12
        print(f"[gen] {uid_tag} head_gate={hg} head_keep={hk} w={a.w}", flush=True)
        im_gate = generate(a.prompt, neg=a.neg, w=a.w, size=a.size, refsize=a.refsize,
                            steps=a.steps, cfg=a.cfg, seed=a.seed, ref_path=ref_path, band=band,
                            head_gate=hg, head_keep=hk, head_seed=a.head_seed,
                            refpos=a.refpos, lowfreq_k=a.lowfreq_k)
        im_gate.save(f"{OUT}/{uid_tag}_{a.tag}_{hg}{hk}_w{a.w:g}.png")
        panels.append((f"{hg}{hk} w{a.w:g}", im_gate))

    canvas = Image.new("RGB", (cell * len(panels), cell), "white")
    for c, (txt, im) in enumerate(panels):
        canvas.paste(label(im, txt, cell), (c * cell, 0))
    out_path = a.out or f"{OUT}/montage_{uid_tag}_{a.tag}.png"
    canvas.save(out_path)
    print("saved montage:", out_path, flush=True)


if __name__ == "__main__":
    main()
