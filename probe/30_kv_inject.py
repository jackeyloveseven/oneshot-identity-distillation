#!/usr/bin/env python
"""
30_kv_inject.py — ref-KV injection on the PURE T2I Z-Image base (NO LoRA).

Reconstruction of the lost probe/29 (sparkid was wiped). Tests whether
concatenating a reference image's attention K/V into identity-band layers
activates identity on the frozen pure-T2I base WITHOUT any per-subject or
omni LoRA. Purest form of the project thesis.

CONSTRAINTS honored:
  * Never edits the diffusers library. InjectProcessor subclasses
    ZSingleStreamAttnProcessor and is installed via block.attention.set_processor().
  * All custom code lives here; big outputs -> /home/ubuntu/chengyanli/items/cherry/out.
Run: source /home/ubuntu/chengyanli/items/cherry/env.sh && $PY 30_kv_inject.py
"""
import os, argparse
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

MODEL = os.environ["ZBASE"]
REF   = os.environ["REF"]
OUT   = "/home/ubuntu/chengyanli/items/cherry/out"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

from diffusers import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor, RopeEmbedder
from diffusers.pipelines.z_image.pipeline_z_image_omni import calculate_shift, retrieve_timesteps

# ------------------------- injection state -------------------------
INJ = {"mode": "off", "w": 2.0, "ref_len": 1024, "tgt_len": 4096,
       "band": set(range(16, 26)), "refpos": "strip", "head_gate": "off", "head_keep": None, "head_seed": 0,
       # E1 spatial routing: gate injected ref-value per target-query so only
       # subject/foreground queries read the reference (kills corner ghost).
       "route": "off", "keep_pct": 40.0, "temp": 3.0, "gate_strength": 1.0,
       # E3: normalize injected ref-value per-token to target value norm
       # (inject identity DIRECTION not raw magnitude -> targets color oversaturation)
       "vnorm": False,
       # 1b geometric addressing: refpos="center" only for the first center_cut
       # fraction of steps, then fall back to strip (early-geometry-then-content
       # schedule, mirrors E3's taper/early logic but for POSITION not weight).
       "step_frac": 0.0, "center_cut": 1.0, "lowfreq_k": 4,
       # --- decisive cross-model investigation instrumentation (default OFF,
       # zero effect on existing behavior unless explicitly set) ---
       "measure": False,   # log per-layer target-image-query -> ref-key softmax attention mass
       "ref_bias": 0.0,    # additive pre-softmax log-bias on ref-key columns, ALL heads uniformly
       # Read-only target-query -> selected text-token affinity capture. This is
       # used by score-distillation probes as a reference-independent spatial
       # support signal; default OFF and therefore a no-op for all old runs.
       "support_capture": False, "support_token_indices": (),
       "support_layers": set(), "support_maps": [],
       # single-model (Z-Image only) theta-gradient causal isolation: refpos="thetasweep"
       # rotates the ref key with a RoPE computed at a SWEEPABLE theta (default 256 =
       # Z's own native theta, reproduces "center" positioning exactly at baseline) instead
       # of the model's baked-in theta. Default off (refpos stays "strip" unless set).
       "ref_theta": 256.0,
       "cur_step": -1}
STORE = {}
ATTN_LOG = {}  # {layer_id: [{"step":.., "ref_mass":.., "k_ref_norm":.., "k_ctx_norm":..}, ...]}
_THETA_CACHE = {}  # {theta: [freqs_cis_F, freqs_cis_H, freqs_cis_W]} memoized custom-theta RoPE tables


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

class InjectProcessor(ZSingleStreamAttnProcessor):
    def __init__(self, lid):
        super().__init__()
        self.lid = lid

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, freqs_cis=None, **kwargs):
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)
        k_pre = k  # post-norm, PRE-RoPE (position-free identity dictionary)

        def rope(x, fc):
            with torch.amp.autocast("cuda", enabled=False):
                xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                fc = fc.unsqueeze(2)
                return torch.view_as_real(xc * fc).flatten(3).type_as(x)
        if freqs_cis is not None:
            q = rope(q, freqs_cis); k = rope(k, freqs_cis)
        dtype = q.dtype; q, k = q.to(dtype), k.to(dtype)
        base_sequence_length = k.shape[1]

        active = self.lid in INJ["band"]
        inj_here = False; inj_Sb = 0; inj_R = 0
        if active and INJ["mode"] == "capture":
            S = k.shape[1]
            rs = slice(S - INJ["tgt_len"] - INJ["ref_len"], S - INJ["tgt_len"])
            if INJ["refpos"] == "strip" or (INJ["refpos"] == "center" and INJ["step_frac"] > INJ["center_cut"]):
                sk = k_pre[:, rs].detach().clone()
            elif INJ["refpos"] in ("center", "lowfreq"):
                # 1b geometric addressing: keep k_pre's true F-axis RoPE component
                # (sliced from the real freqs_cis passed in for this forward), but
                # replace the H,W component with a freshly-computed grid CENTERED
                # inside the target's HxW token grid (ref grid starts at (0,0) by
                # construction -> top-left copy-paste ghost, confirmed empirically
                # via refpos="keep"; centering aligns ref geometry with where the
                # subject actually sits in a centered-portrait composition).
                # "lowfreq": additionally null out all but the lowest lowfreq_k
                # H/W rotation pairs (low freq = coarse/slow-varying position,
                # high freq = fine/exact pixel binding) -> conveys rough layout
                # ("eye above nose, left of center") without exact-position lock
                # that causes center's copy-paste-style instability.
                re = tf.rope_embedder
                dF = re.axes_dims[0] // 2
                ref_side = int(round(INJ["ref_len"] ** 0.5))
                tgt_side = int(round(INJ["tgt_len"] ** 0.5))
                off = (tgt_side - ref_side) // 2
                idx = torch.arange(INJ["ref_len"], device=k.device)
                hh = idx // ref_side + off
                ww = idx % ref_side + off
                fH = re.freqs_cis[1][hh].clone()  # [ref_len, dimH/2]
                fW = re.freqs_cis[2][ww].clone()  # [ref_len, dimW/2]
                if INJ["refpos"] == "lowfreq":
                    kp = INJ["lowfreq_k"]
                    fH[:, kp:] = 1.0 + 0j  # identity rotation = position-free for these channels
                    fW[:, kp:] = 1.0 + 0j
                fF = freqs_cis[:, rs][..., :dF]  # real F-part, [B, ref_len, dimF/2]
                fc_new = torch.cat([fF, fH.unsqueeze(0).expand(fF.shape[0], -1, -1).to(fF.dtype),
                                    fW.unsqueeze(0).expand(fF.shape[0], -1, -1).to(fF.dtype)], dim=-1)
                sk = rope(k_pre[:, rs], fc_new)
            elif INJ["refpos"] == "thetasweep":
                # single-model (Z-only) causal isolation of "phase geometry" as a
                # determinant: SAME center-aligned ref-grid position as "center" above
                # (so at ref_theta==tf.rope_embedder.theta this reproduces "center"
                # exactly), but H/W RoPE freqs are computed at a SWEEPABLE theta
                # instead of the model's baked-in one -> lets us dial Z's own RoPE
                # geometry toward Lumina's operating point (theta=10000) without
                # crossing model/GQA/depth/text-encoder confounds.
                theta_c = float(INJ.get("ref_theta", 256.0))
                cache = _THETA_CACHE.get(theta_c)
                if cache is None:
                    cache = RopeEmbedder.precompute_freqs_cis(
                        list(tf.rope_embedder.axes_dims), list(tf.rope_embedder.axes_lens), theta=theta_c)
                    cache = [c.to(k.device) for c in cache]
                    _THETA_CACHE[theta_c] = cache
                dF = tf.rope_embedder.axes_dims[0] // 2
                ref_side = int(round(INJ["ref_len"] ** 0.5))
                tgt_side = int(round(INJ["tgt_len"] ** 0.5))
                off = (tgt_side - ref_side) // 2
                idx = torch.arange(INJ["ref_len"], device=k.device)
                hh = idx // ref_side + off
                ww = idx % ref_side + off
                fH = cache[1][hh].clone()
                fW = cache[2][ww].clone()
                fF = freqs_cis[:, rs][..., :dF]
                fc_new = torch.cat([fF, fH.unsqueeze(0).expand(fF.shape[0], -1, -1).to(fF.dtype),
                                    fW.unsqueeze(0).expand(fF.shape[0], -1, -1).to(fF.dtype)], dim=-1)
                sk = rope(k_pre[:, rs], fc_new)
            else:  # "keep": raw post-RoPE key, ref grid at its true (0,0)-origin position
                sk = k[:, rs].detach().clone()
            STORE[self.lid] = (sk.to(dtype), v[:, rs].detach().clone())
        elif active and INJ["mode"] == "inject" and self.lid in STORE:
            rk, rv = STORE[self.lid]
            inj_Sb = k.shape[1]; inj_R = rk.shape[1]; inj_here = True  # base seq [cap,tgt]
            rv = rv.to(v.dtype)
            # lit-scan idea #2 (attention-sink / outlier suppression): zero the VALUE
            # of the highest-norm ref tokens (sink/watermark candidates that leak into
            # the fixed corner artifacts) while keeping ALL normal identity tokens at
            # full strength -- distinct from "weaker injection" (which killed the
            # head-gate mirage). clip_mode="random" drops the SAME COUNT of random
            # tokens as a matched-count control (is it the norm-SELECTION or just the count?).
            if INJ.get("clip_ref", 0) > 0:
                rn = rk.float().norm(dim=-1)                                   # [B,R] per-token key norm
                thr = rn.mean(dim=1, keepdim=True) + INJ["clip_ref"] * rn.std(dim=1, keepdim=True)
                drop = (rn > thr)                                             # outlier-high-norm tokens
                if INJ.get("clip_mode", "norm") == "random":
                    ggen = torch.Generator(device=rn.device).manual_seed(int(INJ.get("head_seed", 0)) + self.lid)
                    ndrop = int(drop[0].sum().item())
                    drop = torch.zeros_like(drop)
                    if ndrop > 0:
                        idx = torch.randperm(rn.shape[1], generator=ggen, device=rn.device)[:ndrop]
                        drop[:, idx] = True
                rv = rv * (~drop).to(rv.dtype).unsqueeze(-1)                  # zero dropped tokens' values
            if INJ["vnorm"]:
                # match injected value per-token norm to target-block value norm
                # -> inject identity direction, not raw magnitude (color statistics)
                vt = v[:, k.shape[1] - INJ["tgt_len"]:]      # target block [B,T,H,d]
                vt_norm = vt.float().norm(dim=-1, keepdim=True).mean()   # scalar
                rv = rv / (rv.float().norm(dim=-1, keepdim=True) + 1e-6) * vt_norm
                rv = rv.to(v.dtype)
            k = torch.cat([k, rk.to(k.dtype)], dim=1)
            v = torch.cat([v, INJ["w"] * rv], dim=1)

        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        if (
            INJ.get("support_capture", False)
            and self.lid in INJ.get("support_layers", set())
            and INJ.get("support_token_indices")
        ):
            target_length = INJ["tgt_len"]
            target_start = base_sequence_length - target_length
            token_indices = [
                int(index)
                for index in INJ["support_token_indices"]
                if 0 <= int(index) < target_start
            ]
            if token_indices:
                q_target = qh[:, :, target_start:base_sequence_length, :].float()
                k_subject = kh[:, :, token_indices, :].float()
                affinity = torch.matmul(
                    q_target,
                    k_subject.transpose(-1, -2),
                ) / (qh.shape[-1] ** 0.5)
                affinity = affinity.mean(dim=(1, 3))
                affinity = (
                    affinity - affinity.mean(dim=1, keepdim=True)
                ) / affinity.std(dim=1, keepdim=True).clamp_min(1e-6)
                INJ["support_maps"].append(affinity.detach())

        # E1: per-query ref-gate as additive attention bias on ref-slot columns
        if inj_here and INJ["route"] != "off":
            B, H, Ktot, d = kh.shape
            T = INJ["tgt_len"]; tgt0 = inj_Sb - T; ref0 = inj_Sb
            q_tgt = qh[:, :, tgt0:inj_Sb, :]              # [B,H,T,d]
            k_ref = kh[:, :, ref0:ref0 + inj_R, :]        # [B,H,R,d]
            aff = torch.matmul(q_tgt, k_ref.transpose(-1, -2)) / (d ** 0.5)  # [B,H,T,R]
            a = aff.mean(dim=(1, 3)).float()              # [B,T]
            if INJ["route"] == "soft":
                med = a.median(dim=1, keepdim=True).values
                std = a.std(dim=1, keepdim=True) + 1e-6
                g = torch.sigmoid((a - med) / std * INJ["temp"])
            else:  # pct: keep only top keep_pct% queries by ref-affinity
                kth = torch.quantile(a, 1.0 - INJ["keep_pct"] / 100.0, dim=1, keepdim=True)
                g = (a >= kth).float()
            logg = (torch.log(g + 1e-6) * INJ["gate_strength"]).to(qh.dtype)  # [B,T]
            bias = torch.zeros((B, 1, qh.shape[2], Ktot), device=qh.device, dtype=qh.dtype)
            bias[:, :, tgt0:inj_Sb, ref0:ref0 + inj_R] = logg[:, None, :, None]
            if attention_mask is None:
                attention_mask = bias
            else:
                am = attention_mask
                if am.dtype == torch.bool:
                    am = torch.zeros_like(bias).masked_fill(~am, float("-inf"))
                attention_mask = am + bias

        # head-level ref-injection gate (lit-scan idea #1): inject ref K/V only for
        # the heads whose target queries most (top) / least (bottom) attend to the
        # ref slots; mask ref columns for the rest via a per-head additive bias
        # ([B,H,1,Ktot], broadcast over query rows -- cheap; assumes route=off).
        # Tests whether identity lives in a head subset and whether restricting
        # injection to it cuts the collage/PiP defect without losing identity.
        if inj_here and INJ.get("head_gate", "off") != "off":
            B, H, Ktot, d = kh.shape
            T = INJ["tgt_len"]; tgt0 = inj_Sb - T; ref0 = inj_Sb
            q_tgt = qh[:, :, tgt0:inj_Sb, :]
            k_ref = kh[:, :, ref0:ref0 + inj_R, :]
            a_head = (torch.matmul(q_tgt, k_ref.transpose(-1, -2)) / (d ** 0.5)
                      ).float().mean(dim=(0, 2, 3))            # [H] mean ref-affinity per head
            keep = min(INJ.get("head_keep") or H, H)
            if INJ["head_gate"] == "random":
                # matched-count control: a seeded RANDOM subset of `keep` heads,
                # to test whether it's the SELECTION (top ref-affinity heads) that
                # carries identity, or merely the COUNT (weaker aggregate injection).
                g = torch.Generator(device=a_head.device).manual_seed(int(INJ.get("head_seed", 0)) + self.lid)
                order = torch.randperm(H, generator=g, device=a_head.device)
            else:
                order = torch.argsort(a_head, descending=(INJ["head_gate"] == "top"))
            drop = order[keep:].tolist()                       # heads to STOP injecting
            if drop:
                hbias = torch.zeros((B, H, 1, Ktot), device=qh.device, dtype=qh.dtype)
                for h in drop:
                    hbias[:, h, :, ref0:ref0 + inj_R] = float("-inf")
                attention_mask = hbias if attention_mask is None else (attention_mask + hbias)

        # --- causal knob (cross-model investigation): force attention onto the
        # ref-key columns via a uniform additive pre-softmax bias (all heads, all
        # query rows). Default ref_bias=0.0 -> no-op.
        if inj_here and INJ.get("ref_bias", 0.0) != 0.0:
            B, H, Ktot, d = kh.shape
            rb = torch.zeros((1, 1, 1, Ktot), device=qh.device, dtype=qh.dtype)
            rb[..., inj_Sb:inj_Sb + inj_R] = INJ["ref_bias"]
            attention_mask = rb if attention_mask is None else (attention_mask + rb)

        # --- measurement instrumentation (default off, zero effect on generation):
        # exact softmax attention mass that target IMAGE-token queries assign to
        # the injected ref-key columns, per layer.
        if inj_here and INJ.get("measure", False):
            T = INJ["tgt_len"]; tgt0 = inj_Sb - T
            q_img = qh[:, :, tgt0:inj_Sb, :].float()
            scale = 1.0 / (qh.shape[-1] ** 0.5)
            scores = torch.matmul(q_img, kh.float().transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask.float()
            probs = torch.softmax(scores, dim=-1)
            ref_mass = probs[..., inj_Sb:inj_Sb + inj_R].sum(dim=-1).mean().item()
            k_ref_norm = kh[:, :, inj_Sb:inj_Sb + inj_R, :].float().norm(dim=-1).mean().item()
            k_ctx_norm = kh[:, :, tgt0:inj_Sb, :].float().norm(dim=-1).mean().item()
            # content-selectivity (see 80_lumina_inject.py): normalized entropy of
            # each query's distribution over ONLY the ref columns. Low = content-
            # addressed (peaked), high (->1) = diffuse color-bleed.
            rp = probs[..., inj_Sb:inj_Sb + inj_R].float()
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

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attention_mask,
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).to(dtype)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1: out = attn.to_out[1](out)
        return out

# ------------------------- load pure base -------------------------
print("loading pure-T2I base (no LoRA) ...", flush=True)
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(DEV)
tf = pipe.transformer
DT = tf.dtype
for i, blk in enumerate(tf.layers):
    blk.attention.set_processor(InjectProcessor(i))
print(f"installed InjectProcessor on {len(tf.layers)} layers", flush=True)

PROMPT = ("solo, single character, centered close-up anime portrait, upper body, "
          "detailed face, clean anime style, simple plain white background")
NEG = ("lowres, bad anatomy, worst quality, blurry, cropped, off-center, "
       "duplicate, multiple people, two people, extra figure")

@torch.no_grad()
def enc(text):
    wrapped = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
    ti = pipe.tokenizer([wrapped], padding="max_length", max_length=512,
                        truncation=True, return_tensors="pt")
    ids = ti.input_ids.to(DEV); mask = ti.attention_mask.to(DEV).bool()
    h = pipe.text_encoder(input_ids=ids, attention_mask=mask,
                          output_hidden_states=True).hidden_states[-2]
    return h[0][mask[0]].to(DT)

@torch.no_grad()
def ref_latent(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
    arr = arr.unsqueeze(0).to(DEV)
    lat = (pipe.vae.encode(arr.to(torch.bfloat16)).latent_dist.mode()[0]
           - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return lat.unsqueeze(1).to(DT)  # (16,1,h,w)

@torch.no_grad()
def generate(uid, w, refpos="strip", size=1024, refsize=512, steps=50, cfg=5.0, seed=0,
             route="off", keep_pct=40.0, temp=3.0, gate_strength=1.0,
             sched_mode="const", sched_frac=0.6, vnorm=False,
             prompt=None, neg=None, cfg_id=0.0, center_cut=1.0, lowfreq_k=4,
             head_gate="off", head_keep=None, head_seed=0, clip_ref=0.0, clip_mode="norm",
             measure=False, ref_bias=0.0, ref_theta=256.0):
    INJ["refpos"] = refpos
    INJ["ref_theta"] = float(ref_theta)
    INJ["head_gate"] = head_gate; INJ["head_keep"] = head_keep; INJ["head_seed"] = head_seed
    INJ["clip_ref"] = clip_ref; INJ["clip_mode"] = clip_mode
    INJ["route"] = route; INJ["keep_pct"] = keep_pct
    INJ["temp"] = temp; INJ["gate_strength"] = gate_strength
    INJ["vnorm"] = vnorm
    INJ["center_cut"] = center_cut
    INJ["lowfreq_k"] = lowfreq_k
    INJ["ref_len"] = (refsize // 16) ** 2
    INJ["tgt_len"] = (size // 16) ** 2
    inject_on = (w != "off")
    INJ["measure"] = bool(measure) and inject_on
    INJ["ref_bias"] = float(ref_bias) if inject_on else 0.0
    if measure: ATTN_LOG.clear()
    if inject_on: INJ["w"] = float(w)
    gen = torch.Generator(DEV).manual_seed(seed)
    hlat = 2 * (size // 16)
    latents = torch.randn((1, 16, hlat, hlat), generator=gen, device=DEV, dtype=torch.float32)
    CAP_POS, CAP_NEG = enc(prompt if prompt is not None else PROMPT), enc(neg if neg is not None else NEG)
    # capture context stays on the generic PROMPT even when cond prompt differs
    # (isolates the text-channel effect from the KV-capture context; minefield #3)
    CAP_CAPTURE = enc(PROMPT)
    CAP_REF = enc("anime illustration of a character")
    ref = ref_latent(f"{REF}/u{uid}_ref.png", refsize) if inject_on else None

    sched = pipe.scheduler
    isl = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(isl,
                         sched.config.get("base_image_seq_len", 256),
                         sched.config.get("max_image_seq_len", 4096),
                         sched.config.get("base_shift", 0.5),
                         sched.config.get("max_shift", 1.15))
    sched.sigma_min = 0.0
    try:
        timesteps, _ = retrieve_timesteps(sched, steps, DEV, sigmas=None, mu=mu)
    except Exception:
        timesteps, _ = retrieve_timesteps(sched, steps, DEV, sigmas=None)

    def fwd(cap, x_imgs, nmask):
        return tf(x=[x_imgs], t=ts, cap_feats=[cap], siglip_feats=[None],
                  image_noise_mask=[nmask], return_dict=False)[0][0].float()

    # E3: per-step injection-weight schedule w_t (indexed by loop step, not timestep)
    N = len(timesteps); base_w = float(w) if inject_on else 0.0
    if sched_mode == "early":
        cut = max(1, int(round(sched_frac * N)))
        w_sched = [base_w if i < cut else 0.0 for i in range(N)]
    elif sched_mode == "taper":
        w_sched = [base_w * (1.0 - i / max(1, N - 1)) for i in range(N)]
    elif sched_mode == "late":
        # complement of "early": inject 0 for the first (1-sched_frac) of steps so
        # the frozen base locks a clean single-subject composition, THEN stamp
        # identity in the low-noise detail window. Tests whether the collage/PiP
        # defect is localized in DENOISING TIME (early composition pass) even
        # though identity is distributed across layers/heads (space).
        cut = max(0, int(round((1.0 - sched_frac) * N)))
        w_sched = [0.0 if i < cut else base_w for i in range(N)]
    elif sched_mode == "rampup":
        # smooth complement of taper: 0 at step0 -> base_w at last step.
        w_sched = [base_w * (i / max(1, N - 1)) for i in range(N)]
    else:  # const
        w_sched = [base_w] * N

    for i, t in enumerate(timesteps):
        ts = ((1000 - t.expand(latents.shape[0])) / 1000)
        x0 = latents.to(DT).unsqueeze(2)[0]  # (16,1,H,W)
        wt = w_sched[i]
        INJ["step_frac"] = i / max(1, N - 1)
        INJ["cur_step"] = i
        if inject_on and wt > 0:
            INJ["w"] = wt
            INJ["mode"] = "capture"
            tf(x=[[ref, x0]], t=ts, cap_feats=[[CAP_REF, CAP_CAPTURE]], siglip_feats=[None],
               image_noise_mask=[[0, 1]], return_dict=False)
            INJ["mode"] = "inject"
            cond = fwd([CAP_POS], [x0], [1])
            INJ["mode"] = "off"
            unc = fwd([CAP_NEG], [x0], [1])
            # 1c dual-CFG: identity-delta guidance against the no-injection branch
            if cfg_id > 0:
                cond_noinj = fwd([CAP_POS], [x0], [1])
                pred = cond + cfg * (cond - unc) + cfg_id * (cond - cond_noinj)
            else:
                pred = cond + cfg * (cond - unc)
        else:
            INJ["mode"] = "off"
            cond = fwd([CAP_POS], [x0], [1])
            unc = fwd([CAP_NEG], [x0], [1])
            pred = cond + cfg * (cond - unc)      # (16,1,H,W)
        noise_pred = (-pred).unsqueeze(0).squeeze(2)  # (1,16,H,W)
        latents = sched.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    lat = latents.to(pipe.vae.dtype)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]

def label(im, txt, cell):
    from PIL import ImageDraw
    im = im.resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 260, 34], fill="black")
    d.text((6, 8), txt, fill="white")
    return im

def run_e1(a, subs, ws):
    """E1: per subject, rows=w, cols=[ref, off, soft, pct] at matched w."""
    routes = [("off", {}), ("soft", {"temp": a.temp}),
              ("pct", {"keep_pct": a.keep_pct})]
    cell = 640
    stacks = []
    for uid in subs:
        rimg = Image.open(f"{REF}/u{uid}_ref.png").convert("RGB")
        grid = []
        for w in ws:
            row = [label(rimg, f"ref u{uid}", cell)]
            for rname, kw in routes:
                print(f"[e1] u{uid} w={w} route={rname} {kw}", flush=True)
                im = generate(uid, w, refpos=a.refpos, steps=a.steps, cfg=a.cfg,
                              route=rname, keep_pct=kw.get("keep_pct", a.keep_pct),
                              temp=kw.get("temp", a.temp), gate_strength=a.gate_strength)
                tagw = "off" if w == "off" else f"{w:g}"
                im.save(f"{OUT}/u{uid}_e1_{rname}_w{tagw}.png")
                row.append(label(im, f"{rname} w{tagw}", cell))
            grid.append(row)
        cols = 1 + len(routes); rows = len(ws)
        canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
        for r, row in enumerate(grid):
            for c, im in enumerate(row):
                canvas.paste(im, (c * cell, r * cell))
        canvas.save(f"{OUT}/montage_e1_{uid}.png")
        stacks.append(canvas)
    W = max(s.width for s in stacks); H = sum(s.height for s in stacks)
    combo = Image.new("RGB", (W, H), "white"); y = 0
    for s in stacks:
        combo.paste(s, (0, y)); y += s.height
    combo.save(f"{OUT}/montage_e1_route.png")
    print("saved:", f"{OUT}/montage_e1_route.png", flush=True)

def run_e3(a, subs):
    """E3: kill oversaturation via injection schedule + value-norm (route=off).
    Cells per subject: [ref, const/off@w2 (baseline)] then {const,early,taper}x{vnorm off,on}@w4."""
    cell = 640
    combos = [("const", False, 2.0)]  # low-w baseline
    for sm in ["const", "early", "taper"]:
        for vn in [False, True]:
            combos.append((sm, vn, 4.0))
    stacks = []
    for uid in subs:
        rimg = Image.open(f"{REF}/u{uid}_ref.png").convert("RGB")
        row = [label(rimg, f"ref u{uid}", cell)]
        for sm, vn, wv in combos:
            tag = f"{sm}{'+vn' if vn else ''} w{wv:g}"
            print(f"[e3] u{uid} sched={sm} vnorm={vn} w={wv}", flush=True)
            im = generate(uid, wv, refpos=a.refpos, steps=a.steps, cfg=a.cfg,
                          route="off", sched_mode=sm, sched_frac=a.sched_frac, vnorm=vn)
            fn = f"{OUT}/u{uid}_e3_{sm}_{'vn' if vn else 'raw'}_w{wv:g}.png"
            im.save(fn)
            row.append(label(im, tag, cell))
        canvas = Image.new("RGB", (len(row) * cell, cell), "white")
        for c, im in enumerate(row):
            canvas.paste(im, (c * cell, 0))
        canvas.save(f"{OUT}/montage_e3_{uid}.png")
        print("saved:", f"{OUT}/montage_e3_{uid}.png", flush=True)
        stacks.append(canvas)
    W = max(s.width for s in stacks); H = sum(s.height for s in stacks)
    combo = Image.new("RGB", (W, H), "white"); y = 0
    for s in stacks:
        combo.paste(s, (0, y)); y += s.height
    combo.save(f"{OUT}/montage_e3_all.png")
    print("saved:", f"{OUT}/montage_e3_all.png", flush=True)

def parse_band(s):
    s = s.strip()
    if "-" in s:
        lo, hi = s.split("-"); return set(range(int(lo), int(hi) + 1))
    return {int(s)}

def run_e4(a, subs):
    """E4: ablate injection layer band on the E3 best point (taper, w=4, route=off).
    Find the minimal load-bearing subset for identity."""
    bands = ["16-25", "16-20", "21-25", "16-18", "23-25",
             "21", "25", "12-15", "26-29", "10-29"]
    cell = 512
    stacks = []
    for uid in subs:
        rimg = Image.open(f"{REF}/u{uid}_ref.png").convert("RGB")
        row = [label(rimg, f"ref u{uid}", cell)]
        for b in bands:
            INJ["band"] = parse_band(b)
            print(f"[e4] u{uid} band={b} -> {sorted(INJ['band'])}", flush=True)
            im = generate(uid, 4.0, refpos=a.refpos, steps=a.steps, cfg=a.cfg,
                          route="off", sched_mode="taper", sched_frac=a.sched_frac)
            im.save(f"{OUT}/u{uid}_e4_band{b}.png")
            row.append(label(im, f"L{b}", cell))
        canvas = Image.new("RGB", (len(row) * cell, cell), "white")
        for c, im in enumerate(row):
            canvas.paste(im, (c * cell, 0))
        canvas.save(f"{OUT}/montage_e4_{uid}.png")
        print("saved:", f"{OUT}/montage_e4_{uid}.png", flush=True)
        stacks.append(canvas)
    W = max(s.width for s in stacks); H = sum(s.height for s in stacks)
    combo = Image.new("RGB", (W, H), "white"); y = 0
    for s in stacks:
        combo.paste(s, (0, y)); y += s.height
    combo.save(f"{OUT}/montage_e4_all.png")
    print("saved:", f"{OUT}/montage_e4_all.png", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "e1", "e3", "e4"])
    ap.add_argument("--subjects", default="29998,9172")
    ap.add_argument("--weights", default="off,2,4")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--refpos", default="strip")
    ap.add_argument("--route", default="off", choices=["off", "soft", "pct"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="override PROMPT (capture stays generic)")
    ap.add_argument("--neg", default=None)
    ap.add_argument("--cfg_id", type=float, default=0.0, help="dual-CFG identity-delta weight")
    ap.add_argument("--lowfreq_k", type=int, default=4, help="refpos=lowfreq: # of low-freq H/W rope pairs kept")
    ap.add_argument("--center_cut", type=float, default=1.0, help="refpos=center: fraction of steps before falling back to strip")
    ap.add_argument("--keep_pct", type=float, default=40.0)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--gate_strength", type=float, default=1.0)
    ap.add_argument("--sched_frac", type=float, default=0.6)
    ap.add_argument("--sched", default="const", choices=["const", "early", "taper"])
    ap.add_argument("--band", default="16-25")
    ap.add_argument("--tag", default="nolora")
    ap.add_argument("--measure", action="store_true",
                     help="log per-layer target-image-query -> ref-key softmax attention mass; "
                          "dumps to --measure_out (sweep mode only, last w in --weights)")
    ap.add_argument("--ref_bias", type=float, default=0.0,
                     help="causal knob: additive pre-softmax log-bias on ref-key columns (all heads); "
                          "0.0 = no-op (default)")
    ap.add_argument("--ref_theta", type=float, default=256.0,
                     help="refpos=thetasweep only: RoPE theta used to rotate the injected ref key "
                          "(256=Z native, 10000=Lumina); no-op for other refpos values")
    ap.add_argument("--measure_out", default=None)
    a = ap.parse_args()
    subs = [s for s in a.subjects.split(",") if s]
    INJ["band"] = parse_band(a.band)

    if a.mode == "e3":
        run_e3(a, subs)
        return
    if a.mode == "e4":
        run_e4(a, subs)
        return
    ws = [w if w == "off" else float(w) for w in a.weights.split(",")]

    if a.mode == "e1":
        run_e1(a, subs, ws)
        return

    grid = []
    for uid in subs:
        rimg = Image.open(f"{REF}/u{uid}_ref.png").convert("RGB").resize((1024, 1024))
        row = [rimg]
        for w in ws:
            tagw = "off" if w == "off" else f"{w:g}"
            is_last = (w == ws[-1])
            print(f"[gen] u{uid} w={tagw} seed={a.seed}", flush=True)
            im = generate(uid, w, refpos=a.refpos, steps=a.steps, cfg=a.cfg, sched_mode=a.sched,
                          seed=a.seed, prompt=a.prompt, neg=a.neg, cfg_id=a.cfg_id,
                          lowfreq_k=a.lowfreq_k, center_cut=a.center_cut,
                          measure=(a.measure and is_last), ref_bias=a.ref_bias, ref_theta=a.ref_theta)
            im.save(f"{OUT}/u{uid}_{a.tag}_w{tagw}.png")
            if a.measure and is_last:
                mo = a.measure_out or f"{OUT}/attn_quant_u{uid}_{a.tag}.json"
                dump_attn_log(mo)
            row.append(im)
        grid.append(row)

    cols = 1 + len(ws); rows = len(subs); cell = 1024
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    for r, row in enumerate(grid):
        for c, im in enumerate(row):
            canvas.paste(im.resize((cell, cell)), (c * cell, r * cell))
    mp = f"{OUT}/montage_{a.tag}.png"
    canvas.save(mp)
    print("saved montage:", mp, flush=True)

if __name__ == "__main__":
    main()
