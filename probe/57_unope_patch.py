#!/usr/bin/env python
"""
57_unope_patch.py — UnoPE-style diagonal H/W RoPE offset for the reference
image slot(s) in the omni (in-context) forward path, applied as a RUNTIME
MONKEYPATCH. Does NOT edit diffusers (transformer_z_image.py stays untouched
on disk); this file only swaps an INSTANCE attribute on an already-loaded
`ZImageTransformer2DModel`.

Motivation (source-verified, see task brief / repo memory for the full
derivation): `patchify_and_embed_omni` (diffusers 0.38.0.dev0,
transformer_z_image.py:625-762) calls, per image slot j,

    self._pad_with_ids(x_patches, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), ...)

i.e. EVERY image slot's H/W RoPE grid starts at (0, 0) -- only the F axis
differs (offset by the running caption/image cursor `cap_end_pos[j]`). Ref
and target therefore share the same spatial coordinate system, which enables
the copy-paste / pose-copy failure mode UNO's UnoPE addresses
(arXiv:2504.02160): give the reference slot(s) a spatial (H,W) offset so
their token grid is fully disjoint from the target's, while leaving the
target (last slot, the only one that is denoised/supervised per
`unpatchify`'s "return only the last image" contract) untouched.

Mechanism: bound-method swap via `types.MethodType`. `install_unope_patch(tf)`
sets `tf.patchify_and_embed_omni = types.MethodType(<our function>, tf)`,
which becomes an INSTANCE attribute. Python attribute lookup finds instance
`__dict__` entries before class-level methods (ordinary methods are
non-data descriptors), so this shadows `ZImageTransformer2DModel
.patchify_and_embed_omni` for this one instance only -- the class, and the
file on disk, are never touched. (A subclass would work too; bound-method
swap is simpler and avoids re-registering the model with
diffusers/peft/accelerate machinery that expects a stable class identity.)

`_patchify_and_embed_omni_unope` below is a byte-for-byte copy of the
upstream function (verified against the installed diffusers source at
/home/ubuntu/miniconda3/lib/python3.13/site-packages/diffusers/models/
transformers/transformer_z_image.py, 0.38.0.dev0) with exactly one change:
the "Process images" loop now offsets every NON-LAST image slot's (H,W)
pos_start by the LAST (target) slot's (H_tokens, W_tokens) token-grid size,
computed from that slot's own patchify output. The F-axis start
(`cap_end_pos[j]`) is untouched for every slot, including the target.

Hard invariant (asserted, not assumed): the last slot in each batch item's
image list MUST have images_noise_mask value 1 (the noised/supervised
target) and every other slot MUST have value 0 (a clean reference). This
matches this project's only calling convention, `image_noise_mask=[[0, 1]]`
(30_kv_inject.py / 45_train_lora.py). If some future caller violates it
(e.g. passes noise_mask=[[1,1]] or puts the target first), this raises
RuntimeError immediately rather than silently mis-offsetting a slot.

Wiring (project red line #5 -- must be IDENTICAL in train and inference):
probe/45_train_lora.py's stage_train() and stage_sample() both call
`_load("unope57", "57_unope_patch.py").install_unope_patch(kv.tf)` when
--unope is passed (default OFF, so existing runs/checkpoints/concurrent
jobs using 45_train_lora.py without --unope are byte-for-byte unaffected).

Run (self-test / verification gate, must PASS before any retrain):
  source ../env.sh && $PY 57_unope_patch.py --selftest
"""
import os, sys, types, argparse, importlib.util

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirror the constants the upstream function closes over (module-level in
# transformer_z_image.py); duplicated here (not imported) so this file has
# zero import-time dependency on diffusers internals beyond the public
# torch/diffusers API used at call time. Verified equal to the installed
# diffusers source: transformer_z_image.py lines 33-34.
SEQ_MULTI_OF = 32
X_PAD_DIM = 64


def _load(modname, filename):
    """Import a probe/ script as a module by file path (same pattern as
    45_train_lora.py / 50_production.py's own `_load()`)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patchify_and_embed_omni_unope(
    self,
    all_x: "list[list[torch.Tensor]]",
    all_cap_feats: "list[list[torch.Tensor]]",
    all_siglip_feats: "list[list[torch.Tensor]]",
    patch_size: int,
    f_patch_size: int,
    images_noise_mask: "list[list[int]]",
):
    """UnoPE-offset variant of ZImageTransformer2DModel.patchify_and_embed_omni.
    Identical to upstream except the "Process images" block (marked below)."""
    bsz = len(all_x)
    device = all_x[0][-1].device
    dtype = all_x[0][-1].dtype

    all_x_out, all_x_size, all_x_pos_ids, all_x_pad_mask, all_x_len, all_x_noise_mask = [], [], [], [], [], []
    all_cap_out, all_cap_pos_ids, all_cap_pad_mask, all_cap_len, all_cap_noise_mask = [], [], [], [], []
    all_sig_out, all_sig_pos_ids, all_sig_pad_mask, all_sig_len, all_sig_noise_mask = [], [], [], [], []

    for i in range(bsz):
        num_images = len(all_x[i])
        cap_feats_list, cap_pos_list, cap_mask_list, cap_lens, cap_noise = [], [], [], [], []
        cap_end_pos = []
        cap_cu_len = 1

        # Process captions -- UNCHANGED from upstream.
        for j, cap_item in enumerate(all_cap_feats[i]):
            noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
            cap_out, cap_pos, cap_mask, cap_len, cap_nm = self._pad_with_ids(
                cap_item,
                (len(cap_item) + (-len(cap_item)) % SEQ_MULTI_OF, 1, 1),
                (cap_cu_len, 0, 0),
                device,
                noise_val,
            )
            cap_feats_list.append(cap_out)
            cap_pos_list.append(cap_pos)
            cap_mask_list.append(cap_mask)
            cap_lens.append(cap_len)
            cap_noise.extend(cap_nm)
            cap_cu_len += len(cap_item)
            cap_end_pos.append(cap_cu_len)
            cap_cu_len += 2  # for image vae and siglip tokens

        all_cap_out.append(torch.cat(cap_feats_list, dim=0))
        all_cap_pos_ids.append(torch.cat(cap_pos_list, dim=0))
        all_cap_pad_mask.append(torch.cat(cap_mask_list, dim=0))
        all_cap_len.append(cap_lens)
        all_cap_noise_mask.append(cap_noise)

        # ============= Process images -- THE UNOPE CHANGE IS HERE =============
        x_feats_list, x_pos_list, x_mask_list, x_lens, x_size, x_noise = [], [], [], [], [], []

        # Pre-patchify every present image slot (no image is patchified
        # twice; upstream already calls _patchify_image once per slot, we
        # just do it before the pos_start decision instead of inline) so we
        # know the LAST (target) slot's (H_tokens, W_tokens) before
        # assigning any earlier (ref) slot's position ids.
        _patchified = [None] * num_images
        for j, x_item in enumerate(all_x[i]):
            if x_item is not None:
                _patchified[j] = self._patchify_image(x_item, patch_size, f_patch_size)

        last_j = num_images - 1
        last_noise_val = images_noise_mask[i][last_j] if last_j < len(images_noise_mask[i]) else 1
        if last_noise_val != 1:
            raise RuntimeError(
                f"[57_unope_patch] invariant violated: batch item {i}'s last image slot "
                f"(j={last_j}) must be the noised TARGET (noise_val==1), got "
                f"noise_val={last_noise_val}; images_noise_mask[{i}]={images_noise_mask[i]!r}. "
                f"UnoPE needs the target slot fixed at (H,W) start (0,0) and only knows how "
                f"to offset slots preceding it -- refusing to guess."
            )
        if _patchified[last_j] is None:
            raise RuntimeError(
                f"[57_unope_patch] invariant violated: batch item {i}'s last image slot "
                f"(j={last_j}) is None -- UnoPE needs a real target image to size the "
                f"ref-slot H/W offset."
            )
        _, _, (_, H_tgt_tok, W_tgt_tok) = _patchified[last_j]

        for j, x_item in enumerate(all_x[i]):
            noise_val = images_noise_mask[i][j]
            is_last = j == last_j
            if is_last:
                if noise_val != 1:
                    raise RuntimeError(
                        f"[57_unope_patch] invariant violated: batch item {i} slot j={j} "
                        f"(last) noise_val={noise_val} != 1"
                    )
                h_off, w_off = 0, 0
            else:
                if noise_val != 0:
                    raise RuntimeError(
                        f"[57_unope_patch] invariant violated: batch item {i} slot j={j} "
                        f"(non-last) noise_val={noise_val} != 0 -- UnoPE only knows how to "
                        f"offset CLEAN reference slots (noise_val==0); refusing to guess an "
                        f"offset for a noised non-target slot."
                    )
                h_off, w_off = H_tgt_tok, W_tgt_tok

            if x_item is not None:
                x_patches, size, (F_t, H_t, W_t) = _patchified[j]
                # pos_start F-axis (cap_end_pos[j]) is UNCHANGED from upstream;
                # only H_start/W_start move, and only for non-last (ref) slots.
                x_out, x_pos, x_mask, x_len, x_nm = self._pad_with_ids(
                    x_patches, (F_t, H_t, W_t), (cap_end_pos[j], h_off, w_off), device, noise_val
                )
                x_size.append(size)
            else:
                # Missing-image placeholder slot -- UNCHANGED from upstream
                # (never hit by this project's [ref, target] 2-slot calls;
                # left at (0,0,0) exactly like upstream since there is no
                # real target-grid-relative offset to apply to a slot with
                # no content).
                x_len = SEQ_MULTI_OF
                x_out = torch.zeros((x_len, X_PAD_DIM), dtype=dtype, device=device)
                x_pos = self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(x_len, 1)
                x_mask = torch.ones(x_len, dtype=torch.bool, device=device)
                x_nm = [noise_val] * x_len
                x_size.append(None)
            x_feats_list.append(x_out)
            x_pos_list.append(x_pos)
            x_mask_list.append(x_mask)
            x_lens.append(x_len)
            x_noise.extend(x_nm)

        all_x_out.append(torch.cat(x_feats_list, dim=0))
        all_x_pos_ids.append(torch.cat(x_pos_list, dim=0))
        all_x_pad_mask.append(torch.cat(x_mask_list, dim=0))
        all_x_size.append(x_size)
        all_x_len.append(x_lens)
        all_x_noise_mask.append(x_noise)
        # ============================ end UnoPE change ==========================

        # Process siglip -- UNCHANGED from upstream, EXCEPT a guard: this
        # project always calls with siglip_feats=[None] (30_kv_inject.py /
        # 45_train_lora.py), so this branch is never exercised in practice.
        # If it ever IS exercised with >1 image slot, refuse rather than
        # silently leaving siglip's ref-slot position ids un-offset while
        # the x (VAE-latent) ref-slot position ids ARE offset -- that would
        # be a half-applied, geometry-inconsistent UnoPE.
        if all_siglip_feats[i] is None:
            all_sig_len.append([0] * num_images)
            all_sig_out.append(None)
        else:
            if num_images > 1:
                raise NotImplementedError(
                    "[57_unope_patch] siglip_feats is non-None with multiple image slots -- "
                    "this patch only offsets x (VAE-latent) H/W position ids for ref slots; "
                    "it does not offset siglip position ids (unused in this project's "
                    "train/inference calls, which always pass siglip_feats=[None]). Refusing "
                    "rather than silently producing a half-applied UnoPE geometry."
                )
            sig_feats_list, sig_pos_list, sig_mask_list, sig_lens, sig_noise = [], [], [], [], []
            for j, sig_item in enumerate(all_siglip_feats[i]):
                noise_val = images_noise_mask[i][j]
                if sig_item is not None:
                    sig_H, sig_W, sig_C = sig_item.size()
                    sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                    sig_out, sig_pos, sig_mask, sig_len, sig_nm = self._pad_with_ids(
                        sig_flat, (1, sig_H, sig_W), (cap_end_pos[j] + 1, 0, 0), device, noise_val
                    )
                    if x_size[j] is not None:
                        sig_pos = sig_pos.float()
                        sig_pos[..., 1] = sig_pos[..., 1] / max(sig_H - 1, 1) * (x_size[j][1] - 1)
                        sig_pos[..., 2] = sig_pos[..., 2] / max(sig_W - 1, 1) * (x_size[j][2] - 1)
                        sig_pos = sig_pos.to(torch.int32)
                else:
                    sig_len = SEQ_MULTI_OF
                    sig_out = torch.zeros((sig_len, self.config.siglip_feat_dim), dtype=dtype, device=device)
                    sig_pos = (
                        self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(sig_len, 1)
                    )
                    sig_mask = torch.ones(sig_len, dtype=torch.bool, device=device)
                    sig_nm = [noise_val] * sig_len
                sig_feats_list.append(sig_out)
                sig_pos_list.append(sig_pos)
                sig_mask_list.append(sig_mask)
                sig_lens.append(sig_len)
                sig_noise.extend(sig_nm)

            all_sig_out.append(torch.cat(sig_feats_list, dim=0))
            all_sig_pos_ids.append(torch.cat(sig_pos_list, dim=0))
            all_sig_pad_mask.append(torch.cat(sig_mask_list, dim=0))
            all_sig_len.append(sig_lens)
            all_sig_noise_mask.append(sig_noise)

    # Compute x position offsets -- UNCHANGED from upstream.
    all_x_pos_offsets = [(sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_x_len[i])) for i in range(bsz)]

    return (
        all_x_out,
        all_cap_out,
        all_sig_out,
        all_x_size,
        all_x_pos_ids,
        all_cap_pos_ids,
        all_sig_pos_ids,
        all_x_pad_mask,
        all_cap_pad_mask,
        all_sig_pad_mask,
        all_x_pos_offsets,
        all_x_noise_mask,
        all_cap_noise_mask,
        all_sig_noise_mask,
    )


def install_unope_patch(transformer):
    """Install the UnoPE-offset patchify_and_embed_omni on THIS transformer
    INSTANCE only, via types.MethodType instance-attribute shadowing. Does
    NOT modify ZImageTransformer2DModel (the class) or any file on disk.
    Idempotent (re-installing just rebinds the same function). Returns the
    transformer for chaining."""
    transformer.patchify_and_embed_omni = types.MethodType(_patchify_and_embed_omni_unope, transformer)
    transformer._unope_patch_installed = True
    print(f"[57_unope_patch] installed on {type(transformer).__name__} instance "
          f"id={id(transformer)} (source file on disk untouched)", flush=True)
    return transformer


def uninstall_unope_patch(transformer):
    """Remove the instance-level override, restoring the stock class method
    (transformer_z_image.py's own patchify_and_embed_omni)."""
    if "patchify_and_embed_omni" in transformer.__dict__:
        del transformer.__dict__["patchify_and_embed_omni"]
    transformer._unope_patch_installed = False
    print(f"[57_unope_patch] uninstalled from {type(transformer).__name__} instance "
          f"id={id(transformer)}", flush=True)
    return transformer


# ==========================================================================
# Self-test / verification gate -- run BEFORE any retrain (task STEP 1).
# ==========================================================================

def _ok(cond, msg):
    print(f"[selftest] {'PASS' if cond else 'FAIL'}: {msg}", flush=True)
    return bool(cond)


def run_selftest():
    os.environ.setdefault("REF", "/mnt/local/smoke_anime/anime_260623/img")
    kv = _load("kv_inject_unope_selftest", "30_kv_inject.py")
    tf, DT, pipe = kv.tf, kv.DT, kv.pipe
    tf.eval()
    all_pass = True

    # ---- build a tiny 2-slot omni input: [ref, target] ----
    ref_dir = os.environ["REF"]
    ref_name = sorted(f for f in os.listdir(ref_dir) if f.endswith("_ref.png"))[0]
    ref_path = os.path.join(ref_dir, ref_name)
    with torch.no_grad():
        ref_lat = kv.ref_latent(ref_path, 512)               # (16,1,32,32) DT -- H_tok=W_tok=32
        gen = torch.Generator(kv.DEV).manual_seed(0)
        x_t = torch.randn((16, 1, 128, 128), generator=gen, device=kv.DEV, dtype=torch.float32).to(DT)  # H_tok=W_tok=64
        cap_ref = kv.enc("anime illustration of a character")
        cap_tgt = kv.enc("anime illustration of a character")
    all_x = [[ref_lat, x_t]]
    all_cap = [[cap_ref, cap_tgt]]
    all_sig = [None]
    nmask = [[0, 1]]
    print(f"[selftest] ref_lat shape={tuple(ref_lat.shape)} (H_tok=W_tok=32), "
          f"x_t shape={tuple(x_t.shape)} (H_tok=W_tok=64)", flush=True)

    # ---- (a) run patchify_and_embed_omni WITH and WITHOUT the patch ----
    unpatched_fn = type(tf).patchify_and_embed_omni  # class-level, unbound; bypasses any instance override
    out_no = unpatched_fn(tf, all_x, all_cap, all_sig, 2, 1, nmask)

    # Full unpatched forward() (before installing), same inputs/seed as the
    # patched forward() below -- proves the pos_id offset actually PROPAGATES
    # through _prepare_sequence -> RoPE -> attention into the output, not
    # just that patchify_and_embed_omni's returned pos_ids differ. Without
    # this, a wiring bug that computed the right pos_ids but dropped them
    # before RoPE would still pass every assertion above.
    t_tensor = torch.tensor([0.5], device=kv.DEV, dtype=torch.float32)
    with torch.no_grad():
        out_fwd_unpatched = tf(x=[[ref_lat, x_t]], t=t_tensor, cap_feats=[[cap_ref, cap_tgt]],
                               siglip_feats=[None], image_noise_mask=[[0, 1]], return_dict=False)[0]

    install_unope_patch(tf)
    out_yes = tf.patchify_and_embed_omni(all_x, all_cap, all_sig, 2, 1, nmask)

    (x_out_no, cap_out_no, sig_out_no, x_size_no, x_pos_ids_no, *_rest_no) = out_no
    (x_out_yes, cap_out_yes, sig_out_yes, x_size_yes, x_pos_ids_yes, *_rest_yes) = out_yes

    # slot boundaries within the concatenated x_pos_ids[0]: ref=1024 rows
    # (32*32, already SEQ_MULTI_OF=32-aligned), target=4096 rows (64*64,
    # already aligned) -- row 0 of a slot's segment IS that slot's true
    # (F_start, H_start, W_start) by construction (create_coordinate_grid
    # + flatten puts the start point first; padding rows are appended
    # AFTER the real ones in _pad_with_ids).
    REF_LEN, TGT_LEN = 32 * 32, 64 * 64

    ref_start_no = x_pos_ids_no[0][0].tolist()
    tgt_start_no = x_pos_ids_no[0][REF_LEN].tolist()
    ref_start_yes = x_pos_ids_yes[0][0].tolist()
    tgt_start_yes = x_pos_ids_yes[0][REF_LEN].tolist()
    print(f"[selftest] WITHOUT patch: ref (F,H,W) start={ref_start_no}, target start={tgt_start_no}", flush=True)
    print(f"[selftest] WITH    patch: ref (F,H,W) start={ref_start_yes}, target start={tgt_start_yes}", flush=True)

    # ---- (b) assertions ----
    all_pass &= _ok(tgt_start_no[1] == 0 and tgt_start_no[2] == 0,
                     "target slot H/W start == 0 WITHOUT patch")
    all_pass &= _ok(tgt_start_yes[1] == 0 and tgt_start_yes[2] == 0,
                     "target slot H/W start == 0 WITH patch (unchanged)")
    all_pass &= _ok(ref_start_no[1] == 0 and ref_start_no[2] == 0,
                     "ref slot H/W start == 0 WITHOUT patch (the bug being fixed)")
    all_pass &= _ok(ref_start_yes[1] == 64 and ref_start_yes[2] == 64,
                     "ref slot H/W start == (H_tgt_tok=64, W_tgt_tok=64) WITH patch")
    all_pass &= _ok(ref_start_no[0] == ref_start_yes[0] and tgt_start_no[0] == tgt_start_yes[0],
                     "F-axis start identical patched vs unpatched (ref and target each)")
    all_pass &= _ok(x_out_no.__class__ == x_out_yes.__class__ and len(x_out_no) == len(x_out_yes)
                     and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(x_out_no, x_out_yes)),
                     "all_x_out shapes/dtypes identical patched vs unpatched")
    all_pass &= _ok(all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(x_pos_ids_no, x_pos_ids_yes)),
                     "all_x_pos_ids shapes/dtypes identical patched vs unpatched")
    all_pass &= _ok(x_size_no == x_size_yes, "all_x_size (F,H,W per slot) identical patched vs unpatched")

    # ---- negative test: invariant must hard-fail on a bad noise_mask ----
    try:
        tf.patchify_and_embed_omni(all_x, all_cap, all_sig, 2, 1, [[1, 1]])
        neg_raised = False
    except RuntimeError as e:
        neg_raised = "invariant violated" in str(e)
    all_pass &= _ok(neg_raised, "patch hard-fails (RuntimeError) on an invalid noise_mask ([[1,1]])")

    # ---- (b) full forward() determinism, bit-identical across two calls ----
    with torch.no_grad():
        out1 = tf(x=[[ref_lat, x_t]], t=t_tensor, cap_feats=[[cap_ref, cap_tgt]],
                  siglip_feats=[None], image_noise_mask=[[0, 1]], return_dict=False)[0]
        out2 = tf(x=[[ref_lat, x_t]], t=t_tensor, cap_feats=[[cap_ref, cap_tgt]],
                  siglip_feats=[None], image_noise_mask=[[0, 1]], return_dict=False)[0]
    bit_exact = len(out1) == len(out2) and all(torch.equal(a, b) for a, b in zip(out1, out2))
    all_pass &= _ok(bit_exact, "forward() bit-identical across two calls WITH patch active (determinism)")

    # ---- end-to-end propagation: patched forward() output must DIFFER from
    # unpatched (same inputs/seed) -- proves the pos_id offset reaches
    # attention/output, not just patchify_and_embed_omni's return values.
    diverges = not torch.equal(out1[0], out_fwd_unpatched[0])
    max_abs_diff = (out1[0].float() - out_fwd_unpatched[0].float()).abs().max().item()
    all_pass &= _ok(diverges, f"patched forward() output DIFFERS from unpatched (same seed/inputs), "
                               f"max_abs_diff={max_abs_diff:.4f} -- proves offset propagates to attention output")

    # ---- (c) unpatchify still returns only the target slot, correct shape ----
    shape_ok = tuple(out1[0].shape) == (16, 1, 128, 128)
    all_pass &= _ok(shape_ok, f"unpatchify returns target-only tensor of shape (16,1,128,128), got {tuple(out1[0].shape)}")
    nan_ok = not torch.isnan(out1[0]).any().item()
    all_pass &= _ok(nan_ok, "forward() output contains no NaNs")

    uninstall_unope_patch(tf)
    out_restored = unpatched_fn(tf, all_x, all_cap, all_sig, 2, 1, nmask)
    restored_ok = out_restored[4][0][REF_LEN][1].item() == 0 and out_restored[4][0][0][1].item() == 0
    all_pass &= _ok(restored_ok, "uninstall_unope_patch() restores stock (0,0) H/W starts for both slots")

    print(f"\n[selftest] ================ OVERALL: {'PASS' if all_pass else 'FAIL'} ================", flush=True)
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the geometry + determinism verification gate")
    a = ap.parse_args()
    if a.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)
    ap.error("nothing to do; pass --selftest")


if __name__ == "__main__":
    main()
