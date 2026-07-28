# Identity Residual TSD Prescreen

Date: **Thursday, July 23, 2026 UTC**

## Purpose

Test whether a transient strip-reference teacher can provide an identity
direction that is distilled into the latent while the scheduler consumes only
the recomputed pure-T2I prediction.

## Code

- `probe/110_identity_residual_tsd.py`
- `probe/test_110_identity_residual_tsd.py`

Five helper tests pass: scheduler sigma lookup before the first step, mean
consensus, Gram-SVD finiteness, zero-residual behavior, and low-frequency ratio.

The first implementation had two bugs caught during smoke testing:

1. `scheduler.step_index` is unset before the first reverse step. Sigma is now
   resolved by nearest scheduler timestep.
2. The x0 residual used the wrong squeeze/unsqueeze order and created a 5-D
   latent. A shape assertion now protects `(1,C,H,W)` output.

`torch.inference_mode()` and explicit transient cleanup keep persistent GPU
allocation stable at about 11.60 GiB during denoising, allowing the 512 px test
to coexist with unrelated GPU jobs.

## Matched Results

| Arm | Output | CCIP |
|---|---|---:|
| Prompt only | `out/identity_residual_tsd/u1839_prompt_only_512_n8_s0_mean_a2.png` | 0.326769 |
| TSD 0.01, first 2 | `out/identity_residual_tsd/u1839_smoke_512_n8_inference_s0_mean_a2.png` | 0.296381 |
| TSD 0.03, first 2 | `out/identity_residual_tsd/u1839_tsd003_512_n8_s0_mean_a2.png` | 0.307921 |
| TSD 0.01, first half | `out/identity_residual_tsd/u1839_tsd001_half_512_n8_s0_mean_a2.png` | 0.360401 |
| High-pass + channel-centered TSD | `out/identity_residual_tsd/u1839_nuisance_hpch_512_n8_s0_mean_a2.png` | 0.326464 |
| Direct teacher | `out/identity_residual_tsd/u1839_direct_teacher_512_n8_s0_mean_a1.png` | 0.241509 |

Reference: `/mnt/local/smoke_anime/anime_260623/img/u1839_ref.png`.

## Interpretation

The teacher contains useful identity signal, but direct deployment corrupts
color and structure. Weak early latent distillation transfers a small portion
of the signal without those defects. Stronger or repeated updates amplify a
nuisance direction and worsen identity. The next experiment must change
residual decomposition, not merely alpha or update count.

The simple nuisance projection is also negative: it reduces the first-step
low-frequency ratio from 0.881 to 0.094 but removes the CCIP gain. Identity and
composition cannot be separated by a generic high-pass filter.
