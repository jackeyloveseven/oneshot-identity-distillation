# Identity Capsule and Residual Audit

Date: **Thursday, July 23, 2026 UTC**

## Question

Can same-state multi-reference residual distillation add identity detail beyond
a concise semantic identity capsule in a frozen pure-T2I Z-Image model?

## Implemented Arms

- `probe/111_identity_capsule.py`: Gemini 3 Flash multi-reference identity
  capsule extraction with strict exclusion of pose, background, expression,
  framing, lighting, and conflicting details.
- `probe/110_identity_residual_tsd.py --aggregation semantic`: true-reference
  consensus, local agreement, and within-reference nuisance-tangent projection.
- optional `--hidden-support-layers 21-25`: hidden difference support only; no
  hidden-state transplant.
- optional `--residual-contrast identity_interaction`: difference-in-differences
  subtracting generic reference response without the identity capsule.

All deployed residual arms update only the latent, recompute the pure T2I
prediction, and pass only that prediction to `scheduler.step`.

## Capsule Quality Guard

The first `u9728` capsule repeated purple/black palette terms and caused a blue
structure collapse. The generator now enforces 12-28 words and rejects repeated
color words. The corrected capsule restored coherent generation. This failure
is retained as an implementation warning, not counted as a method result.

Capsule sidecars:

- `out/identity_capsules/u1839.json`
- `out/identity_capsules/u4229.json`
- `out/identity_capsules/u9728.json`

## Fixed Protocol

- subjects: `u1839`, `u4229`, `u9728`;
- references: original reference plus same-subject white-background reference;
- seed 0, 512 px output, 256 px reference encoding;
- teacher weight 4, TSD step 0.01;
- semantic update on the first two denoising steps;
- no best-of-N and no image-specific rerun;
- primary verification at 20 denoising steps.

CCIP is averaged over both references; lower is better and 0.178 is the SAME
threshold. Every output was inspected directly for identity and visible defects.

## Results

### Portrait

| UID | capsule prompt-only | semantic residual | delta |
|---|---:|---:|---:|
| u1839 | 0.115624 | 0.122607 | +0.006984 |
| u4229 | 0.059339 | 0.083124 | +0.023785 |
| u9728 | 0.094133 | 0.106026 | +0.011893 |

Capsule prompt-only is coherent and identity-correct for all three. The
semantic residual worsens all three.

### Common rainy-street edit

Prompt: one subject standing on a rain-soaked neon city street at night,
three-quarter body, holding a transparent umbrella, no other people.

| UID | capsule prompt-only | semantic residual | delta |
|---|---:|---:|---:|
| u1839 | 0.102128 | 0.084433 | -0.017696 |
| u4229 | 0.122075 | 0.159004 | +0.036929 |
| u9728 | 0.118632 | 0.119277 | +0.000645 |

All six outputs are coherent and complete the requested edit. Residual
distillation improves only one subject, is neutral on one, and harms one.

### Negative mechanism checks

- Layers 21-25 hidden support worsens semantic residual by +0.005019,
  +0.003179, and +0.020076 on the three 8-step capsule checks.
- Identity-interaction double difference worsens rainy-street capsule
  prompt-only by +0.023993, +0.018995, and +0.005988.
- The earlier generic-prompt output residual did not recover correct identity;
  high residual agreement was agreement on nuisance, not proof of identity.

## Decision

Reject output-residual distillation as the current method. It does not pass the
two-of-three improvement rule at 20 steps, and the hidden or double-difference
extensions do not rescue it. Do not continue scalar alpha/window/filter,
hidden-threshold, or second-order sweeps.

The validated positive result is multi-reference identity capsule prompting.
It is a strong baseline and a necessary semantic coordinate system, but its
novelty is unverified and it is not yet sufficient for a paper claim. The next
method must recover visual detail not expressible in the capsule through a
different objective, then undergo a prompt/MLLM-personalization collision audit.

## Verification

- `probe/test_110_identity_residual_tsd.py`: 12/12 tests pass.
- Persistent denoising allocation with hidden recording: about 11.90 GiB.
- Source repositories remained read-only.
