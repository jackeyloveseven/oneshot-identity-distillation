# Edit-to-Generation Identity Distillation

Date: Monday, July 27, 2026 UTC
Status: **completed locked v3 evaluation and active paper method**

## Paper thesis

The active paper studies whether a reference-conditioned image-edit model can
serve as a transient teacher for subject-driven generation. The teacher is
queried only for same-state score differences. Its prediction never enters the
distilled scheduler. A trust-region inner loop writes the teacher signal into
the current latent, after which the ordinary generation trajectory continues.

Paper title:

> One-Shot Identity Personalization by Distilling an Image Editor at Test Time

The superseded localization and Band-LoRA manuscript has been removed. The
canonical paper is `paper/main_aaai_tsd.tex`.

## Method contract

For current packed latent \(z_\sigma\), attribute-matched text-only base
velocity \(v_A\), and reference-conditioned edit-teacher velocity \(v_T\):

\[
\Delta s = -\frac{1-\sigma}{\sigma}(v_T-v_A).
\]

The score direction is normalized per packed token and used to form a detached
clean-image pseudo-target:

\[
\hat x_0 = z_\sigma-\sigma\,\operatorname{sg}(v_A),\qquad
x_0^\star =
\operatorname{sg}(\hat x_0)+
\eta\,\operatorname{RMSNorm}_{\rm token}(\Delta s).
\]

Three inner updates minimize

\[
\frac12\lVert\hat x_0-\operatorname{sg}(x_0^\star)\rVert_2^2.
\]

The ordinary scheduler step also uses \(v_A\). Every candidate update is
projected around that post-scheduler text-only state into both a global RMS
ball and a per-token RMS ball. The global radius is bounded by three times the
teacher's one-Euler-step displacement, with caps 0.10 globally and 0.085 per
token.

Implementation:

- `probe/115_krea2_identity_tsd.py`
- `probe/test_115_krea2_identity_tsd.py`

## Why the teacher receives an automatic attribute phrase

The three-identity class-only pilot showed that reference-only editing can work,
but an input-only external check exposed a generalization failure: for u5319,
the edit teacher converted a distinctive blonde, blue-eyed character with a
pinned braided crown into a generic brunette even after foreground isolation
and face-centered cropping.

The LoRA-off matched reconstruction preserved the missing traits, so the
reference path was functioning. The failure was specifically in the edit
teacher's ability to retain identity through a large scene change. The fixed
protocol therefore supplies both the text-only base and the teacher with the
same pre-existing automatically extracted visible-attribute phrase. Only the
teacher receives reference-image tokens and source latents.

This creates a necessary stronger baseline:

- class-only base: no identity information;
- attribute-only base: automatic visible attributes, no reference image;
- distilled: the attribute-only trajectory plus the reference-conditioned
  teacher-minus-text score residual;
- direct teacher: reference-conditioned edit trajectory;
- matched reconstruction: LoRA-off reference-conditioning control;
- wrong-ID teacher: same-class deranged reference with its own matched
  attribute phrase.

The main method claim requires distilled output to improve over the
attribute-only base. Improvement over the class-only base alone is
insufficient.

## Held-out protocol

Development identities:

- u1839
- u4229
- u9728

Evaluation identities:

- 30 subjects, disjoint from development;
- anonymous synthetic anime character references from the pre-existing
  GPT-Image-2 reverse-reference data factory; no real-person photographs;
- selected by input-only review from the pre-existing `repprobe30` and
  `holdout30` sets;
- one human 2D anime or manga subject with visible identity cues;
- photographic, 3D, non-human, and development inputs excluded;
- 16 women and 14 men under the automatic class tags.

The full UID list and source-set provenance are frozen in
`data/krea2_eval30_protocol.json`.

### Reference preprocessing

All selected identities had an existing foreground-isolated white-background
reference. The largest anime-head detection is expanded by a fixed 0.35
head-box padding, squared with white out-of-bounds fill, and resized to
768 by 768. Detection succeeded on 30/30 inputs.

This crop is determined from the reference only. No generated output or score
is used. Exact boxes, detector confidences, source hashes, and output hashes are
stored in `data/krea2_eval30_reference_preprocessing.json`.

### Prompts and generation

Three class-only scenes test different composition regimes:

1. close-up portrait in a glass atrium, 640 by 640;
2. three-quarter rain-street umbrella scene, 512 by 768;
3. waist-up tea preparation in a kitchen, 768 by 512.

Each subject-scene pair uses seeds 2027 and 2028. The complete experiment is:

- 30 identities;
- 3 scenes;
- 2 fixed seeds;
- 180 paired observations;
- 6 output arms per observation;
- 1,080 generated images.

Wrong references use a deterministic cyclic derangement within the same
subject class.

### Locked distillation settings

- Krea-2-Turbo base;
- Identity-Edit LoRA v1.2 teacher;
- 12 Euler steps;
- 3 inner iterations;
- inner learning rate 0.05;
- pseudo-step size 1.0;
- reference attention boost 4;
- active over the full trajectory;
- per-token RMS score normalization;
- raw teacher-minus-attribute-matched-text residual;
- text-only automatic-attribute deployment trajectory;
- global trust cap 0.10;
- per-token trust cap 0.085;
- adaptive trust ratio 3.0.

## Evaluation

Primary identity outcomes:

- full-image anime CCIP distance to the correct reference;
- detected-head CCIP distance;
- OpenAI CLIP ViT-B/32 image similarity to the reference;
- strict CCIP SAME rate at 0.178;
- distance improvement over class-only and attribute-only bases;
- correct-reference specificity margin against the same-class wrong reference;
- top-1 identity retrieval, same-class top-1 retrieval, mean rank, and
  nearest-wrong margin against all 30 evaluation references.

Scene and structure outcomes:

- OpenAI CLIP ViT-B/32 similarity to the class-only scene prompt;
- anime-head detector counts for missing or duplicate subjects;
- anime-domain aesthetic percentile and label distribution;
- visible anatomy, collage, duplicate-person, and identity-drift audit;
- pixel L1 against the reference as a copy diagnostic.

Statistical unit:

- 30 subject clusters;
- all scene and seed observations remain nested inside subjects;
- subject-clustered bootstrap intervals;
- two-sided subject-level sign tests.

The 180 generated observations must not be treated as 180 independent
identities.

## Pre-expansion implementation audit

The first large-run attempt exposed that the nominal attribute-only baseline
reused a grounded embedding containing reference-image tokens. Those outputs
were quarantined and the run was restarted with a separately encoded
text-only prompt. A second formula audit then changed the distilled scheduler
and residual denominator from the class-only base to this attribute-matched
text base. This holds the explicit visible-attribute phrase constant at the
method level. The teacher still requires reference-use instruction wording, so
the prompt strings are not claimed to be identical. The wrong-reference
same-text intervention separately tests visual-reference causality.

The locked v3 smoke test uses held-out u5319, portrait, seed 2027:

| class-only | text-only attributes | distilled | teacher | wrong-ID teacher |
|---:|---:|---:|---:|---:|
| 0.3157 | 0.1672 | **0.0410** | 0.0507 | 0.2544 |

Visual inspection shows recovery of the pinned crown braid, narrow eye shape,
sharp jaw, and stern expression while retaining the new glass-atrium scene.
This observation validates execution only and remains part of the complete
held-out evaluation.

## Locked v3 results

The complete 30-identity run contains 180 paired observations. Distillation
reduces mean full-image CCIP from 0.1860 for attribute-only text to 0.0659;
the direct teacher obtains 0.0751. Correct-reference 30-way Top-1 retrieval
rises from 42.8% to 91.7%, compared with 88.3% for the teacher. The blind
three-pass VLM audit assigns identity scores of 24.7%, 93.9%, and 88.9% to
attribute text, distillation, and teacher, respectively.

The gain is not free. Relative to attribute-only text, scene CLIP changes by
-0.0090 and predicted aesthetics by -0.0651. The distilled visible-defect rate
is 3.9% under the automated audit.

On the 10-identity mechanism subset, the full method obtains 0.0554 CCIP.
Removing projection gives 0.1289, using the raw score magnitude gives 0.1288,
and using one inner update gives 0.0624. Replacing the correct reference with a
same-class wrong reference gives 0.2138 CCIP and 0% correct-identity Top-1,
supporting visual-reference specificity.

The complete reports are:

- `out/krea2_tsd_eval30_tagged_v3/eval30_report.html`
- `out/krea2_tsd_eval30_ablations_v3/ablation_report.html`

## Claim boundary

Supported:

- an edit model can act as a transient score teacher for a generation
  trajectory;
- trust-region test-time distillation can improve identity relative to both a
  class-only input and an automatic attribute-only baseline while retaining
  scene control;
- same-class wrong-reference controls can verify subject specificity.

Not supported:

- state of the art;
- reference-only identity transfer, because the teacher receives an automatic
  attribute phrase;
- generalization beyond human anime or manga subjects;
- success when the edit teacher itself lacks the target identity;
- a claim about where identity is represented inside the transformer.

Data-release limitation:

- references are synthetic and anonymous, but their upstream character-bot
  contexts may contain familiar fictional archetypes; any public release
  requires a separate provenance and rights review.

## Runners and reports

Resumable runner:

- `probe/121_run_krea2_eval30.py`

Scoring and complete visual audit:

- `probe/123_krea2_eval30_report.py`
- `out/krea2_tsd_eval30_tagged_v3/eval30_metrics.json`
- `out/krea2_tsd_eval30_tagged_v3/eval30_report.html`

Mechanism ablation:

- first five identities per automatic class in the frozen protocol order;
- all three scenes and seed 2027;
- 30 paired observations per condition;
- fixed trust radius;
- global-only trust radius;
- loose trust radius;
- one inner update;
- global instead of tokenwise RMS normalization;
- unnormalized analytic score magnitude;
- no reference-attention boost;
- early-only update window;
- late-only update window;
- matched-LoRA instead of raw teacher-minus-text residual;
- constant reference-reconstruction identity anchor;
- quadratically increasing late identity anchor;
- same-class wrong reference with the target text held fixed.

Runner and report:

- `probe/124_run_krea2_eval30_ablations.py`
- `probe/125_krea2_eval30_ablation_report.py`
- `out/krea2_tsd_eval30_ablations_v3/ablation_metrics.json`
- `out/krea2_tsd_eval30_ablations_v3/ablation_report.html`
