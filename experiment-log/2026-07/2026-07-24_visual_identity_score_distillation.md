# Visual-Identity Score Distillation

Date: **Friday, July 24, 2026 UTC**

## Question

Can a frozen visual-reference teacher inject character identity through
test-time score distillation when the text prompt contains no identity capsule
and the scheduler consumes only the pure T2I base prediction?

## Score conversion

The Z-Image probe returns prediction `p = -v`. Under rectified-flow
parameterization,

\[
\epsilon = z_\sigma + (1-\sigma)v
             = z_\sigma-(1-\sigma)p
\]

and

\[
s = -\epsilon/\sigma
  = -z_\sigma/\sigma + (1-\sigma)p/\sigma.
\]

Therefore, for two predictions evaluated at the same latent and timestep,

\[
\Delta s
= \frac{1-\sigma}{\sigma}\Delta p.
\]

This is the sign and coefficient implemented in
`probe/112_true_score_tsd.py`.

## Removing the generic reference response

Let `p0` be the pure-T2I prediction, `p+` the mean teacher prediction from
target-ID references, and `p-` the mean teacher prediction from mismatched-ID
references:

\[
r_+ = \bar p_+ - p_0,\qquad
r_- = \bar p_- - p_0.
\]

The original `r+` contains both target identity and a large generic
"reference present" response. On u1839, the positive and negative residual
cosine is 0.67--0.96 (mean 0.80).

### Exact difference-in-differences

\[
g_{\mathrm{sub}}=r_+-r_-=\bar p_+-\bar p_-.
\]

This cancels the base algebraically but also injects the negative identities
with a negative sign. It failed empirically.

### Least-squares projection

\[
\beta^*
=\operatorname{clip}_{[0,2]}
  \frac{\langle r_+,r_-\rangle}{\lVert r_-\rVert_2^2+\epsilon},
\qquad
g_\perp=r_+-\beta^*r_-.
\]

When the coefficient is not clipped, `g_perp` is orthogonal to the estimated
generic-reference direction. This avoids treating a mismatched character as
an explicit negative identity.

The score pseudo-direction is

\[
\hat g_s
=\operatorname{RMSNorm}
  \left(\frac{1-\sigma}{\sigma}g_\perp\right).
\]

With base velocity `v0`, the detached target and loss are

\[
\hat x_0=z_\sigma-\sigma v_0,\qquad
x_0^*=\operatorname{sg}(\hat x_0)+\eta\hat g_s,
\]

\[
L
=2\cdot\frac{1}{2}\lVert\hat x_0-\operatorname{sg}(x_0^*)\rVert_2^2
0.2\,d_i\lVert z_\sigma-z_{\mathrm{anchor}}\rVert_1.
\]

Five AdamW iterations are used per active outer state.

## Critical ordering correction

The initial implementation optimized the current latent and immediately
passed it through `scheduler.step` with a pure base score. This erased the
distilled identity.

The source TSD ordering is:

1. perform exactly one pure-base scheduler step at `t_k`;
2. take the resulting latent as the state at `t_{k+1}`;
3. run the teacher/base inner optimization at `sigma_{k+1}`;
4. feed that optimized state into the next outer step.

Reference K/V is still never passed to `scheduler.step`. The post-step
ordering only changes which latent becomes the next outer state.

## u1839 matched results

All visual-distillation arms omit the identity capsule. CCIP is averaged over
the two target references; lower is better and `< 0.178` is SAME.

| Arm | CCIP | Result |
|---|---:|---|
| Pure T2I base | 0.210479 | DIFF; black silhouette |
| Pre-step projected TSD | 0.242142 | DIFF; worse than base |
| Post-step exact subtraction | 0.220597 | DIFF |
| Post-step LS projection | **0.132801** | **SAME; clean visual ID** |
| Direct visual teacher | 0.133582 | SAME; severe layout collage |
| Capsule-only base | 0.102128 | SAME; text-assisted upper reference |

Important outputs:

- `out/true_score_tsd/u1839_vid_project_post_e95_s0_n20_i5.png`
- `out/true_score_tsd/u1839_vid_subtract_post_e95_s0_n20_i5.png`
- `out/identity_residual_tsd/u1839_direct_nocap_diag_s0_mean_a1.png`
- `out/true_score_tsd/u1839_ablation_nocap_base_s0_n20_i5.png`

The projected post-step arm matches the direct teacher's CCIP while removing
the direct teacher's reference-background and picture-in-picture defects.

## Cross-identity check: u4229

| Arm | CCIP | Visual result |
|---|---:|---|
| Pure T2I base | 0.341806 | DIFF; black silhouette |
| Full LS projection | **0.260681** | Partial gain; spiky-hair silhouette |
| Global shrinkage, alpha=0.5 | 0.249748 | Small metric gain; severe collage |
| Spatial gated shrinkage | 0.294691 | Worse; local collage remains |
| Direct visual teacher | 0.096741 | SAME; severe collage |

The direct teacher proves that u4229 identity exists in the frozen visual
teacher. Full projection removes layout contamination but also removes
identity components that are collinear with generic character-reference
structure.

Global shrinkage

\[
g_\alpha=r_+-\alpha\beta^*r_-,\quad 0\le\alpha\le1
\]

restored identity only weakly and immediately restored collage. A first
spatial gate,

\[
g=g_\perp+\lambda M(g_\perp)\odot(\beta^*r_-),
\]

also failed because the residual-energy gate covered 0.71 of the latent grid
on average. The teacher artifact itself produces high residual energy, so the
same residual cannot provide an independent subject mask.

Important outputs:

- `out/true_score_tsd/u4229_vid_project_post_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_shrink05_post_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_gated05q75_post_e95_s0_n20_i5.png`
- `out/identity_residual_tsd/u4229_direct_nocap_diag_s0_mean_a1.png`

## Portrait-stability diagnosis

Visual review separates two failure modes that CCIP alone conflates:

- the full-projection arm preserves a coherent silhouette but does not recover
  the u4229 face;
- shrinkage, residual-energy gating, and the direct teacher recover more
  reference-like local evidence by introducing face/limb deformation and
  picture-in-picture collage.

The latent inner loss in `112_true_score_tsd.py` has the form

\[
\mathcal L_{\rm style}
=\frac12\left\|x_0(z)-\operatorname{sg}
\left[x_0(z)+\eta d\right]\right\|_2^2,
\qquad
x_0(z)=z-\operatorname{sg}[\sigma v_b].
\]

Because the base velocity and pseudo-target are detached,

\[
\nabla_z\mathcal L_{\rm style}=-\frac{\eta}{N}d.
\]

Reinitializing AdamW at every outer step makes its first update approximately
a coordinate-wise sign step. It therefore removes much of the intended scale
control from `pseudo_step_size`, taper, and the raw teacher confidence.
The original e95 u4229 arm reached mean/max per-outer latent RMS offsets of
0.010446/0.014938.

`probe/113_trust_region_score_tsd.py` replaces only that optimizer with a
mean-corrected SGD step and projects the cumulative inner displacement onto an
RMS ball around the post-scheduler latent:

\[
\Delta z=-\alpha N\nabla_z\mathcal L,\qquad
z\leftarrow z_{\rm anchor}
+\Delta z\min\left(1,\frac{\rho}{\operatorname{RMS}(\Delta z)}\right).
\]

At \(\rho=0.003\), the mean/max offsets fell to 0.001036/0.001941. At
\(\rho=0.006\) with a 3x corrected step, the portrait remained coherent and
collage-free, but the target identity was still absent. Short-window Adam
controls gave the same result. This establishes that optimizer stabilization
solves the collapse axis but not the identity/layout separation axis.

New outputs:

- `out/true_score_tsd/u4229_vid_project_safe_a05e25_lr2e3_s0_n20_i3.png`
- `out/true_score_tsd/u4229_vid_project_safe_a20e50_lr2e3_s0_n20_i3.png`
- `out/true_score_tsd/u4229_vid_project_trust_r003_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_project_trust_r006_lr3e2_e95_s0_n20_i5.png`

## Paper-facing story

### Mini-outline

1. Reference-only identity injection is a constrained transfer problem:
   stronger teacher guidance improves identity evidence but also imports layout.
2. Cross-identity score projection controls the **direction** of transfer by
   removing reference responses shared with mismatched identities.
3. Post-scheduler trust-region distillation controls the **radius** of transfer
   by keeping each optimized latent near the pure-base reverse trajectory.
4. u1839 validates that clean identity transfer is possible, while u4229 shows
   that direction and radius control still cannot identify where correlated
   identity structure should be restored.
5. The remaining technical problem is independent subject-support estimation,
   not a larger teacher weight or a weaker global projection.

### Draft narrative

**Opening.** Reference-conditioned diffusion teachers contain useful identity
signals, but directly deploying their predictions couples identity transfer
with the reference image's pose, crop, and layout. The practical objective is
therefore not to maximize teacher influence. It is to move the pure text-to-image
trajectory toward the target identity without leaving the manifold of coherent
scene generation.

**Challenge.** This objective has two independent failure axes. A contaminated
score direction copies reference structure even when its norm is small, while
an identity-selective direction can still damage anatomy when repeated latent
updates accumulate across denoising states. Treating either failure as a single
guidance-weight problem hides the distinction between *where to move* and *how
far to move*.

**Method.** We decompose safe identity distillation into direction and radius
control. First, a same-state cross-identity projection removes the component of
the target-reference residual aligned with mismatched-reference residuals.
Second, each pure-base scheduler step defines a fresh latent anchor, and a
mean-corrected trust-region optimizer constrains the subsequent inner-loop
displacement in RMS norm. The scheduler never consumes a reference-conditioned
prediction.

**Evidence and limitation.** On u1839, post-step projection reaches CCIP
0.1328 while removing the direct teacher's collage artifacts. On u4229, the
direct teacher reaches CCIP 0.0967 but visibly collapses, whereas trust-region
distillation lowers the mean per-state latent displacement from 0.01045 to
0.00312 and restores structural coherence. The target identity remains absent,
however, because its discriminative structure is collinear with the generic
reference response removed by global projection. This counterexample narrows
the next contribution to an independent subject-support gate.

### Claim-evidence map

- Claim: cross-identity projection can produce clean identity transfer.
  Evidence: u1839 post-step projection, CCIP 0.1328 with no visible collage.
  Status: supported for one identity; broader evidence needed.
- Claim: trust-region optimization controls portrait collapse.
  Evidence: u4229 latent RMS decreases from 0.01045 to 0.00312 with coherent
  anatomy in the trust-region output. Status: supported qualitatively and by
  optimizer diagnostics.
- Claim: direction and radius control solve reference-only ID injection for
  arbitrary identities. Evidence: contradicted by the stable but identity-free
  u4229 output. Status: unsupported and intentionally not claimed.

### Self-review

- Clarity: direction and radius have distinct definitions and failure modes.
- Flow: task conflict leads to two constraints, then to cross-identity evidence.
- Terminology: use `score projection`, `trust region`, and `subject support`
  consistently.
- Unsupported claims: do not describe the method as generally successful until
  multi-ID tests recover identity without collage.
- Missing evidence: independent support-gate ablation, more identities, and a
  structure-sensitive metric are still required.

## Decision

Keep:

- post-scheduler, next-sigma inner optimization;
- positive/mismatched-reference same-state queries;
- full least-squares projection as the current safe default;
- pure-base-only scheduler contract.

Reject:

- pre-step inner optimization;
- exact positive-minus-negative subtraction;
- global partial projection;
- subject gates derived only from the same residual energy.

The next defensible iteration is an **independent subject-support gate**:
derive target-region support from the pure base trajectory (attention,
segmentation, or a separately frozen subject localizer), then allow a small
amount of the projected common component back only inside that support. This
must be tested on multiple IDs because u1839 already succeeds without it while
u4229 exposes the missing shared-structure problem.

## Support-gate and local trust-region follow-up

The independent-support hypothesis was tested on the same u4229 prompt and
seed with two base-only support sources:

1. a score-map gate from the frozen base full-prompt prediction minus a matched
   empty-scene prediction;
2. a token-attention gate from image queries to the `character` prompt token,
   averaged over layers 16-25.

The gated direction is

\[
d_{\rm in}=d_{\perp}+\kappa(r_+-d_{\perp}),\qquad
d=M d_{\rm in}+\lambda_{\rm out}(1-M)d_{\perp}.
\]

The score-map gate failed at CCIP 0.361155. The DiT output difference is
already globally mixed by self-attention and is not a segmentation map. The
token gate did localize the generated person, but concentrating the global RMS
budget into a support covering roughly 0.32 of the grid produced local
skeleton-like anatomy. This follows directly from sparse support: a fixed
global RMS budget permits characteristic in-support amplitude to grow as
approximately \(1/\sqrt{m}\).

The optimizer was therefore extended from one global RMS ball to the
intersection of global and per-location channel-RMS constraints:

\[
\operatorname{RMS}(\Delta z)\le\rho_g,\qquad
\sqrt{\frac1C\sum_c \Delta z_{c,h,w}^2}\le\rho_l
\quad\forall(h,w).
\]

With \(\rho_g=0.010\) and \(\rho_l=0.008\), the token-gated portrait no longer
showed skeleton collapse and CCIP improved from 0.293041 to 0.259859. The
identity still failed the 0.178 SAME threshold. A teacher-residual gate with a
36% mean support reached 0.256248 and was also structurally stable, but
generated the wrong identity.

| Follow-up arm | CCIP | Visual result |
|---|---:|---|
| Residual gate + global trust | 0.256248 | Stable, wrong identity |
| Base score-map subject gate | 0.361155 | Stable silhouette, regressed |
| Token gate + global trust | 0.293041 | Local skeleton collapse |
| Token gate + global-local trust | 0.259859 | Stable, target ID absent |

New outputs:

- `out/true_score_tsd/u4229_vid_gated10q90p2_trust_r010_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_subjectgate_k1o0q90p2_trust_r010_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_tokengate_k1o0q85p1_l16_25_trust_r010_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_tokengate_k1o0q85p1_l16_25_trust_g010l008_lr3e2_e95_s0_n20_i5.png`

### Updated interpretation

Support and local-radius controls solve real failure modes but do not solve the
remaining representation problem. The next iteration should not be another
gate-threshold sweep. It should estimate separate identity and layout
subspaces before spatial routing, for example from cross-reference identity
consensus plus within-reference layout perturbations. The acceptance test is
strict: improve u4229 identity while retaining the local-trust stability, then
repeat on at least two additional identities.

## Representation-space separation follow-up

Two separation locations were tested under the same global-local trust region.

### Output-score nuisance subspace

For positive residuals \(p_i\) and negative residuals \(n_j\), a nuisance basis
was built from the negative mean, centered negative variation, and centered
positive-view variation. The positive mean was spectrally projected out of
that basis. A removed component was restored only where positive references
agreed after negative centering and were discriminative relative to the
positive/negative response magnitude.

The rank-4 arm was structurally stable but reached CCIP 0.419012. It removed
0.760 of the positive residual RMS on average and rescued only 0.129. Reducing
the basis to rank 1 and increasing consensus rescue to 1.0 still regressed to
0.434786. This falsifies the hypothesis that the failure was merely caused by
an over-large selected rank.

### Teacher K/V value projection

Separation was then moved before the frozen DiT mixed reference and target
tokens. At every injected layer, positive-reference keys were averaged and
retained for content addressing. The mismatched-reference component was
least-squares projected from the positive-reference values:

\[
V_{\rm id}=V_+ - \alpha
\frac{\langle V_+,V_-\rangle}{\lVert V_-\rVert^2}V_-.
\]

This changed only about 6% of value RMS on average but reached CCIP 0.371235,
worse than the unprojected raw teacher under the same trust region.

### Missing raw-teacher control and radius boundary

The previously missing control combined the raw positive teacher residual with
global-local trust directly. It remained coherent at
\((\rho_g,\rho_l)=(0.010,0.008)\) and reached 0.309989. Expanding the radii to
\((0.015,0.012)\) increased the mean latent update from 0.00290 to 0.00481 but
worsened CCIP to 0.390109. Identity did not emerge before the trajectory moved
away from the reference metric.

| Arm | CCIP | Interpretation |
|---|---:|---|
| Raw teacher + trust 0.010/0.008 | 0.309989 | Stable, identity absent |
| Raw teacher + trust 0.015/0.012 | 0.390109 | Larger update, worse identity |
| Score subspace rank 4 | 0.419012 | Severe identity over-removal |
| Score subspace rank 1 | 0.434786 | Rank reduction does not rescue |
| K/V value projection | 0.371235 | Earlier linear separation also fails |

New outputs:

- `out/true_score_tsd/u4229_rawteacher_trust_g010l008_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_rawteacher_trust_g015l012_lr5e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_subspace_r4ridge05_rescue05_trust_g010l008_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_subspace_r1ridge05_rescue10_trust_g010l008_lr3e2_e95_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_kvvalue_project10_trust_g010l008_lr3e2_e95_s0_n20_i5.png`

### Decision after follow-up

Stop linear separation of the current teacher in output-score and K/V value
space. The negative controls show that projection consistently removes more
identity than nuisance. Further progress requires an explicit identity-aware
representation, such as a frozen character encoder loss on decoded
\(\hat x_0\), or a teacher trained to separate identity from pose and layout.
The trust-region optimizer remains useful as a safety mechanism around that
new identity objective.

## Verification

- `probe/test_110_identity_residual_tsd.py`
- `probe/test_112_true_score_tsd.py`
- `probe/test_113_trust_region_score_tsd.py`
- 29 tests pass.

## Explicit decoded-identity objective follow-up

The linear teacher decompositions above were replaced by an explicit frozen
CCIP-Torch identity objective on the decoded clean-image estimate:

\[
\mathcal L =
\lambda_{\rm score}\mathcal L_{\rm TSD}
+\lambda_{\rm id}
\left[1-\cos\left(E(D(\hat x_0)),\bar e_{\rm ref}\right)\right]
+\lambda_z\lVert z-z_{\rm anchor}\rVert_1 .
\]

The local CCIP port was first checked independently. On the stable u4229
output, a pixel-space step of RMS \(1/255\) lowered its differentiable feature
loss from 0.528880 to 0.288918 with finite gradients.

Applying the identity loss inside the low-sigma denoising loop failed:
although every inner loop lowered its local identity loss, the following
frozen-base scheduler step erased the change and the final CCIP regressed to
0.362948. A terminal identity projection after the final scheduler step
therefore became the controlled mechanism.

The successful strict control exactly preserves the previous residual-gate
trajectory (`active_start=0.05`) and changes only the terminal projection. Four
terminal steps use learning rate 0.0003 inside the global latent trust region.
Because a small latent RMS change can be large under the VAE Jacobian, a
decoded-pixel trust penalty was added:

\[
\lambda_{\rm px}
\lVert D(z)-D(z_{\rm terminal\ anchor})\rVert_1 .
\]

| Arm | CCIP | Pixel L1 vs stable baseline | Visual result |
|---|---:|---:|---|
| Residual gate + global trust | 0.256248 | 0 | Stable, wrong identity |
| CCIP inside low-sigma loop | 0.362948 | not matched | Frozen-base overwrite |
| Terminal CCIP, mismatched trajectory | 0.222035 | 0.110812 | Metric improves, face collapses |
| Terminal CCIP + decoded trust, strict control | **0.190013** | **0.002009** | Structure preserved |

The terminal objective lowered its internal cosine loss monotonically from
0.513671 to 0.392267 while the terminal latent moved by RMS 0.002846. Final
CCIP improved by 0.066234 over the stable baseline, but remains 0.012013 above
the 0.178 SAME threshold.

### Interpretation and boundary

This is evidence that an explicit identity representation supplies a useful
gradient that the score teacher did not isolate. It is not yet evidence of
visually convincing identity transfer: the accepted image is almost unchanged
at full-frame scale, and the subject's face is small. The large metric movement
under a pixel L1 change of only 0.002009 may partly reflect movement near the
CCIP decision boundary. Do not claim successful ID injection until a close-up
protocol shows recognizable target traits and passes an independent visual
identity judge.

New accepted output:

- `out/true_score_tsd/u4229_vid_gated_idccip_terminalonly4_tlr3e4_px05_trust_g010_a05e95_s0_n20_i5.png`

### Visual audit

A same-crop image plate compared the reference, stable baseline, terminal
result, matched face crops, and an absolute RGB difference map. The terminal
result preserves the baseline composition, but it does not visibly acquire the
reference character's spiked hair silhouette, long frontal lock, angular manga
face, earrings, or coat details. Changes are low-amplitude texture and
illumination shifts around the existing face and umbrella rather than semantic
identity traits.

Because the optimized identity objective and reported CCIP metric use the same
encoder family, the 0.066234 gain is not independent evidence. The visual audit
therefore classifies this arm as **metric improvement without demonstrated
visible identity recovery**.

Audit artifact:

- `visualization/identity_tsd_gallery/audit/identity_terminal_visual_audit.png`

## Distillation-loss redesign: independent patch features and structural TSD

The visual audit above showed that terminal CCIP optimization changed fine
texture without recovering the target's spiked hair or facial structure. Three
follow-up loss mechanisms were implemented and tested on the exact same u4229
prompt, seed, scheduler, and stable score-TSD trajectory.

### 1. ROI multi-encoder objective

The terminal identity objective was extended to

\[
\mathcal L_{\rm id} =
0.03\mathcal L_{\rm CCIP}^{\rm ROI}
+0.05\mathcal L_{\rm DINO\text{-}CLS}^{\rm ROI}
+0.1\mathcal L_{\rm pixel}
+0.5\mathcal L_{\rm outside\ ROI}.
\]

Both independently computed ROI losses decreased, but the face crop remained
visually almost identical to the baseline. Full-frame CCIP moved only from
0.256248 to 0.246020. This arm is retained as a negative control: independent
global embeddings reduce circularity, but still permit a fine-texture shortcut.

### 2. DINO spatial patch loss

A bidirectional differentiable soft-Chamfer loss was added over normalized
DINOv2 patch tokens:

\[
\mathcal L_{\rm patch} =
\frac12\left[
\frac1N\sum_i {\rm softmin}_j(1-p_i^\top q_j)
+\frac1M\sum_j {\rm softmin}_i(1-p_i^\top q_j)
\right].
\]

At the stable baseline, the unweighted pixel-gradient RMS values were
0.000573 for ROI CCIP, 0.000414 for DINO CLS, and 0.000354 for DINO patches.
Patch and DINO-CLS gradients were aligned (cosine 0.730), while patch and CCIP
were nearly orthogonal (cosine 0.018).

Two mechanisms were evaluated:

- post-step patch optimization at low sigma plus two terminal iterations;
- a trajectory-consistent look-ahead velocity correction. For
  \(z_{k+1}=z_k+(\sigma_{k+1}-\sigma_k)v_k\), the correction
  \[
  \Delta v_k =
  \frac{\rho_{\rm id}}{\sigma_k-\sigma_{k+1}}
  \frac{\nabla_{\hat x_0}\mathcal L_{\rm id}}
       {{\rm RMS}(\nabla_{\hat x_0}\mathcal L_{\rm id})}
  \]
  gives the next latent an exact RMS-\(\rho_{\rm id}\) descent displacement.

The look-ahead formulation fixed the scheduler-overwrite bug and applied
identity corrections at sigma 0.68, 0.62, 0.53, 0.41, and 0.25. Nevertheless,
the cross-domain manga-to-photo patch objective produced a hat-brim-like head
artifact and worsened external CCIP. Patch matching was therefore rejected as
the main identity loss.

### 3. ROI low-frequency multi-scale pseudo-target loss

The explicit image encoders were then removed. The original detached
teacher-base score pseudo-target was optimized with a 75% pooled structural
term (4x4 latent pooling), 25% native-resolution term, ROI weight 1.0, and
outside-ROI weight 0.05. This is the most direct change to the score-distillation
loss itself.

The result collapsed into a dark person silhouette and reached CCIP 0.409551.
Low-pass filtering did not uncover identity structure; it amplified the
teacher residual's pose, lighting, and composition components. This falsifies
the hypothesis that the current residual contains a clean identity direction
that is merely hidden by high-frequency noise.

| Arm | External CCIP | Full-image L1 vs stable | Decision |
|---|---:|---:|---|
| Stable residual-gate TSD | 0.256248 | 0 | Keep as stable baseline |
| ROI CCIP + DINO CLS terminal | 0.246020 | 0.002462 | Metric gain, no visible ID |
| DINO patch post-step + terminal | 0.271843 | 0.006092 | Reject |
| DINO patch look-ahead trajectory | 0.292234 | 0.013785 | Reject |
| ROI low-frequency structural TSD | 0.409551 | 0.152328 | Reject: collapse |

### Revised conclusion

Changing the loss alone does not make this pipeline SOTA. The trustworthy
finding is stronger than another hyperparameter result: the current
reference-K/V teacher score residual is not identity-pure, and generic
cross-domain embedding losses are exploitable at this small face scale.
Further weight or iteration sweeps are not justified.

The next viable mechanism should change the identity information source:
train or adopt an identity-conditioned diffusion adapter/teacher whose
conditioning survives every scheduler step, then use the existing trust-region
machinery only as a safety constraint. A close-up protocol is required before
returning to three-quarter-body evaluation.

New rejected outputs:

- `out/true_score_tsd/u4229_vid_gated_roi_patch_c015_d020_p050_smax40_term2_g010_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_gated_roi_patchtraj_c015_d020_p050_smax70_r004_g010_s0_n20_i5.png`
- `out/true_score_tsd/u4229_vid_gated_multiscale_roi_lp4w75_out05_g010_s0_n20_i5.png`

Verification:

- `probe/test_112_true_score_tsd.py`
- `probe/test_113_trust_region_score_tsd.py`
- 25 tests pass.

## Pareto-constrained identity distillation

The scalarized objective was replaced by an explicit multi-objective update.
For identity encoder gradients \(g_1,\ldots,g_m\), each gradient is first
RMS-normalized and the minimum-norm convex combination is computed:

\[
w^\star =
\arg\min_{w\in\Delta_m}
\left\lVert
\sum_i w_i\frac{g_i}{\operatorname{RMS}(g_i)}
\right\rVert_2^2,
\qquad
g_{\rm id} =
\overline{\operatorname{RMS}(g_i)}
\sum_i w_i^\star\frac{g_i}{\operatorname{RMS}(g_i)}.
\]

This prevents CCIP or DINO from winning only because its numerical gradient
scale is larger. For protected score/content gradients \(h_j\), the identity
gradient is projected onto the feasible cone

\[
h_j^\top g_{\rm id}\geq 0,
\]

so the actual descent step \(-g_{\rm id}\) cannot increase those losses to
first order.

### Failure found in the first constrained formulation

The first formulation still added the raw teacher-score gradient after
projecting the identity update. On the generic prompt it produced the same
dark distant-person failure family and reached CCIP 0.351151. The mathematical
error was structural: constraining the identity correction does not make an
unvalidated teacher update identity-specific.

### Local identity-aligned score projection

The corrected distillation loss uses the teacher only after local identity
validation. At every latent spatial position \(p\), with channel vectors
\(g_{{\rm score},p}\) and \(g_{{\rm id},p}\),

\[
\alpha_p =
\operatorname{clip}_{[0,\alpha_{\max}]}
\frac{
g_{{\rm score},p}^{\top}g_{{\rm id},p}
}{
\lVert g_{{\rm id},p}\rVert_2^2+\epsilon
},
\qquad
\widetilde g_{{\rm score},p} =
\alpha_p g_{{\rm id},p}.
\]

Negative coefficients are removed because score and identity disagree.
Orthogonal components are removed because they carry no encoder-validated
identity evidence. Global/local RMS trust regions are applied after this
directional filter.

| Arm | External CCIP | Mean retained score RMS | Visual decision |
|---|---:|---:|---|
| Generic matched pure T2I | 0.341806 | n/a | Distant silhouette, no ID |
| Generic Pareto constrained, raw score retained | 0.351151 | 100% raw score path | Reject |
| Generic local aligned distill | 0.350491 | 18.5% | Safe filtering, no recoverable ID |
| Capsule matched pure T2I | 0.122075 | n/a | Strong visible identity baseline |
| Capsule Pareto constrained | **0.117138** | raw score retained | Small metric gain, no new visible trait |
| Capsule local aligned distill | 0.122057 | 16.5% | Structure preserved; no extra visual ID |

The generic matched control is essential: its distant silhouette exists before
distillation. The aligned loss suppresses most teacher energy but cannot invent
identity information that the teacher gradient does not contain. With the
capsule, the generated image visibly contains the reference character's spiky
light hair, two long frontal strands, large hoop earrings, and buckle choker.
The aligned distillation safely preserves that result but adds no independently
visible trait.

### Decision

Keep `pareto_aligned_distill` as a safety filter, not as the identity source.
The load-bearing result is now mathematical rather than a weight sweep:
teacher score and multi-encoder identity gradients are globally nearly
orthogonal, and 81.5-83.5% of teacher score RMS is rejected by local positive
alignment. An identity-conditioned diffusion adapter or teacher is still
required for reference-only injection. The capsule path is the current
production-quality fallback.

New outputs:

- `out/true_score_tsd/u4229_generic_puret2i_matched_s0_n20_i3.png`
- `out/true_score_tsd/u4229_generic_pareto_constrained_late_g006l004_s0_n20_i3.png`
- `out/true_score_tsd/u4229_generic_pareto_aligned_distill_g006l004_s0_n20_i3.png`
- `out/true_score_tsd/u4229_capsule_puret2i_matched_s0_n20_i5.png`
- `out/true_score_tsd/u4229_capsule_pareto_constrained_late_g006l004_s0_n20_i3.png`
- `out/true_score_tsd/u4229_capsule_pareto_aligned_distill_g006l004_s0_n20_i3.png`

Verification:

- `probe/test_112_true_score_tsd.py`
- `probe/test_113_trust_region_score_tsd.py`
- 31 tests pass.
