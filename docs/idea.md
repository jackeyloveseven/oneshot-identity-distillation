# Distilling a reference-conditioned image editor into a frozen text-only generator at test time

## Motivation

One-shot identity personalization — reproducing the specific subject in a single reference image (face geometry, hair-strand arrangement, small accessories, and their joint style) rather than just its coarse attributes — currently demands one of three things from the *generator*: per-subject optimization (DreamBooth), a trained identity adapter (InstantID, PhotoMaker, IP-Adapter), or direct feature/token injection into the denoising network. All three assume you can *change or interface with* the deployed generator.

But the strongest text-to-image generators are often shipped frozen and text-only: no personalization interface, no place to inject a reference, no license/ability to fine-tune. Meanwhile, reference-conditioned image *editors* that can consume a reference do exist for the same model family. This exposes an unaddressed gap: **can a reference-conditioned editor transfer one-shot identity into a frozen, text-only generator at test time, without updating either model and without the editor becoming the generator's sampler?** A naive answer — just run the editor — fails the setting: the editor's own trajectory advances the sample (the editor *becomes* the generator), and it can stay coupled to the reference's composition. The open problem is to extract only the editor's *conditional identity information* and inject it into the text-only generator's own trajectory as guidance, not as the sampler.

## Method

Test-time conditional score distillation (TSD) for flow-based generators:

1. **Same-state query.** At the same noisy latent `z_σ` and time `σ`, query two frozen models: an attribute-matched text-only base (velocity `v_A`) and a reference-conditioned edit teacher (`v_R`). Because the state is shared, the common text-only field cancels in the difference.
2. **Closed-form conditional score.** Convert the velocity difference to a conditional score direction via the linear-flow identity: `Δs_R = -((1-σ)/σ)·(v_R - v_A)`. This is a *difference between conditions* (reference-vs-text), not an unconditional teacher score — which is what makes reference-swap and matched-conditioning controls meaningful.
3. **Detached pseudo-target latent update.** Take a few (3) inner latent updates along the normalized score direction, re-evaluating both fields at the updated state each time (so the step is nonlinear), with both model predictions stop-gradiented.
4. **Adaptive two-level trust region.** Project every update into intersecting global and tokenwise RMS trust regions around the post-scheduler anchor, with the radius adapting to the teacher's one-Euler-step discrepancy — allowing strong identity motion where the two conditional fields disagree while bounding trajectory drift.
5. **Text-only scheduling.** Only the text-only base advances the scheduler; the editor supplies a transient conditional-score difference and never advances the sample. Nothing (weights, optimizer state, per-subject token) is retained after sampling.

The design separates *teacher information* from *teacher integration*: the editor informs the direction, the frozen text-only model owns the trajectory. It also yields a built-in causal test — swapping only the reference (holding text fixed) must change the identity if the residual truly carries reference information.
