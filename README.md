# Cherry: test-time identity distillation

Cherry studies whether a reference-conditioned image editor can transfer
identity into a text-only diffusion trajectory at inference time. The current
paper is:

> **One-Shot Identity Personalization by Distilling an Image Editor at Test
> Time**

The method queries a Krea 2 text model and a Krea 2 Identity Edit teacher at
the same noisy latent, converts their velocity difference into a conditional
score direction, and optimizes the latent inside adaptive global and tokenwise
trust regions. Only the text model advances the scheduler; neither model is
updated.

Updated: **2026-07-27 UTC**

## Current evidence

The locked evaluation contains 30 held-out identities, three scenes, and two
fixed seeds: 180 paired observations. The mechanism study contains 10
identities, three scenes, one seed, and 15 variants: 450 outputs.

| Result | Attribute text | Distilled | Direct teacher |
|---|---:|---:|---:|
| CCIP distance, lower is better | 0.1860 | **0.0659** | 0.0751 |
| 30-way identity Top-1 | 42.8% | **91.7%** | 88.3% |
| Blind-VLM identity score | 24.7% | **93.9%** | 88.9% |
| Blind-VLM visible-defect rate | 0.0% | 3.9% | 8.3% |

Relative to the attribute-text baseline, distillation changes scene CLIP by
-0.0090 and predicted aesthetics by -0.0651. The result supports
editor-to-generator conditional information transfer in this anime-domain
model family. It does **not** establish universal state of the art,
reference-only personalization, or a localized identity representation.

## Canonical artifacts

| Artifact | Path |
|---|---|
| Paper source | `paper/main_aaai_tsd.tex` |
| Compiled paper | `paper/main_aaai_tsd.pdf` |
| Paper sections | `paper/method_v3/` |
| Paper build and artifact guide | `paper/README.md` |
| Method and protocol record | `docs/research/KREA2_EDIT_TO_GENERATION_DISTILLATION.md` |
| Locked protocol | `data/krea2_eval30_protocol.json` |
| Reference preprocessing provenance | `data/krea2_eval30_reference_preprocessing.json` |
| Complete main visual audit | `out/krea2_tsd_eval30_tagged_v3/eval30_report.html` |
| Complete mechanism audit | `out/krea2_tsd_eval30_ablations_v3/ablation_report.html` |
| Release landing page | `identity_tsd_v3_index.html` |
| Release manifest | `release_manifest_v3.json` |

Public evaluation release:

<https://yata-image-publics.flowgpt.com/chengyanli/cherry/identity-tsd-v3-eval30/index.html>

The landing page links every reference ID, attribute-text input, distilled
output, teacher output, control, metric, and mechanism variant. Paper
qualitative examples are selected only from a declared qualified set; all 180
main observations remain visible in the report.

## Reproduce and validate

Run the method unit tests:

```bash
cd /home/ubuntu/chengyanli/items/cherry
/mnt/local/krea2_edit_serve/venv/bin/python -m pytest -q \
  probe/test_115_krea2_identity_tsd.py
```

Regenerate reports and figures from the frozen outputs:

```bash
python probe/123_krea2_eval30_report.py
python probe/125_krea2_eval30_ablation_report.py
python probe/127_make_main_figure.py
python probe/128_make_method_figure.py
python probe/129_make_ablation_figure.py
```

Build the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main_aaai_tsd.tex
bibtex main_aaai_tsd
pdflatex -interaction=nonstopmode -halt-on-error main_aaai_tsd.tex
pdflatex -interaction=nonstopmode -halt-on-error main_aaai_tsd.tex
```

The generation runners are resumable:

- `probe/121_run_krea2_eval30.py`
- `probe/124_run_krea2_eval30_ablations.py`

Do not rerun the locked protocol with changed prompts, references, seeds, or
hyperparameters under the same output directory.

## Directory policy

- `paper/`, the two `out/krea2_tsd_eval30_*_v3/` directories, and the two
  locked JSON files above define the current paper.
- `experiment-log/` contains dated audit records, not current instructions.
- Other `data/`, `out/`, and `probe/` files are retained experimental evidence.
  They may describe rejected Z-Image, K/V-injection, or Band-LoRA branches and
  must not override the canonical artifacts above.
- The project root is not a Git repository. Avoid deleting raw metrics or
  generated outputs when cleaning prose documentation.
