# Data and metric records

This directory contains small, machine-readable experiment records. The
current paper uses the Krea2 eval30 v3 files listed below. Remaining files are
retained as historical audit evidence and are not current paper inputs.

## Current protocol

| File | Role |
|---|---|
| `krea2_eval30_protocol.json` | Frozen 30-identity split, prompts, scenes, seeds, output arms, wrong-reference derangement, and method settings |
| `krea2_eval30_reference_preprocessing.json` | Input-only crop boxes, detector confidence, source hashes, and prepared-reference hashes |

The primary derived records live with the generated outputs:

| File | Role |
|---|---|
| `../out/krea2_tsd_eval30_tagged_v3/eval30_metrics.json` | Aggregate and per-observation main metrics |
| `../out/krea2_tsd_eval30_tagged_v3/eval30_metrics.csv` | Flat 180-observation metric table |
| `../out/krea2_tsd_eval30_tagged_v3/gemini3flash_blind_summary.json` | Three-pass blind-VLM majority labels and agreement |
| `../out/krea2_tsd_eval30_tagged_v3/gemini3flash_blind_raw.json` | Per-pass blind-VLM judgments |
| `../out/krea2_tsd_eval30_tagged_v3/eval30_report.html` | Complete reference/input/output visual audit |
| `../out/krea2_tsd_eval30_ablations_v3/ablation_metrics.json` | Fifteen-condition mechanism results |
| `../out/krea2_tsd_eval30_ablations_v3/ablation_report.html` | Complete mechanism visual audit |

Paper plotting data are exported separately under `../paper/figures/`:

- `main_eval30_source_data.csv`
- `main_eval30_qualitative_selection.csv`
- `mechanism_ablation_source_data.csv`

## Interpretation rules

- CCIP is a distance; lower is better.
- The statistical unit is the identity. Scene and seed observations are nested
  inside 30 subject clusters.
- The VLM labels are automated audits, not human evaluation.
- The attribute-only baseline is separately encoded text with no reference
  tokens.
- A wrong-reference control changes the visual reference while holding the
  target's attribute text fixed.
- Raw JSON, CSV, and generated images take precedence over prose if a number
  disagrees.

## Historical records

Files mentioning `strip`, `lowfreq`, layer localization, head/token/time
gating, Band-LoRA, Z-Image, or older Gemini judging belong to superseded
experiments. They remain here only to preserve provenance. Do not cite them as
results of the current Krea2 distillation paper.

Do not overwrite judged outputs or frozen protocol files. A changed
instrument, prompt, model, seed, or metric definition requires a new filename
and output directory.
