#!/usr/bin/env python3
"""Anonymous reviewer-style audit of the current AAAI paper via Kaon."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
KEY_SRC = Path("/home/ubuntu/smoke_anime/flash_caption_smoke.py")
BASE_URL = "https://kaon-router.kaonai.com/v1"
MODEL = "kaon/gemini-3-flash-preview"
OUT = ROOT / "data" / "gemini3flash_paper_blind_review.json"


def load_key() -> str:
    match = re.search(r"KAON_KEY\s*=\s*[\"']([^\"']+)", KEY_SRC.read_text())
    if not match:
        raise RuntimeError("KAON_KEY not found")
    return match.group(1)


def paper_text() -> str:
    main = (ROOT / "paper" / "main_aaai.tex").read_text()
    for name in ("00_intro", "05_related", "10_method", "20_results_trainingfree", "30_results_traininglight", "40_limits_conclusion"):
        main = main.replace(f"\\input{{aaai_draft/{name}}}", (ROOT / "paper" / "aaai_draft" / f"{name}.tex").read_text())
    main = re.sub(r"\\author\{.*?\}\\affiliations\{.*?\}", r"\\author{Anonymous Submission}", main, flags=re.S)
    return main


PROMPT = """Act as a strict anonymous AAAI reviewer. Review the supplied paper without guessing author identity or using external information.

Focus on:
1. Whether the title and abstract are exactly supported by the experiments.
2. Whether comparison experiments and ablations are sufficient for the claimed contribution.
3. Whether null results are phrased as failure to detect rather than equivalence.
4. Whether VLM evaluation is clearly distinguished from human annotation.
5. Internal numerical or logical contradictions.
6. The three highest remaining rejection risks and concrete text/experiment fixes.

Return ONLY valid JSON with keys:
{
  "overall_score_1_to_10": number,
  "recommendation": "accept|weak_accept|borderline|weak_reject|reject",
  "title_abstract": {"verdict":"pass|revise","issues":[string]},
  "comparisons": {"verdict":"pass|revise|new_experiment","issues":[string]},
  "ablations": {"verdict":"pass|revise|new_experiment","issues":[string]},
  "claim_evidence": [{"claim":string,"status":"supported|overstated|unclear","reason":string}],
  "remaining_risks": [{"severity":"high|medium|low","risk":string,"fix":string}],
  "strongest_point": string,
  "final_summary": string
}
"""


def main() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT + "\n\nPAPER:\n" + paper_text()}],
        "temperature": 0.1,
        "max_tokens": 5000,
    }
    response = requests.post(
        BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + load_key(), "Content-Type": "application/json"},
        json=body,
        timeout=600,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"review did not contain JSON: {text[:200]!r}")
    review = json.loads(match.group(0))
    payload = {"model": MODEL, "definition": "anonymous LLM paper review; not human peer review", "review": review}
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(review, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
