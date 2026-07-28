#!/usr/bin/env python3
"""Flatten the modular AAAI-27 paper into a single self-contained .tex.

The AAAI-27 AuthorKit ("What Files to Submit") requires the LaTeX source to be
a SINGLE .tex file: "do not use the input command to include sections of your
paper --- every section must be in the single source file. (The only allowable
exception is .bib file.)"

We keep the maintainable modular source (paper/main_aaai_tsd.tex + method_v3/*)
as the working copy and generate the compliant flattened archive from it. This
script is deterministic and re-runnable, so a later writing pass just re-flattens.

Output: paper/aaai27_submission/ containing exactly the files AAAI wants:
  - <flat>.tex          single source, every \\input inlined
  - references.bib
  - aaai2027.sty, aaai2027.bst
  - figures/*.pdf       only the figures actually included
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
MAIN = PAPER / "main_aaai_tsd.tex"
OUT_DIR = PAPER / "aaai27_submission"
FLAT_NAME = "anonymous_submission.tex"  # first-author family name unknown (anon)

INPUT_RE = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def resolve(rel: str) -> Path:
    p = PAPER / rel
    if p.suffix != ".tex":
        p = p.with_suffix(".tex")
    return p


def flatten(main: Path, inlined: list[str] | None = None) -> tuple[str, list[str]]:
    """Return the flattened source text and the list of inlined files.

    Recurses so that nested \\input (e.g. a tables file included from a section)
    is inlined too, leaving no \\input in the final single-file source.
    """
    if inlined is None:
        inlined = []
    lines_out: list[str] = []
    for line in main.read_text(encoding="utf-8").splitlines():
        m = INPUT_RE.match(line)
        if not m:
            lines_out.append(line)
            continue
        target = resolve(m.group(1))
        rel = str(target.relative_to(PAPER))
        inlined.append(rel)
        body, _ = flatten(target, inlined)  # recurse into nested \input
        lines_out.append(f"% ---- begin {rel} ----")
        lines_out.extend(body.rstrip("\n").splitlines())
        lines_out.append(f"% ---- end {rel} ----")
    return "\n".join(lines_out) + "\n", inlined


def main() -> int:
    flat, inlined = flatten(MAIN)

    # sanity: no \input survives in the flattened source (bib is \bibliography, fine)
    leftover = [ln for ln in flat.splitlines() if INPUT_RE.match(ln)]
    assert not leftover, f"un-inlined \\input remains: {leftover}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    # write the single flattened source
    (OUT_DIR / FLAT_NAME).write_text(flat, encoding="utf-8")

    # copy the exactly-required support files
    for name in ("references.bib", "aaai2027.sty", "aaai2027.bst"):
        shutil.copy2(PAPER / name, OUT_DIR / name)

    # copy only the figures actually included by \includegraphics
    figs = sorted(set(INCLUDE_RE.findall(flat)))
    for rel in figs:
        src = PAPER / rel
        if src.suffix == "":
            src = src.with_suffix(".pdf")
        dst = OUT_DIR / Path(rel)
        if dst.suffix == "":
            dst = dst.with_suffix(".pdf")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f"flattened -> {OUT_DIR / FLAT_NAME}")
    print(f"inlined {len(inlined)} section files:")
    for f in inlined:
        print(f"  + {f}")
    print(f"copied {len(figs)} figures:")
    for f in figs:
        print(f"  + {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
