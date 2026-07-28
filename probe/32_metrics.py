#!/usr/bin/env python
"""
32_metrics.py — shared CCIP metric helpers + Wave-1 1e backfill CLI.

CCIP = character-level likeness (NOT pixel face geometry). diff < 0.178 => SAME.
Run under env.sh: HF_HOME=/mnt/local/hf_home already holds ccip_onnx + wd14-swinv2
(verified 2026-07-02), so HF_HUB_OFFLINE=1 is fine.

Backfill (1e): score every u{uid}_*.png in out/ against its ref:
    source ../env.sh && $PY 32_metrics.py --mode backfill
Writes data/ccip_backfill.json and prints a per-experiment markdown table.
"""
import os, re, json, argparse, itertools

REF = os.environ["REF"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")
SAME_THR = 0.178

_feat_cache = {}

def _feat(path):
    from imgutils.metrics import ccip_extract_feature
    if path not in _feat_cache:
        _feat_cache[path] = ccip_extract_feature(path)
    return _feat_cache[path]

def ccip_diff(a, b):
    from imgutils.metrics import ccip_difference
    return float(ccip_difference(_feat(a), _feat(b)))

def ref_path(uid):
    return f"{REF}/u{uid}_ref.png"

def pairwise_consistency(paths):
    """Mean pairwise CCIP diff among generations (lower = more seed-consistent)."""
    if len(paths) < 2:
        return None
    ds = [ccip_diff(a, b) for a, b in itertools.combinations(paths, 2)]
    return sum(ds) / len(ds)

def score_group(uid, paths):
    """Per subject-arm aggregate: per-image diff vs ref, SAME count, consistency."""
    diffs = {os.path.basename(p): ccip_diff(ref_path(uid), p) for p in sorted(paths)}
    same = sum(1 for d in diffs.values() if d < SAME_THR)
    return {"diffs": diffs, "n": len(diffs), "same": same,
            "mean": sum(diffs.values()) / len(diffs),
            "consistency": pairwise_consistency(sorted(paths))}

# ------------------------- 1e backfill -------------------------
FN_RE = re.compile(r"^u(\d+)_(.+)\.png$")

def backfill():
    os.makedirs(DATA, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(OUT)):
        m = FN_RE.match(fn)
        if not m or fn.startswith("montage"):
            continue
        uid, tag = m.groups()
        d = ccip_diff(ref_path(uid), os.path.join(OUT, fn))
        rows.append({"file": fn, "uid": uid, "tag": tag,
                     "ccip": round(d, 4), "same": d < SAME_THR})
        print(f"{fn:45s} ccip={d:.4f} {'SAME' if d < SAME_THR else 'DIFF'}", flush=True)
    out = os.path.join(DATA, "ccip_backfill.json")
    with open(out, "w") as f:
        json.dump({"threshold": SAME_THR, "rows": rows}, f, indent=1)
    print(f"\nwrote {out} ({len(rows)} images)")
    # per-experiment summary (experiment = tag prefix before first '_')
    print("\n| experiment | n | SAME | mean ccip |\n|---|---|---|---|")
    key = lambda r: r["tag"].split("_")[0]
    for exp, g in itertools.groupby(sorted(rows, key=key), key=key):
        g = list(g)
        print(f"| {exp} | {len(g)} | {sum(r['same'] for r in g)} |"
              f" {sum(r['ccip'] for r in g)/len(g):.4f} |")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="backfill", choices=["backfill"])
    ap.parse_args()
    backfill()
