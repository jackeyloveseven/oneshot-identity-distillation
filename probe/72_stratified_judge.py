#!/usr/bin/env python
"""
72_stratified_judge.py -- stratified defect judge for the content-leakage
hypothesis (2026-07-19). Measurement instrument for the "orig ref" vs
"matted ref" A/B (probe/71_bg_isolate.py produces the matted refs; this
script judges the GENERATIONS, not the refs).

Why this exists (do not collapse this back into a single visible_defect
rate): the production defect-aware judge (probe/38_gemini_judge.py,
HOLDOUT_PROMPT) only asks yes/no "visible_defect" and lumps every failure
mode together. Our hypothesis is specifically that PiP/collage defects are
CONTENT LEAKAGE -- the reference's background/composition getting copied
wholesale into the output -- which background-matting the ref should fix.
"limb_geometry" defects (malformed hands/anatomy) are a DIFFERENT failure
mode that has nothing to do with leakage and will NOT move when we matte
the ref; mixing them into one rate would dilute or hide the leakage signal.
So every judged image gets a same_character verdict AND a defect_type in
{pip_collage, limb_geometry, other, none}, with pip_collage as the primary
metric for the A/B.

Reuses the debugged Kaon/Gemini integration from probe/38_gemini_judge.py
verbatim (endpoint, auth, key-from-file, datauri encoding, retry/backoff,
"raise on bad JSON so the retry loop actually engages" fix from 2026-07-07)
-- only the prompt/schema and dataset wiring are new here. Adds a schema/
enum consistency check (visible_defect must agree with defect_type) that
ALSO raises-to-retry, since a self-contradictory verdict from the model is
exactly the kind of silent-corruption failure the 38 bugfix was about.

Identity is always judged against the ORIGINAL reference image (never the
matted one) regardless of --set -- matting is a generation-time
intervention, not a ground-truth change; see task spec.

CLI:
  $PY 72_stratified_judge.py --set holdout_orig   --arm old
  $PY 72_stratified_judge.py --set holdout_orig   --arm new
  $PY 72_stratified_judge.py --set holdout_matted --arm old   # once matted gens exist
  $PY 72_stratified_judge.py --set holdout_matted --arm new
  $PY 72_stratified_judge.py --set holdout_orig --arm old --limit 2   # smoke test

Idempotent: re-running reuses any prior verdict in the output JSON that
parsed with a valid schema; only missing/error entries get re-judged.

Run: source ../env.sh && $PY 72_stratified_judge.py --set holdout_orig --arm old
"""
import os, json, base64, io, argparse, time, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
KAON_BASE = "https://kaon-router.kaonai.com/v1"
PRO = "google/gemini-3.1-pro-preview"
KEY_SRC = "/home/ubuntu/smoke_anime/flash_caption_smoke.py"

# Do NOT rely on the caller's $REF env var: env.sh currently exports
# REF=/home/ubuntu/smoke_anime/revref_30k/img, which is STALE (that dir is
# empty -- verified 2026-07-19). The real holdout ref images live at
# anime_260623. So this script ignores $REF entirely and hardcodes the
# correct default, with a --ref-dir CLI escape hatch for whoever fixes
# env.sh later or needs a different ref set.
REF_DEFAULT = "/mnt/local/smoke_anime/anime_260623/img"

GEN_DIRS = {
    "holdout_orig":   "/mnt/local/cherry_out/holdout30",
    "holdout_matted": "/mnt/local/cherry_out/holdout30_matted",
}

SAME_CHAR_VALUES = ("yes", "partial", "no")
VISIBLE_DEFECT_VALUES = ("yes", "no")
DEFECT_TYPE_VALUES = ("pip_collage", "limb_geometry", "other", "none")

# NOTE on prompt shape (2026-07-19 smoke test finding): an earlier, longer
# version of this prompt (full failure-mode taxonomy spelled out BEFORE the
# "think step by step" instruction, 4-item CoT checklist, "reply with ONLY
# compact JSON (no markdown fence)") reliably made gemini-3.1-pro-preview
# skip the reasoning entirely and emit just the JSON -- verified on 6 live
# calls, 0/6 had any reasoning text. Confirmed via a side-by-side test that
# 38_gemini_judge.py's HOLDOUT_PROMPT (shorter, 2-item CoT) DOES reliably
# elicit reasoning on the exact same image pair. The version below keeps the
# taxonomy compact, explicitly says "You MUST reason before answering... Do
# not skip this reasoning paragraph", and requires a literal "JSON:" line
# marker before the payload (a concrete phase-boundary instead of a vague
# "on a new line") -- this combination produced full CoT reasoning on every
# test call (2/2 then re-confirmed against the earlier two spot-checks).
PROMPT = (
    'Image 1 is a REFERENCE anime character portrait. Image 2 is a GENERATED portrait '
    'meant to depict the same character. Judge (A) same_character (identity only: face '
    'shape, hair color, hair length/style, eye color, skin tone, gender, distinctive '
    'marks -- ignore pose/outfit/expression/background/art-style), and (B) whether '
    'Image 2 has a visible rendering defect, classified into exactly one of: '
    '"pip_collage" (the reference\'s background/composition, or a framed/tiled/paneled '
    'copy of it, is visibly copied into Image 2 as a distinct chunk -- a content-leakage '
    'failure), "limb_geometry" (malformed/extra/missing/fused hands, arms, legs, or '
    'other anatomy -- unrelated to leakage), "other" (watermark, text, logo, blur, or '
    'other artifact), or "none" (no defect; in this case visible_defect must be "no"). '
    'If more than one applies, pick the single most dominant one as defect_type but '
    'mention the rest in notes.\n\n'
    'You MUST reason before answering. Write a reasoning paragraph that goes through, '
    'IN ORDER: (1) each identity axis one at a time with your observation in each image '
    'and whether it matches; (2) Image 2\'s overall composition -- is it a single '
    'coherent picture, or does any region reproduce the reference\'s background/framing '
    'as a separate panel/tile/border?; (3) Image 2\'s hand/limb/anatomy correctness; '
    '(4) any other artifacts (watermark/text/blur). Do not skip this reasoning '
    'paragraph.\n\n'
    'After the reasoning paragraph, on a new line write exactly "JSON:" followed by '
    'ONLY compact JSON: {"same_character":"yes|partial|no","visible_defect":"yes|no",'
    '"defect_type":"pip_collage|limb_geometry|other|none","notes":"<=20 words"}'
)


def _key(var):
    for l in open(KEY_SRC):
        if l.startswith(var + "="):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in {KEY_SRC}")


def _datauri(p, px=768, q=88):
    im = Image.open(p).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _extract_json(txt):
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        # Same fix as 38_gemini_judge.py (2026-07-07): raise, don't return an
        # error dict directly -- an error dict here would bypass judge()'s
        # retry loop (only network/HTTP-status failures would retry).
        raise ValueError(f"no JSON object found in response (len={len(txt)}): {txt[:120]!r}")
    return json.loads(m.group(0))


def _validate(v):
    """Enum + cross-field consistency check. Raises ValueError (-> retry) on
    any violation instead of silently accepting a malformed/self-contradictory
    verdict -- this is a measurement instrument for a specific hypothesis test,
    a bad label here corrupts the pip_collage count we actually care about."""
    sc = v.get("same_character")
    vd = v.get("visible_defect")
    dt = v.get("defect_type")
    if sc not in SAME_CHAR_VALUES:
        raise ValueError(f"same_character invalid: {sc!r}")
    if vd not in VISIBLE_DEFECT_VALUES:
        raise ValueError(f"visible_defect invalid: {vd!r}")
    if dt not in DEFECT_TYPE_VALUES:
        raise ValueError(f"defect_type invalid: {dt!r}")
    if vd == "no" and dt != "none":
        raise ValueError(f"inconsistent: visible_defect=no but defect_type={dt!r}")
    if vd == "yes" and dt == "none":
        raise ValueError(f"inconsistent: visible_defect=yes but defect_type=none")
    if "notes" not in v:
        v["notes"] = ""
    return v


def judge(ref, tgt, key, retries=4, prompt=None):
    body = {"model": PRO, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt or PROMPT},
        {"type": "image_url", "image_url": {"url": _datauri(ref)}},
        {"type": "image_url", "image_url": {"url": _datauri(tgt)}}]}],
        "max_tokens": 1300, "temperature": 0.1}
    last = ""
    for a in range(retries + 1):
        try:
            r = requests.post(KAON_BASE + "/chat/completions",
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                               json=body, timeout=180)
            if r.status_code in (408, 429) or r.status_code >= 500:
                last = f"http {r.status_code}: {r.text[:200]}"
                time.sleep(3 * (a + 1))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            verdict = _extract_json(content)
            verdict = _validate(verdict)
            verdict["_reasoning"] = content  # full CoT trace, kept for spot-checking
            return verdict
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:200]
            time.sleep(2 * (a + 1))
    return {"same_character": "error", "visible_defect": "error", "defect_type": "error",
            "notes": last}


def build_pairs(set_name, arm, ref_dir):
    tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))
    uids = [u for u in tags if not u.startswith("_")]
    gen_dir = GEN_DIRS[set_name]
    pairs, missing = [], []
    for uid in uids:
        ref = f"{ref_dir}/u{uid}_ref.png"
        for s in (0, 1):
            tgt = f"{gen_dir}/u{uid}_holdout_{arm}_s{s}.png"
            name = f"u{uid}_s{s}"
            if os.path.exists(tgt) and os.path.exists(ref):
                pairs.append((name, ref, tgt))
            else:
                reasons = []
                if not os.path.exists(ref):
                    reasons.append("ref missing")
                if not os.path.exists(tgt):
                    reasons.append("gen missing")
                missing.append((name, tgt, ", ".join(reasons)))
    return pairs, missing


def _is_valid_prior(v):
    return (isinstance(v, dict)
            and v.get("same_character") in SAME_CHAR_VALUES
            and v.get("visible_defect") in VISIBLE_DEFECT_VALUES
            and v.get("defect_type") in DEFECT_TYPE_VALUES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["holdout_orig", "holdout_matted"])
    ap.add_argument("--arm", required=True, choices=["old", "new"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                     help="only judge the first N pairs (smoke test)")
    ap.add_argument("--ref-dir", default=REF_DEFAULT,
                     help="dir containing u{uid}_ref.png (identity ground truth, "
                          "always the ORIGINAL i.e. non-matted ref); "
                          f"default={REF_DEFAULT}, ignores $REF env var")
    a = ap.parse_args()
    key = _key("KAON_KEY")

    pairs, missing = build_pairs(a.set, a.arm, a.ref_dir)
    if missing:
        print(f"[72] WARNING: {len(missing)} expected image(s) missing, skipped:", flush=True)
        for name, tgt, reason in missing[:20]:
            print(f"    {name}: {tgt}  ({reason})", flush=True)
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more", flush=True)

    if a.limit:
        pairs = pairs[:a.limit]

    outp = os.path.join(DATA, f"stratified_judge_{a.set}_{a.arm}.json")
    existing_items = {}
    if os.path.exists(outp):
        try:
            existing_items = json.load(open(outp)).get("items", {})
        except Exception:
            existing_items = {}

    reused, todo = {}, []
    for name, ref, tgt in pairs:
        prior = existing_items.get(name)
        if _is_valid_prior(prior):
            reused[name] = prior
        else:
            todo.append((name, ref, tgt))

    print(f"[72] set={a.set} arm={a.arm}: {len(pairs)} pairs to consider "
          f"({len(reused)} reused, {len(todo)} to judge), model={PRO}", flush=True)

    def work(item):
        name, ref, tgt = item
        return name, judge(ref, tgt, key, retries=a.retries)

    results = dict(reused)
    n_error = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, item) for item in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, res = f.result()
            results[name] = res
            if res.get("same_character") == "error":
                n_error += 1
            print(f"  [{i}/{len(todo)}] {name}: same_character={res.get('same_character')} "
                  f"visible_defect={res.get('visible_defect')} defect_type={res.get('defect_type')} "
                  f"notes={res.get('notes', '')!r}", flush=True)

    ordered = {name: results[name] for name, _, _ in pairs if name in results}

    c_char = Counter(v.get("same_character") for v in ordered.values())
    c_def = Counter(v.get("visible_defect") for v in ordered.values())
    c_type = Counter(v.get("defect_type") for v in ordered.values())

    summary = {
        "n_judged": len(ordered),
        "n_missing_images": len(missing),
        "n_api_errors": n_error,
        "same_character": {k: c_char.get(k, 0) for k in (*SAME_CHAR_VALUES, "error")},
        "visible_defect": {k: c_def.get(k, 0) for k in (*VISIBLE_DEFECT_VALUES, "error")},
        "defect_type": {k: c_type.get(k, 0) for k in (*DEFECT_TYPE_VALUES, "error")},
    }

    out = {
        "_meta": {"set": a.set, "arm": a.arm, "ref_dir": a.ref_dir, "gen_dir": GEN_DIRS[a.set],
                  "model": PRO, "n_expected_pairs": len(pairs) + len(missing)},
        "items": ordered,
        "summary": summary,
    }
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print(f"\n=== set={a.set} arm={a.arm} (n={summary['n_judged']}) ===")
    for k in (*SAME_CHAR_VALUES, "error"):
        print(f"  same_character={k:8s} {c_char.get(k, 0)}")
    print(f"  visible_defect=yes  {c_def.get('yes', 0)}")
    print(f"  visible_defect=no   {c_def.get('no', 0)}")
    if c_def.get("error", 0):
        print(f"  visible_defect=error {c_def.get('error', 0)}")
    print("  --- defect_type breakdown (pip_collage = primary content-leakage metric) ---")
    for k in (*DEFECT_TYPE_VALUES, "error"):
        print(f"  defect_type={k:14s} {c_type.get(k, 0)}")
    if missing:
        print(f"  (skipped {len(missing)} missing image(s), see WARNING above)")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
