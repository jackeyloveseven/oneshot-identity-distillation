#!/usr/bin/env python
"""
38_gemini_judge.py — third independent identity signal: Gemini 3.1 Pro (Kaon
router) looks at ref vs generated portrait directly and judges same-character
+ per-attribute correctness. Requested by user after the WithAnyone/InfiniteYou
research (docs/design §7) confirmed CCIP+wd14 ensemble is the right in-domain
approach but is still tag-vocabulary-bounded — a VLM judge adds a genuinely
independent, semantic-level check neither CCIP nor wd14 can give.

Reuses the working Kaon/Gemini pattern from
/home/ubuntu/cluster/improve/id_consistency/id_match/gemini_judge.py verbatim
(endpoint, auth, datauri encoding, retry/backoff) — do not re-derive this,
it's an already-debugged integration. Only the PROMPT and dataset wiring are
cherry-specific. Model pinned to gemini-3.1-pro-preview per direct user ask
(no flash-first escalation — small enough batch that cost isn't a concern).

Run: source ../env.sh && $PY 38_gemini_judge.py --set gen10
     $PY 38_gemini_judge.py --set bv2
"""
import os, json, base64, io, argparse, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
KAON_BASE = "https://kaon-router.kaonai.com/v1"
PRO = "google/gemini-3.1-pro-preview"
KEY_SRC = "/home/ubuntu/smoke_anime/flash_caption_smoke.py"

# CoT judging (2026-07-07): user flagged that a direct-to-JSON verdict is
# unreliable for a nuanced visual comparison task like this — the model can
# pattern-match to a verdict without actually working through each identity
# dimension. Both prompts below now require an explicit step-by-step
# reasoning trace BEFORE the JSON verdict (walk each identity axis one at a
# time, then separately reason about defects for HOLDOUT_PROMPT), following
# standard CoT-improves-VLM-judgment practice. max_tokens raised accordingly
# (judge()'s 200 was sized for JSON-only replies and silently truncated any
# reasoning). This superseded the original direct-JSON prompts used for the
# pre-2026-07-07 gen10/dev-set numbers — those are NOT re-judged (see
# experiment-log decision entry for why); only the held-out set is re-run.
PROMPT = ('Image 1 is a REFERENCE anime character portrait. Image 2 is a GENERATED '
          'portrait meant to depict the same character. Judge IDENTITY only: face '
          'shape, hair color, hair length/style, eye color, skin tone, gender '
          'presentation, and any distinctive marks (scars, accessories). IGNORE '
          'pose, outfit, expression, background, and rendering/art-style differences. '
          'First, think step by step: go through each identity axis one at a time '
          '(face shape, hair color, hair length/style, eye color, skin tone, gender, '
          'distinctive marks) and state what you observe in each image and whether '
          'they match. Then, on a new line, reply with ONLY compact JSON: '
          '{"same_character":"yes|no|partial","hair_color_match":"yes|no|n/a",'
          '"gender_match":"yes|no|n/a","confidence":0..1,"note":"<=15 words"}')

# Held-out validation (probe/41) additionally needs a rendering-quality signal
# for the production go/no-go — same identity prompt plus one extra field so
# the artifact rate (tiling/duplication/broken anatomy, seen sporadically on
# u29235-class seeds all session) gets counted on a set that was never used
# to pick the recipe, instead of guessed from spot-checks.
HOLDOUT_PROMPT = ('Image 1 is a REFERENCE anime character portrait. Image 2 is a GENERATED '
          'portrait meant to depict the same character. Judge IDENTITY only: face '
          'shape, hair color, hair length/style, eye color, skin tone, gender '
          'presentation, and any distinctive marks (scars, accessories). IGNORE '
          'pose, outfit, expression, background, and rendering/art-style differences. '
          'Also separately flag rendering defects in image 2 ONLY: tiled/duplicated '
          'sub-images, garbled or extra anatomy (limbs, eyes, faces), or a broken/'
          'collaged composition. Do not count normal art-style or minor blur as a defect. '
          'First, think step by step: (1) go through each identity axis one at a time '
          '(face shape, hair color, hair length/style, eye color, skin tone, gender, '
          'distinctive marks) and state what you observe in each image and whether '
          'they match; (2) separately, describe image 2\'s overall composition — is '
          'it a single coherent picture, or does it contain any tiled/duplicated/'
          'collaged sub-regions, and if so, describe exactly where and what they '
          'show. Then, on a new line, reply with ONLY compact JSON: '
          '{"same_character":"yes|no|partial","hair_color_match":"yes|no|n/a",'
          '"gender_match":"yes|no|n/a","visible_defect":"yes|no","confidence":0..1,'
          '"note":"<=15 words"}')

def _key(var):
    for l in open(KEY_SRC):
        if l.startswith(var + "="):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in {KEY_SRC}")

def _datauri(p, px=768, q=88):
    im = Image.open(p).convert("RGB"); im.thumbnail((px, px))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def _extract_json(txt):
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        # bug fixed 2026-07-07: this used to return an "error" dict directly,
        # which meant judge()'s retry loop never engaged for empty/malformed
        # 200-status responses (only network/HTTP-status failures retried) --
        # ~15% of held-out CoT judgments were silently one-shot failures.
        # Raising here routes through the same retry path as real errors.
        raise ValueError(f"no JSON object found in response (len={len(txt)}): {txt[:120]!r}")
    return json.loads(m.group(0))

def judge(ref, tgt, key, retries=4, prompt=None):
    body = {"model": PRO, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt or PROMPT},
        {"type": "image_url", "image_url": {"url": _datauri(ref)}},
        {"type": "image_url", "image_url": {"url": _datauri(tgt)}}]}],
        "max_tokens": 900, "temperature": 0.1}
    last = ""
    for a in range(retries + 1):
        try:
            r = requests.post(KAON_BASE + "/chat/completions",
                              headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                              json=body, timeout=180)
            if r.status_code in (408, 429) or r.status_code >= 500:
                last = f"http {r.status_code}: {r.text[:200]}"; time.sleep(3 * (a + 1)); continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            verdict = _extract_json(content)
            verdict["_reasoning"] = content  # full CoT trace, kept for spot-checking
            return verdict
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:200]; time.sleep(2 * (a + 1))
    return {"same_character": "error", "confidence": 0, "note": last}

def gen10_pairs():
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    pairs = []
    for uid in tags:
        if uid == "11663":
            continue
        for s in range(4):
            p = f"/mnt/local/cherry_out/gen10/u{uid}_gen10_B_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_B_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def gen10v2_pairs():
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    pairs = []
    for uid in tags:
        if uid == "11663":
            continue
        for s in range(4):
            p = f"/mnt/local/cherry_out/gen10v2/u{uid}_gen10v2_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_v2_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def band1025_pairs():
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    pairs = []
    for uid in tags:
        if uid == "11663":
            continue
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/band1025_10/u{uid}_b1025lf8_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_b1025lf8_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def lowfreq10_pairs():
    """Diagnostic re-judge (2026-07-16): lowfreq k=8 at the ORIGINAL narrow
    band (21-25, unchanged from strip's band width) -- images already exist
    from the pre-widening dev-set sweep (probe/30_kv_inject.py refpos=lowfreq,
    band=21-25) but were only ever identity-judged (gemini_judge_lowfreq10.json,
    no visible_defect field, predates the defect-aware prompt). Re-judging with
    HOLDOUT_PROMPT isolates band-width from the refpos/frequency-gating
    mechanism: lowfreq@21-25 (this set) vs lowfreq@10-25 (holdout 'new' arm,
    82% defect) holding refpos fixed and varying only band width, vs
    strip@21-25 (holdout 'old' arm, 20% defect) holding band width fixed and
    varying only refpos. Dev-set (gen10, 10 subjects) -- diagnostic only, not
    an acceptance number; any fix hypothesis this suggests still needs a
    single held-out validation pass before being reported as a result."""
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    pairs = []
    for uid in tags:
        if uid == "11663":
            continue
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/lowfreq10/u{uid}_lf8_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_lf8narrow_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def center10_pairs():
    """Diagnostic re-judge (2026-07-16): companion to lowfreq10_pairs -- 'center'
    refpos (full-precision RoPE re-centered, NO low-freq gating) at the same
    narrow band 21-25. Separates 'any repositioning at all' from 'low-freq-
    gated repositioning specifically' as a candidate cause of the collage/
    tiling collapse. Dev-set (gen10, 10 subjects), diagnostic only."""
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    pairs = []
    for uid in tags:
        if uid == "11663":
            continue
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/center10/u{uid}_center_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_center_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def widerband_decompose_pairs():
    """probe/62 band-width-vs-refpos decisive decomposition: strip_narrow
    (21-25, production 'old'), lowfreq_wide (10-25 k=8, frozen 'new'),
    strip_wide (10-25, THE decisive new cell) -- all three fresh on the
    same 12 devprobe subjects/seeds, judged with the same HOLDOUT_PROMPT
    instrument for a clean three-way comparison. Diagnostic only."""
    tags = json.load(open(os.path.join(DATA, "devprobe12_tags.json")))
    arms = ["strip_narrow", "lowfreq_wide", "strip_wide"]
    pairs = []
    for uid in tags:
        if uid.startswith("_"):
            continue
        for arm in arms:
            for s in [0, 1]:
                p = f"/mnt/local/cherry_out/widerband_decompose/u{uid}_wbd_{arm}_s{s}.png"
                if os.path.exists(p):
                    pairs.append((f"{uid}_{arm}_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def bv2_pairs():
    pairs = []
    for uid in ["18522", "29998"]:
        for s in range(8):
            p = f"/mnt/local/cherry_out/t0v2/u{uid}_bv2_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_bv2_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def holdout_pairs():
    """probe/41 frozen-recipe validation: 30 fresh subjects x 2 seeds x
    {old=strip/band21-25, new=lowfreq8/band10-25}, judged in one blind pass."""
    tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))
    pairs = []
    for uid in tags:
        for arm in ["old", "new"]:
            for s in [0, 1]:
                p = f"/mnt/local/cherry_out/holdout30/u{uid}_holdout_{arm}_s{s}.png"
                if os.path.exists(p):
                    pairs.append((f"{uid}_{arm}_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def headgate_quant_pairs():
    """distributed-identity quantification: lowfreq head-gate configs
    (baseline=all 30 heads / topqtr=top-7 / random7 / bottom7) x 12 subjects,
    judged for visible_defect (does defect track injection head-count?) and
    same_character. Ref = the true target ref. Grouped by config in a separate
    aggregation step."""
    REF_ = os.environ.get("REF", "/mnt/local/smoke_anime/anime_260623/img")
    subs = ["2542", "4073", "9871", "2636", "1317", "1620", "7577", "3090", "5412", "4057", "6021", "2838"]
    cfgs = ["baseline", "topqtr", "random7", "bottom7"]
    pairs = []
    for u in subs:
        for c in cfgs:
            p = f"/mnt/local/cherry_out/headgate/u{u}_{c}_q_s0.png"
            if os.path.exists(p):
                pairs.append((f"{u}_{c}", f"{REF_}/u{u}_ref.png", p))
    return pairs

def krea2_pairs():
    """probe/43 second baseline: conradlocke/krea2-identity-edit (community
    LoRA on Krea-2-Turbo), same 30 held-out subjects/refs/prompts, same
    defect-aware prompt as the held-out evaluation for direct comparability."""
    tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))
    pairs = []
    for uid in tags:
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/baseline_krea2/u{uid}_krea2_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_krea2_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def krea2_shuffled_pairs():
    """probe/73 shuffled-ref FLOOR for the krea2 baseline: same 30 held-out
    subjects/prompts as krea2_pairs(), but the GENERATED images were made by
    feeding krea2 a DIFFERENT (deranged) subject's ref image as input. Identity
    judgment here still compares against the TARGET's own real ref -- the
    swapped-in wrong ref was only input-side contamination; we're asking
    whether the output looks like the target, which it shouldn't if krea2's
    correct-ref identity number (krea2_pairs, ~33% clean-yes) actually depends
    on being fed the right ref, symmetric with cherry's own shuffled-ref
    control (docs/design/2026-07-11_traininglight_design.md §11.1, ~5%)."""
    tags = json.load(open(os.path.join(DATA, "holdout30_tags.json")))
    pairs = []
    for uid in tags:
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/baseline_krea2_shuffled/u{uid}_krea2shuf_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_krea2shuf_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def devprobe_strip_pairs():
    """probe/46 M1 dev-probe (design §5.1/§7 M1) strip arm: 12 subjects
    disjoint from train_manifest_7k5 + capstone/gen10/holdout30 (data/
    devprobe12_tags.json), production strip/band21-25/taper/w=4.0 recipe,
    2 seeds. Same defect-aware HOLDOUT_PROMPT as holdout/krea2 for
    direct comparability -- but per design §5.1 this is a REUSABLE monitoring
    set, not an acceptance set: no number from this --set may be cited as a
    final/acceptance result, only used to watch trend across checkpoints."""
    tags = json.load(open(os.path.join(DATA, "devprobe12_tags.json")))
    pairs = []
    for uid in tags:
        if uid.startswith("_"):  # '_amendment' etc: top-level metadata, not a uid
            continue
        for s in [0, 1]:
            p = f"/mnt/local/cherry_out/devprobe/strip/u{uid}_strip_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_strip_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def devprobe_lora_pairs():
    """probe/46 M1 dev-probe (design §5.1/§7 M1) training-light LoRA arm:
    same 12 subjects/seeds/prompts as devprobe_strip_pairs, but reading a
    specific checkpoint's generations. There is no single canonical lora
    checkpoint directory -- caller MUST set DEVPROBE_LORA_DIR to the
    checkpoint's devprobe output dir (probe/46_devprobe.py --stage gen --arm
    lora writes to /mnt/local/cherry_out/devprobe/lora_<ckpt_name>/), e.g.:
      DEVPROBE_LORA_DIR=/mnt/local/cherry_out/devprobe/lora_step_2000 \\
        $PY 38_gemini_judge.py --set devprobe_lora
    pid embeds the checkpoint name (from DEVPROBE_LORA_DIR's basename) so
    that if this --set is re-run across multiple checkpoints and the output
    JSON files are kept around under different names, pids stay
    disambiguated by arm+checkpoint, not just uid+seed."""
    tags = json.load(open(os.path.join(DATA, os.environ.get("DEVPROBE_TAGS", "devprobe12_tags.json"))))
    lora_dir = os.environ.get("DEVPROBE_LORA_DIR")
    if not lora_dir:
        raise RuntimeError(
            "--set devprobe_lora requires DEVPROBE_LORA_DIR=/mnt/local/cherry_out/"
            "devprobe/lora_<ckpt_name> (one checkpoint's image dir, written by "
            "probe/46_devprobe.py --stage gen --arm lora --adapter <ckpt_dir>)")
    ckpt_tag = os.path.basename(os.path.normpath(lora_dir)).removeprefix("lora_")
    pairs = []
    for uid in tags:
        if uid.startswith("_"):  # '_amendment' etc: top-level metadata, not a uid
            continue
        for s in [0, 1]:
            p = f"{lora_dir}/u{uid}_lora_s{s}.png"
            if os.path.exists(p):
                pairs.append((f"{uid}_lora_{ckpt_tag}_s{s}", f"{REF}/u{uid}_ref.png", p))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                     choices=["gen10", "gen10v2", "bv2", "band1025", "holdout", "krea2",
                              "krea2_shuffled", "headgate_quant",
                              "devprobe_strip", "devprobe_lora", "lowfreq10", "center10",
                              "widerband_decompose"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = _key("KAON_KEY")
    pairs = {"gen10": gen10_pairs, "gen10v2": gen10v2_pairs, "bv2": bv2_pairs,
             "band1025": band1025_pairs, "holdout": holdout_pairs,
             "krea2": krea2_pairs,
             "krea2_shuffled": krea2_shuffled_pairs,
             "headgate_quant": headgate_quant_pairs,
             "devprobe_strip": devprobe_strip_pairs, "devprobe_lora": devprobe_lora_pairs,
             "lowfreq10": lowfreq10_pairs, "center10": center10_pairs,
             "widerband_decompose": widerband_decompose_pairs}[a.set]()
    prompt = HOLDOUT_PROMPT if a.set in ("holdout", "krea2", "krea2_shuffled",
                                          "headgate_quant",
                                          "devprobe_strip", "devprobe_lora",
                                          "lowfreq10", "center10",
                                          "widerband_decompose") else PROMPT
    print(f"[gemini] judging {len(pairs)} pairs with {PRO}", flush=True)

    def work(item):
        pid, ref, tgt = item
        return pid, judge(ref, tgt, key, prompt=prompt)

    out = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, item) for item in pairs]
        for i, f in enumerate(as_completed(futs), 1):
            pid, res = f.result()
            out[pid] = res
            print(f"  [{i}/{len(pairs)}] {pid}: {res}", flush=True)

    outp = os.path.join(DATA, f"gemini_judge_{a.set}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    from collections import Counter
    if a.set in ("holdout", "widerband_decompose"):
        arms = ["old", "new"] if a.set == "holdout" else ["strip_narrow", "lowfreq_wide", "strip_wide"]
        for arm in arms:
            sub = {k: v for k, v in out.items() if f"_{arm}_" in k}
            c = Counter(v.get("same_character") for v in sub.values())
            d = Counter(v.get("visible_defect") for v in sub.values())
            n = len(sub)
            print(f"\n=== arm={arm} (n={n}) ===")
            for k in ("yes", "partial", "no", "error"):
                print(f"  same_character={k:8s} {c.get(k,0)}  ({100*c.get(k,0)/max(1,n):.0f}%)")
            print(f"  visible_defect=yes  {d.get('yes',0)}  ({100*d.get('yes',0)/max(1,n):.0f}%)")
    elif a.set in ("krea2", "krea2_shuffled", "devprobe_strip", "devprobe_lora", "lowfreq10", "center10"):
        c = Counter(v.get("same_character") for v in out.values())
        d = Counter(v.get("visible_defect") for v in out.values())
        n = len(out)
        note = "  (dev-probe: trend-only, per design §5.1 not an acceptance number)" \
            if a.set.startswith("devprobe") else ""
        print(f"\n=== arm={a.set} (n={n}) ==={note}")
        for k in ("yes", "partial", "no", "error"):
            print(f"  same_character={k:8s} {c.get(k,0)}  ({100*c.get(k,0)/max(1,n):.0f}%)")
        print(f"  visible_defect=yes  {d.get('yes',0)}  ({100*d.get('yes',0)/max(1,n):.0f}%)")
    else:
        c = Counter(v.get("same_character") for v in out.values())
        n = len(out)
        print(f"\n=== Gemini 3.1 Pro verdict summary (n={n}) ===")
        for k in ("yes", "partial", "no", "error"):
            print(f"  same_character={k:8s} {c.get(k,0)}  ({100*c.get(k,0)/n:.0f}%)")
    print(f"wrote {outp}")

if __name__ == "__main__":
    main()
