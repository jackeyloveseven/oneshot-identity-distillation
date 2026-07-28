#!/usr/bin/env python
"""
61_defect_only_diagnostic.py -- band-width vs refpos-mechanism decomposition
for the lowfreq recipe's 82% held-out defect collapse (see
docs/paper/section_trainlight_band_ablation.md-style caveat pattern; this is
the training-free-mechanism counterpart, targeting draft_v0.md's Limitations
line "why widening the band and frequency-gating together destabilize
composition is unexplained").

Motivation: `lowfreq` (band 10-25, k=8) changed TWO variables vs `strip`
(band 21-25) at once -- band width (5 layers -> 16 layers) AND refpos
(none -> low-freq-gated re-centered). The validated holdout numbers only
speak to the combined effect (82% defect). This script isolates band width
by re-judging OLDER dev-set images that hold refpos=lowfreq fixed but use
the ORIGINAL narrow band (21-25, `lowfreq10` dir) and the position-only
`center` variant (also 21-25, `center10` dir, no freq-gating) -- both
generated during the original band-widening exploration, never defect-judged
(predates the defect-aware prompt).

BLOCKER WORKED AROUND: gen10's original reference images
(/home/ubuntu/smoke_anime/revref_30k/img/u*_ref.png) no longer exist on disk
(cleaned up since that early sweep; confirmed via filesystem-wide search).
The standard HOLDOUT_PROMPT needs image 1 (reference) for identity judging,
but visible_defect (tiled/collaged/broken composition) is a property of
image 2 ALONE -- it needs no reference. This script judges defect-only,
single-image, sidestepping the missing-reference blocker entirely. This is
a narrower instrument than HOLDOUT_PROMPT (no identity signal), used only
for this diagnostic; identity numbers for these arms remain whatever the
old identity-only judge already recorded (gemini_judge_lowfreq10.json /
gemini_judge_center10.json, both pre-defect-prompt).

Diagnostic only -- dev-set (gen10, 10 subjects), NOT an acceptance number.
Any fix hypothesis this suggests still needs a single held-out validation
pass (defect-aware, ref-comparison HOLDOUT_PROMPT, fresh holdout subjects)
before being reported as a validated result.

Run: source ../env.sh && $PY 61_defect_only_diagnostic.py
"""
import os, json, base64, io, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
KAON_BASE = "https://kaon-router.kaonai.com/v1"
PRO = "google/gemini-3.1-pro-preview"
KEY_SRC = "/home/ubuntu/smoke_anime/flash_caption_smoke.py"

DEFECT_ONLY_PROMPT = (
    'Look at this single generated anime character portrait. Judge ONLY its '
    'composition/rendering coherence -- ignore identity, pose, outfit, art '
    'style. Flag a defect if the image contains: tiled/duplicated sub-images, '
    'a collaged/patchwork composition (multiple disconnected panel-like '
    'regions), garbled or extra anatomy (extra/malformed limbs, eyes, faces, '
    'hands), or any other broken/incoherent composition. Do NOT count normal '
    'art-style choices, minor blur, or a plain/simple background as a defect. '
    'First, describe the overall composition in 1-2 sentences: is it a single '
    'coherent picture, or does it contain tiled/duplicated/collaged '
    'sub-regions or anatomical breaks -- and if so, describe exactly where '
    'and what they show. Then, on a new line, reply with ONLY compact JSON: '
    '{"visible_defect":"yes|no","confidence":0..1,"note":"<=15 words"}'
)


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
        raise ValueError(f"no JSON object found in response (len={len(txt)}): {txt[:120]!r}")
    return json.loads(m.group(0))


def judge_single(img_path, key, retries=4):
    body = {"model": PRO, "messages": [{"role": "user", "content": [
        {"type": "text", "text": DEFECT_ONLY_PROMPT},
        {"type": "image_url", "image_url": {"url": _datauri(img_path)}}]}],
        "max_tokens": 500, "temperature": 0.1}
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
            verdict["_reasoning"] = content
            return verdict
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:200]; time.sleep(2 * (a + 1))
    return {"visible_defect": "error", "confidence": 0, "note": last}


ARMS = {
    "lowfreq_narrow_21-25": "/mnt/local/cherry_out/lowfreq10/u{uid}_lf8_s{s}.png",
    "center_narrow_21-25": "/mnt/local/cherry_out/center10/u{uid}_center_s{s}.png",
}


def main():
    tags = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    uids = [u for u in tags if u != "11663"]
    items = []
    for arm, tmpl in ARMS.items():
        for uid in uids:
            for s in [0, 1]:
                p = tmpl.format(uid=uid, s=s)
                if os.path.exists(p):
                    items.append((f"{arm}__u{uid}_s{s}", p))
    print(f"[defect-only] {len(items)} images across {len(ARMS)} arms, model={PRO}", flush=True)

    key = _key("KAON_KEY")

    def work(item):
        pid, p = item
        return pid, judge_single(p, key)

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, item) for item in items]
        for i, f in enumerate(as_completed(futs), 1):
            pid, res = f.result()
            out[pid] = res
            print(f"  [{i}/{len(items)}] {pid}: visible_defect={res.get('visible_defect')} {res.get('note','')}", flush=True)

    outp = os.path.join(DATA, "defect_only_diagnostic_band_width.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    from collections import Counter
    print(f"\n=== defect-only band-width-isolation diagnostic ===")
    for arm in ARMS:
        sub = {k: v for k, v in out.items() if k.startswith(arm + "__")}
        n = len(sub)
        d = Counter(v.get("visible_defect") for v in sub.values())
        rate = 100 * d.get("yes", 0) / max(1, n)
        print(f"  {arm:28s} n={n:3d}  defect={d.get('yes',0)}/{n} ({rate:.1f}%)")
    print(f"\n  reference points (validated, holdout30, n=60, HOLDOUT_PROMPT ref-comparison):")
    print(f"    strip        band21-25 (narrow, no refpos)      defect=20.0% [12,32]")
    print(f"    lowfreq k=8  band10-25 (WIDE, refpos=lowfreq)   defect=82.0% [70,89]")
    print(f"\n  wrote {outp}")


if __name__ == "__main__":
    main()
