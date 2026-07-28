#!/usr/bin/env python
"""
60_crossjudge_gpt55.py -- independent cross-judge check for the band-position
ablation's defect-rate finding (mid 10-25 vs full-depth 0-29), requested to
strengthen idea_thesis.md's Axis-A weakness: the "two independent
replications" claim (training-free recipe reversal + training-light band
ablation) both ran under the same judge (Gemini 3.1 Pro, Kaon router) and
same project/domain -- a skeptical reviewer could ask how independent that
really is. This adds a THIRD signal: re-judge the same mid-vs-full-depth
image pairs with a different model family (GPT-5.5, via Kaon router) using
the identical HOLDOUT_PROMPT protocol from probe/38_gemini_judge.py, and
check whether the defect-rate gap (mid low, full-depth high) survives.

Reuses probe/38_gemini_judge.py's exact prompt text, image-loading, and
retry logic verbatim -- only the model ID and endpoint model field change.
kaon/claude-opus-4-* were tried first (closer "different lab" signal than
GPT) but are 403 "not in allowlist" for this project's KAON_KEY; kaon/gpt-5.5
(OpenAI family, genuinely different from Google's Gemini) is allowlisted and
verified working before this script was written.

s0 seed only (not s0+s1) -- 30 subjects x 2 arms = 60 calls, proportionate
to the original ablation's n=60/arm without doubling the full re-judge cost.

Run: source ../env.sh && $PY 60_crossjudge_gpt55.py
"""
import os, json, base64, io, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
KAON_BASE = "https://kaon-router.kaonai.com/v1"
MODEL = "kaon/gpt-5.5"
KEY_SRC = "/home/ubuntu/smoke_anime/flash_caption_smoke.py"

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
          'they match; (2) separately, describe image 2\'s overall composition -- is '
          'it a single coherent picture, or does it contain any tiled/duplicated/'
          'collaged sub-regions, and if so, describe exactly where and what they '
          'show. Then, on a new line, reply with ONLY compact JSON: '
          '{"same_character":"yes|no|partial","hair_color_match":"yes|no|n/a",'
          '"gender_match":"yes|no|n/a","visible_defect":"yes|no","confidence":0..1,'
          '"note":"<=15 words"}')

REF = "/mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png"
MID = "/mnt/local/cherry_out/devprobe/base_rep30_step4000/u{uid}_lora_s0.png"
FULLDEPTH = "/mnt/local/cherry_out/devprobe/lora_ckptlink_band0-29_r4/u{uid}_lora_s0.png"


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


def judge(ref, tgt, key, retries=4):
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": HOLDOUT_PROMPT},
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
            verdict["_reasoning"] = content
            return verdict
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:200]; time.sleep(2 * (a + 1))
    return {"same_character": "error", "confidence": 0, "note": last}


def main():
    mid_j = json.load(open(os.path.join(DATA, "gemini_judge_devprobe_base_rep30_step4000.json")))
    uids = sorted(set(k.split("_")[0] for k in mid_j if k.endswith("_s0")))
    print(f"[crossjudge] {len(uids)} subjects x 2 arms = {len(uids)*2} calls, model={MODEL}", flush=True)

    key = _key("KAON_KEY")
    pairs = []
    for uid in uids:
        pairs.append((f"{uid}_mid_s0", REF.format(uid=uid), MID.format(uid=uid)))
        pairs.append((f"{uid}_fulldepth_s0", REF.format(uid=uid), FULLDEPTH.format(uid=uid)))

    def work(item):
        pid, ref, tgt = item
        return pid, judge(ref, tgt, key)

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, item) for item in pairs]
        for i, f in enumerate(as_completed(futs), 1):
            pid, res = f.result()
            out[pid] = res
            print(f"  [{i}/{len(pairs)}] {pid}: same_character={res.get('same_character')} "
                  f"visible_defect={res.get('visible_defect')}", flush=True)

    outp = os.path.join(DATA, "crossjudge_gpt55_band_mid_vs_fulldepth.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[crossjudge] saved {outp}")


if __name__ == "__main__":
    main()
