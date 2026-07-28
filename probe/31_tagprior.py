#!/usr/bin/env python
"""
31_tagprior.py — Wave-1 1a: T0 tag-prior direct-write (review avg 7.7, 3/3 keep).

Thesis test: structural attributes (gender / hair length+structure / skin tone /
occlusion) that KV provably cannot carry (capstone) are written into the scene
caption via wd14 tags; KV keeps anchoring the position-free residual.

Three stages (run in order; gen is resumable / skip-if-exists):
  --stage tags   wd14 on the 6 capstone refs -> structural tags -> rendered
                 prompts (arms B/D) -> data/t0_tags.json.  CPU-cheap, no pipe.
  --stage gen    4 arms x 6 subjects x 8 seeds = 192 imgs -> $T0OUT.
                 Default --max_imgs 10 per invocation (batch pacing); 0 = all.
  --stage score  CCIP vs ref + pairwise seed consistency -> data/t0_scores.json
                 + pre-registered verdict table.

Arms (KV config fixed at current best: band 21-25, taper, w=4, strip, route off):
  A = generic PROMPT + KV        (status quo baseline)
  B = tag-phrase caption + KV    (the hypothesis)
  C = tag-phrase caption, no KV  (does text alone suffice?)
  D = booru raw-tag caption + KV (dialect probe: same info, native wording)

Pre-registered criteria (docs/design/2026-07-02_wave1_exec_design.md):
  rescue      = subject reaches >=4/8 seeds CCIP-SAME (<0.178)
  H1 (rescue) : B rescues subjects failing under A; overall A 4/6 -> B >=5/6
  H2 (var)    : B mean pairwise-CCIP consistency < A and < C
  H3 (anchor) : C failing where B passes SUPPORTS (not proves) KV-anchored residual

Minefields honored: capture forward always uses generic PROMPT (30_kv_inject
handles this); wd14 general threshold fixed at >=0.5; bulk output on /mnt/local
(root disk 99% full); tags + confidences logged (a mis-written 1boy kills identity).
"""
import os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF = os.environ["REF"]
T0OUT = os.environ.get("T0OUT", "/mnt/local/cherry_out/t0")
SUBJECTS = ["10000", "18522", "20582", "29998", "4238", "9172"]  # capstone 6
SEEDS = list(range(8))
ARMS = ["A", "B", "C", "D"]
WD14_THR = 0.5  # minefield #4: fixed
TAGS_JSON = os.path.join(DATA, "t0_tags.json")

GENERIC = ("solo, single character, centered close-up anime portrait, upper body, "
           "detailed face, clean anime style, simple plain white background")
TEMPLATE = ("solo, single character, {ident}, centered close-up anime portrait, "
            "upper body, detailed face, clean anime style, simple plain white background")

# ---------------- structural tag whitelist (wd14 v3 general tags) ----------------
GENDER = {"1girl": "young woman", "1boy": "young man"}
AGE = {"child": "child"}
SKIN = {"dark_skin": "dark-skinned", "dark-skinned_female": "dark-skinned",
        "dark-skinned_male": "dark-skinned", "tan": "tanned"}
HAIR_COLOR = {"white_hair": "white", "silver_hair": "silver", "grey_hair": "grey",
              "black_hair": "black", "brown_hair": "brown", "blonde_hair": "blonde",
              "red_hair": "red", "pink_hair": "pink", "purple_hair": "purple",
              "light_purple_hair": "light purple", "blue_hair": "blue",
              "light_blue_hair": "light blue", "aqua_hair": "aqua", "green_hair": "green",
              "orange_hair": "orange", "multicolored_hair": "multicolored"}
EYE_COLOR = {"blue_eyes": "blue eyes", "red_eyes": "red eyes", "green_eyes": "green eyes",
             "brown_eyes": "brown eyes", "yellow_eyes": "yellow eyes", "purple_eyes": "purple eyes",
             "black_eyes": "black eyes", "grey_eyes": "grey eyes", "pink_eyes": "pink eyes",
             "orange_eyes": "orange eyes", "amber_eyes": "amber eyes", "aqua_eyes": "aqua eyes",
             "heterochromia": "heterochromia (two different eye colors)"}
HAIR_LEN = {"short_hair": "short", "medium_hair": "medium-length", "long_hair": "long",
            "very_long_hair": "very long", "absurdly_long_hair": "extremely long"}
HAIR_ADJ = {"braid": "braided", "twin_braids": "twin-braided", "french_braid": "braided",
            "drill_hair": "drill-curled", "wavy_hair": "wavy", "curly_hair": "curly",
            "straight_hair": "straight", "messy_hair": "messy", "spiked_hair": "spiky"}
HAIR_POST = {"ponytail": "tied in a ponytail", "side_ponytail": "tied in a side ponytail",
             "twintails": "tied in twintails", "low_twintails": "tied in low twintails",
             "hair_bun": "tied in a bun", "double_bun": "tied in two buns",
             "hime_cut": "in a hime cut", "bob_cut": "in a bob cut"}
HAIR_EXTRA = {"blunt_bangs": "with blunt bangs", "ahoge": "with an ahoge",
              "hair_over_one_eye": "with hair covering one eye"}
WEAR = {"glasses": "wearing glasses", "sunglasses": "wearing sunglasses",
        "eyepatch": "wearing an eyepatch", "mask": "wearing a mask",
        "mouth_mask": "wearing a mouth mask", "hat": "wearing a hat",
        "hood_up": "with the hood up"}
FACE = {"beard": "with a beard", "mustache": "with a mustache", "scar": "with a scar",
        "scar_across_eye": "with a scar across the eye", "freckles": "with freckles",
        "mole_under_eye": "with a mole under the eye"}
# accessories: same whitelist-gap pattern as HAIR_COLOR/EYE_COLOR (found via Gemini
# 3.1 Pro judge flagging "missing hair ribbon"/"missing nose ring"/"missing piercings"
# on otherwise-good generations, 2026-07-04).
ACCESSORY = {"hair_ribbon": "with a hair ribbon", "ribbon": "with a ribbon",
             "hair_bow": "with a hair bow", "earrings": "wearing earrings",
             "ear_piercing": "with ear piercings", "nose_piercing": "with a nose piercing",
             "hairpin": "with a hairpin", "hair_flower": "with a hair flower",
             "headband": "wearing a headband", "choker": "wearing a choker",
             "necklace": "wearing a necklace",
             # whitelist-gap fix round 2 (2026-07-12, manifest_mix3_report.json
             # signature_item_diagnostics on u7495/u1138/u3737): tattoo variants all
             # collapse to one phrase; more piercing sites; eyewear pushed to the head.
             "tattoo": "with tattoos", "arm_tattoo": "with tattoos", "chest_tattoo": "with tattoos",
             "facial_tattoo": "with tattoos", "neck_tattoo": "with tattoos", "leg_tattoo": "with tattoos",
             "piercing": "with a piercing", "lip_piercing": "with a lip piercing",
             "eyebrow_piercing": "with an eyebrow piercing",
             "eyewear_on_head": "with sunglasses on the head",
             # non-human species markers (found via u6300 tiger-boy character, cache scan
             # top-40 non-STRUCT tag frequency)
             "animal_ears": "with animal ears", "tail": "with a tail", "horns": "with horns",
             "pointy_ears": "with pointy ears", "fangs": "with fangs", "wings": "with wings",
             # facial mark beyond mole_under_eye
             "mole": "with a facial mole"}
STRUCT = {**GENDER, **AGE, **SKIN, **EYE_COLOR, **HAIR_COLOR, **HAIR_LEN, **HAIR_ADJ, **HAIR_POST,
          **HAIR_EXTRA, **WEAR, **FACE, **ACCESSORY}

def render_phrase(tags):
    """Deterministic tags -> natural noun phrase. `tags` = struct tags present."""
    has = lambda d: [t for t in d if t in tags]
    noun = GENDER.get((has(GENDER) or ["1girl"])[0], "young woman")
    if has(AGE):
        noun = "little " + ("girl" if "woman" in noun else "boy")
    skin = SKIN[has(SKIN)[0]] + " " if has(SKIN) else ""
    parts = [f"a {skin}{noun}"]
    hc = has(HAIR_COLOR); hl = has(HAIR_LEN); ha = has(HAIR_ADJ); hp = has(HAIR_POST); hx = has(HAIR_EXTRA)
    if hc or hl or ha:
        adj = " ".join([HAIR_COLOR[t] for t in hc[:1]] + [HAIR_LEN[t] for t in hl[:1]]
                       + [HAIR_ADJ[t] for t in ha])
        clause = f"with {adj} hair"
        if hp:
            clause += " " + HAIR_POST[hp[0]]
        parts.append(clause)
    elif hp:
        parts.append(f"with hair {HAIR_POST[hp[0]]}")
    parts += [HAIR_EXTRA[t] for t in hx]
    ec = has(EYE_COLOR)
    if ec:
        parts.append(EYE_COLOR[ec[0]])
    parts += [WEAR[t] for t in has(WEAR)]
    parts += [FACE[t] for t in has(FACE)]
    parts += [ACCESSORY[t] for t in has(ACCESSORY)]
    parts = list(dict.fromkeys(parts))
    return ", ".join(parts)

def stage_tags():
    from imgutils.tagging import get_wd14_tags
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for uid in SUBJECTS:
        rating, general, chars = get_wd14_tags(f"{REF}/u{uid}_ref.png",
                                               model_name="SwinV2_v3",
                                               general_threshold=WD14_THR)
        struct = {t: round(float(c), 3) for t, c in general.items() if t in STRUCT}
        phrase = render_phrase(struct)
        booru = ", ".join(t.replace("_", " ") for t in struct)
        out[uid] = {"general": {t: round(float(c), 3) for t, c in general.items()},
                    "struct": struct, "phrase": phrase, "booru": booru,
                    "prompt_B": TEMPLATE.format(ident=phrase),
                    "prompt_D": TEMPLATE.format(ident=booru)}
        print(f"u{uid}: struct={sorted(struct)}\n  B: {phrase}\n  D: {booru}", flush=True)
    with open(TAGS_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {TAGS_JSON}")

def arm_cfg(arm, rec):
    """(w, prompt) for an arm; prompt=None => generic PROMPT."""
    return {"A": (4.0, None), "B": (4.0, rec["prompt_B"]),
            "C": ("off", rec["prompt_B"]), "D": (4.0, rec["prompt_D"])}[arm]

def img_path(uid, arm, seed):
    return f"{T0OUT}/u{uid}_t0_{arm}_s{seed}.png"

def stage_gen(max_imgs):
    tags = json.load(open(TAGS_JSON))
    todo = [(u, a, s) for u in SUBJECTS for a in ARMS for s in SEEDS
            if not os.path.exists(img_path(u, a, s))]
    print(f"[gen] {len(todo)} images remaining of {len(SUBJECTS)*len(ARMS)*len(SEEDS)}", flush=True)
    if not todo:
        return
    if max_imgs > 0:
        todo = todo[:max_imgs]
    os.makedirs(T0OUT, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "30_kv_inject.py"))
    kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
    kv.INJ["band"] = kv.parse_band("21-25")
    for uid, arm, seed in todo:
        w, prompt = arm_cfg(arm, tags[uid])
        print(f"[gen] u{uid} arm={arm} seed={seed} w={w}", flush=True)
        im = kv.generate(uid, w, sched_mode="taper", seed=seed, prompt=prompt)
        im.save(img_path(uid, arm, seed))
    left = sum(1 for u in SUBJECTS for a in ARMS for s in SEEDS
               if not os.path.exists(img_path(u, a, s)))
    print(f"[gen] batch done, {left} images still missing", flush=True)

def stage_score():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", os.path.join(HERE, "32_metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    res = {}
    for uid in SUBJECTS:
        res[uid] = {}
        for arm in ARMS:
            paths = [p for s in SEEDS if os.path.exists(p := img_path(uid, arm, s))]
            if paths:
                res[uid][arm] = m.score_group(uid, paths)
    with open(os.path.join(DATA, "t0_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    hdr = " | ".join(f"{a}: same/n mean cons" for a in ARMS)
    print(f"\nuid    | {hdr}")
    rescued = {a: 0 for a in ARMS}
    for uid in SUBJECTS:
        cells = []
        for arm in ARMS:
            g = res[uid].get(arm)
            if not g:
                cells.append("-"); continue
            cons = f"{g['consistency']:.3f}" if g["consistency"] is not None else "-"
            cells.append(f"{g['same']}/{g['n']} {g['mean']:.3f} {cons}")
            if g["n"] >= 8 and g["same"] >= 4:
                rescued[arm] += 1
        print(f"u{uid:6s}| " + " | ".join(cells))
    print(f"\nrescue (>=4/8 SAME) per arm: " +
          ", ".join(f"{a}={rescued[a]}/6" for a in ARMS))
    print("H1: B >= 5/6 and > A | H2: B consistency < A and < C per subject | "
          "H3: (C fail & B pass) supports KV-anchored residual")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tags", "gen", "score"])
    ap.add_argument("--max_imgs", type=int, default=10,
                    help="gen batch size per invocation; 0 = unlimited")
    a = ap.parse_args()
    if a.stage == "tags":
        stage_tags()
    elif a.stage == "gen":
        stage_gen(a.max_imgs)
    else:
        stage_score()
