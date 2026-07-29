#!/usr/bin/env python
"""
46_devprobe.py — M1 dev-probe: the small, repeatedly-reusable monitoring set
for training-light checkpoints (design docs/design/2026-07-11_traininglight_
design.md §5.1 "dev-probe" + §7 M1). This is NOT the frozen-acceptance
held-out set (§5.2, random.Random(2026), N=30) and NOT holdout30 (§5.3,
single-shot terminal eval) — dev-probe is the layer that "allows repeated
reuse, allows tuning against it", used only to decide "continue training /
stop / switch approach", per the design's explicit rule that no number from
this set may ever be cited as the acceptance/terminal result.

CRITICAL non-negotiable per design §2.5 + §5.1: EXCLUDE for this set must
include not just capstone(6)+gen10_devset(11)+holdout30(30) (the 47-uid union
already frozen in data/exclude_uids.json) but ALSO every uid that is actually
IN the training manifest (data/train_manifest_7k5.jsonl, 5,945 uids) — this
is the training-line-specific risk the training-free stages never had: a
dev-probe subject the model was literally trained on would make "does the
checkpoint look better than strip on this subject" a meaningless question.

Two stages:
  --stage build              build the 12-subject set + prompt_B captions +
                              a human-review contact sheet of the 12 ref crops.
  --stage gen --arm strip    generate the training-free production-recipe
                              baseline arm (12 uid x 2 seed = 24 images).
  --stage gen --arm lora --adapter <ckpt_dir>
                              generate the training-light LoRA arm via
                              probe/45_train_lora.py's --stage sample CLI
                              (subprocess, one call per uid x seed -- see
                              docstring on stage_gen_lora for why per-call
                              subprocess, not a single batched import call,
                              is the reliable choice here).

Sampling recipe (design §5.1, verbatim):
  candidate pool = ref+target both present under anime_260623/img, AND uid in
                   training_triples.jsonl's uid space (training_triples uid
                   field is int -- str()-normalized here, see EXCLUDE note)
  EXCLUDE = data/exclude_uids.json["union"] (47) | every train_manifest_7k5
            uid (5,945)   -- union'd BEFORE the shuffle, str-typed throughout
  random.Random(2025).shuffle(candidate_pool)   -- new seed, frozen by design
  is_solo() filter reused verbatim from probe/41_holdout_eval.py (wd14
  SwinV2_v3, general_threshold=0.3, solo>=0.5, no multi-character tag) --
  imported via importlib with REF pre-set to anime_260623/img so its
  module-level `REF = os.environ["REF"]` resolves correctly, exactly the
  "same code, same params, just a new uid space" reuse the task calls for.
  N=12, first 12 passing uids in shuffle order kept.

prompt_B generation reuses probe/41_holdout_eval.py's stage_tags() wd14
tag-prior rendering path verbatim (same probe/31_tagprior.py STRUCT
whitelist, WD14_THR, render_phrase, TEMPLATE) -- just applied to the new 12
uids instead of holdout30's 30. Output data/devprobe12_tags.json is
structurally identical to data/holdout30_tags.json ({uid: {struct, phrase,
prompt_B}}) so any downstream code that already reads one can read the other.

Extra manual-review gate (this task's own addition, beyond the design doc):
wd14 solo>=0.5 should reject non-character reference images (the design
doc's own worked example is u2939, a ruins/scenery ref physically present in
training_triples' uid space with ref+target both on disk -- confirmed by
this script's build stage to be in-scope-by-the-numeric-rules-alone, and
presumed caught by is_solo()), but since a wrong caption-less ref silently
entering training is exactly the failure mode design §2.5/§5.1 exists to
prevent, --stage build also renders a 12-up contact sheet of the selected
ref crops (labeled by uid) for a human to eyeball before any generation
spends GPU time on a bad subject.

Run:
  source ../env.sh
  $PY 46_devprobe.py --stage build
  $PY 46_devprobe.py --stage gen --arm strip
  $PY 46_devprobe.py --stage gen --arm lora --adapter out/trainlight/run_full1/step_2000
  $PY 38_gemini_judge.py --set devprobe_strip
  DEVPROBE_LORA_DIR=out/devprobe/lora_step_2000 $PY 38_gemini_judge.py --set devprobe_lora
"""
import os, sys, json, time, random, argparse, subprocess, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

REF_IMG_DIR = os.environ.get("REF", "/mnt/local/smoke_anime/anime_260623/img")
os.environ["REF"] = REF_IMG_DIR  # normalize so every importlib-loaded module
                                  # we pull in below (41_holdout_eval.py,
                                  # 30_kv_inject.py) resolves the same REF at
                                  # their own module-import time.
print(f"[46_devprobe] using REF={REF_IMG_DIR}", flush=True)

TRIPLES_PATH = "/mnt/local/smoke_anime/training_triples.jsonl"
EXCLUDE_JSON = os.path.join(DATA, "exclude_uids.json")
MANIFEST_PATH = os.path.join(DATA, "train_manifest_7k5.jsonl")
TAGS_JSON = os.path.join(DATA, os.environ.get("DEVPROBE_TAGS", "devprobe12_tags.json"))
CONTACT_SHEET = "out/trainlight/devprobe_refs.png"
GEN_OUT_ROOT = "out/devprobe"

N_SUBJECTS = 12
SAMPLE_SEED = 2025  # design §5.1: "新种子，不与 gen10 的 42、holdout30 的 2024 撞"
SEEDS = [0, 1]


def _load(modname, filename):
    """Same read-only importlib-spec pattern 45_train_lora.py/50_production.py
    already use to pull in sibling probe/ scripts without editing them."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# stage: build
# --------------------------------------------------------------------------

def build_exclude():
    excl = json.load(open(EXCLUDE_JSON))
    excl47 = set(excl["union"])
    assert len(excl47) == 47, f"expected 47-uid EXCLUDE union, got {len(excl47)}"
    manifest_uids = set()
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            manifest_uids.add(str(json.loads(line)["uid"]))  # manifest uid is
                                                               # already str,
                                                               # str() is a
                                                               # no-op safety net
    print(f"[build] EXCLUDE: {len(excl47)} (exclude_uids.json union) | "
          f"{len(manifest_uids)} (train_manifest_7k5 uids) | "
          f"{len(excl47 & manifest_uids)} overlap", flush=True)
    return excl47 | manifest_uids


def build_candidate_pool(exclude):
    t45 = _load("t45_devprobe", "45_train_lora.py")
    avail_pairs = t45.scan_available_pairs()  # ref+target both present, str uids
    triples_uids = set()
    with open(TRIPLES_PATH) as f:
        for line in f:
            triples_uids.add(str(json.loads(line)["uid"]))  # training_triples.jsonl
                                                              # uid field is int --
                                                              # str()-normalize here
                                                              # (design §5.1 note)
    candidate_all = avail_pairs & triples_uids
    print(f"[build] candidate pool (ref+target present AND uid in "
          f"training_triples.jsonl uid space): {len(candidate_all)}", flush=True)

    hit = candidate_all & exclude
    print(f"[build] EXCLUDE hits within candidate pool: {len(hit)}", flush=True)

    pool = sorted(candidate_all - exclude, key=int)  # deterministic pre-shuffle
                                                       # order (numeric uid) --
                                                       # os.listdir() order is
                                                       # not guaranteed stable
                                                       # across runs/machines,
                                                       # unlike 41's original
                                                       # one-shot use, this set
                                                       # must reproduce exactly
    print(f"[build] candidate pool after EXCLUDE: {len(pool)}", flush=True)
    return pool


def stage_build():
    exclude = build_exclude()
    pool = build_candidate_pool(exclude)

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(pool)

    h41 = _load("h41_devprobe", "41_holdout_eval.py")  # reuses is_solo() verbatim
                                                         # (wd14 SwinV2_v3,
                                                         # general_threshold=0.3,
                                                         # solo>=0.5, no
                                                         # multi-char tag) --
                                                         # module-level REF is
                                                         # already normalized
                                                         # above before this load

    kept = []       # [(uid, 1-indexed position in shuffled pool)]
    rejected = 0
    for i, uid in enumerate(pool, start=1):
        if len(kept) >= N_SUBJECTS:
            break
        if h41.is_solo(uid):
            kept.append((uid, i))
        else:
            rejected += 1
    print(f"[build] is_solo filter: kept {len(kept)}, rejected {rejected} "
          f"(multi-char/no-solo/non-character)", flush=True)
    if len(kept) < N_SUBJECTS:
        print(f"[build] WARNING: only {len(kept)}/{N_SUBJECTS} subjects found "
              f"in the entire shuffled pool of {len(pool)}", flush=True)

    for uid, pos in kept:
        print(f"[build] selected u{uid}  (candidate position {pos} of {len(pool)})", flush=True)

    # prompt_B: verbatim reuse of 41_holdout_eval.py's stage_tags() wd14
    # tag-prior rendering path (same probe/31_tagprior.py STRUCT/WD14_THR/
    # render_phrase/TEMPLATE), just run on these 12 uids instead of holdout30's.
    tp = _load("tp_devprobe", "31_tagprior.py")
    from imgutils.tagging import get_wd14_tags
    out = {}
    for uid, pos in kept:
        rating, general, chars = get_wd14_tags(f"{REF_IMG_DIR}/u{uid}_ref.png",
                                                model_name="SwinV2_v3",
                                                general_threshold=tp.WD14_THR)
        struct = {t: round(float(c), 3) for t, c in general.items() if t in tp.STRUCT}
        phrase = tp.render_phrase(struct)
        out[uid] = {"struct": struct, "phrase": phrase,
                    "prompt_B": tp.TEMPLATE.format(ident=phrase),
                    "candidate_position": pos}
        print(f"[build] u{uid}: {phrase!r}", flush=True)

    os.makedirs(DATA, exist_ok=True)
    with open(TAGS_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[build] wrote {TAGS_JSON} ({len(out)} subjects)", flush=True)

    build_contact_sheet([uid for uid, _ in kept])


def build_contact_sheet(uids, cols=4, cell=320):
    """12-up grid of the selected ref crops, uid-labeled, for the human
    solo/non-character sanity check this task calls out explicitly (design's
    own worked failure case: u2939 is a ruins/scenery ref sitting inside
    training_triples' uid space with ref+target both present -- is_solo()
    should reject it before it ever reaches this function, but this contact
    sheet is the belt-and-suspenders manual check on top of that filter)."""
    from PIL import Image, ImageDraw
    rows = (len(uids) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    for idx, uid in enumerate(uids):
        im = Image.open(f"{REF_IMG_DIR}/u{uid}_ref.png").convert("RGB")
        im = im.resize((cell, cell))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, cell, 28], fill="black")
        d.text((6, 6), f"u{uid}", fill="white")
        r, c = divmod(idx, cols)
        canvas.paste(im, (c * cell, r * cell))
    os.makedirs(os.path.dirname(CONTACT_SHEET), exist_ok=True)
    canvas.save(CONTACT_SHEET)
    print(f"[build] wrote contact sheet {CONTACT_SHEET} ({len(uids)} refs, "
          f"{cols}x{rows} grid) -- please eyeball before spending GPU time "
          f"on generation", flush=True)


# --------------------------------------------------------------------------
# stage: gen --arm strip
# --------------------------------------------------------------------------

def load_tags():
    """devprobe12_tags.json subjects only -- top-level keys starting with
    '_' are metadata (e.g. the 2026-07-11 '_amendment' record documenting
    the human-review swap of u1491/u4459 for u7495/u1138), not uids."""
    tags = json.load(open(TAGS_JSON))
    return {k: v for k, v in tags.items() if not k.startswith("_")}


def strip_img_path(uid, seed):
    return f"{GEN_OUT_ROOT}/strip/u{uid}_strip_s{seed}.png"


def stage_gen_strip():
    """Full reuse of 50_production.py's frozen RECIPE dict (strip / band
    21-25 / taper / w=4.0 / steps=50 / cfg=5.0 / refsize=512 / size=1024) --
    the RECIPE values are read from that module, not re-hardcoded here, so
    this stays byte-identical to the production config if 50_production.py's
    RECIPE ever changes. prompt is devprobe12_tags.json's prompt_B (NOT
    50_production.IdentityGenerator.caption_for()'s live wd14 re-tag -- using
    the cached prompt_B guarantees the strip and lora arms see the exact same
    caption text for a fair M1 comparison, design §5.3's "same prompt_B for
    all arms" fairness rule applied one layer early)."""
    tags = load_tags()
    todo = [(u, s) for u in tags for s in SEEDS if not os.path.exists(strip_img_path(u, s))]
    print(f"[gen/strip] {len(todo)} images remaining of {len(tags) * len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(f"{GEN_OUT_ROOT}/strip", exist_ok=True)

    prod = _load("prod50_devprobe", "50_production.py")
    kv = _load("kv_devprobe", "30_kv_inject.py")
    kv.REF = REF_IMG_DIR
    kv.INJ["band"] = kv.parse_band(prod.RECIPE["band"])

    t0 = time.time()
    for uid, seed in todo:
        prompt = tags[uid]["prompt_B"]
        print(f"[gen/strip] u{uid} seed={seed}", flush=True)
        im = kv.generate(uid, prod.RECIPE["w"], refpos=prod.RECIPE["refpos"],
                         size=prod.RECIPE["size"], refsize=prod.RECIPE["refsize"],
                         steps=prod.RECIPE["steps"], cfg=prod.RECIPE["cfg"],
                         seed=seed, sched_mode=prod.RECIPE["sched_mode"], prompt=prompt)
        im.save(strip_img_path(uid, seed))
    dt = time.time() - t0
    print(f"[gen/strip] batch done: {len(todo)} images in {dt:.1f}s "
          f"({dt/max(1,len(todo)):.1f}s/image)", flush=True)


# --------------------------------------------------------------------------
# stage: gen --arm lora
# --------------------------------------------------------------------------

def lora_img_path(ckpt_name, uid, seed):
    return f"{GEN_OUT_ROOT}/lora_{ckpt_name}/u{uid}_lora_s{seed}.png"


def stage_gen_lora(adapter):
    """Calls probe/45_train_lora.py --stage sample as a subprocess, once per
    (uid, seed) pair.

    Why subprocess-per-call and not one importlib call over all 12 uids: 45's
    stage_sample() DOES support a comma-separated --ref uid list in one call
    (resolve_refs()), but it only accepts a SINGLE --prompt string applied to
    every uid in that list -- there is no per-uid prompt override inside a
    single invocation, and the project's constraint here explicitly forbids
    editing 45_train_lora.py to add one. Since devprobe12_tags.json's
    prompt_B differs per subject (and must exactly match what the strip arm
    used, for a fair M1 comparison), the only correct way to get per-uid
    prompts through 45's unmodified CLI is one subprocess call per uid, each
    with --ref u{uid} --prompt "<that uid's prompt_B>". This costs a pipeline
    reload per call (~24 reloads for the full set) instead of 1-2, but is not
    run by this task (lora checkpoints don't exist yet) -- correctness and
    process isolation (a bad/OOM sample can't take out the whole batch) were
    weighed as more important than that reload cost for a set this small.

    45's stage_sample writes to a fixed path
    OUT_DIR/sample_u{uid}_seed{seed}_adpt-{ckpt_basename}.png (its own
    hardcoded OUT_DIR="out/trainlight", tag scheme from its
    own source) -- this function moves each output to this project's
    devprobe/lora_{ckpt_name}/... layout afterward, it does not touch 45's
    output convention.
    """
    tags = load_tags()
    ckpt_name = os.path.basename(os.path.normpath(adapter))
    todo = [(u, s) for u in tags for s in SEEDS
            if not os.path.exists(lora_img_path(ckpt_name, u, s))]
    print(f"[gen/lora] adapter={adapter} ckpt_name={ckpt_name}: "
          f"{len(todo)} images remaining of {len(tags) * len(SEEDS)}", flush=True)
    if not todo:
        return
    os.makedirs(f"{GEN_OUT_ROOT}/lora_{ckpt_name}", exist_ok=True)

    py = os.environ.get("PY", sys.executable)
    script45 = os.path.join(HERE, "45_train_lora.py")
    TRAINLIGHT_OUT = "out/trainlight"  # 45's own hardcoded OUT_DIR

    t0 = time.time()
    for uid, seed in todo:
        prompt = tags[uid]["prompt_B"]
        src = f"{TRAINLIGHT_OUT}/sample_u{uid}_seed{seed}_adpt-{ckpt_name}.png"
        dst = lora_img_path(ckpt_name, uid, seed)
        print(f"[gen/lora] u{uid} seed={seed} -> {dst}", flush=True)
        cmd = [py, script45, "--stage", "sample", "--ref", str(uid),
               "--seed", str(seed), "--adapter", adapter, "--prompt", prompt]
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"[gen/lora] FAILED (exit {r.returncode}) for u{uid} seed={seed}, "
                  f"skipping move", flush=True)
            continue
        if not os.path.exists(src):
            print(f"[gen/lora] WARNING: expected output {src} not found after "
                  f"subprocess exit 0, skipping move", flush=True)
            continue
        os.replace(src, dst)
    dt = time.time() - t0
    print(f"[gen/lora] batch done: {len(todo)} images attempted in {dt:.1f}s", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "gen"])
    ap.add_argument("--arm", choices=["strip", "lora"], help="required for --stage gen")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir, required for --arm lora")
    a = ap.parse_args()

    if a.stage == "build":
        stage_build()
    elif a.stage == "gen":
        if a.arm is None:
            ap.error("--arm is required for --stage gen")
        if a.arm == "strip":
            stage_gen_strip()
        else:
            if not a.adapter:
                ap.error("--adapter is required for --arm lora")
            stage_gen_lora(a.adapter)


if __name__ == "__main__":
    main()
