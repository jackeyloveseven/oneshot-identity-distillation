#!/usr/bin/env python
"""
68_pilot2_eval.py — PILOT (pre-registered smoke gate, NOT final evidence)
eval+report for the two additive loss upgrades (--loss_perc, --loss_ccip)
added to probe/45_train_lora.py. Run once probe/67_pilot2_run.sh has trained
all THREE arms (baseline/ccip/perc) reach step 1500 via probe/69_pilot_real.sh
-- baseline is a clean config-matched mse-only arm (mix3, seed 0), NOT the old
confounded run_mix1/step_1500 (mix2, no ref_caption); see probe/69's header.

Three checkpoints, distinct ckptlink_pilot_* namespaces (adds a symlink next
to each checkpoint dir -- same already-established convention as
run_mix1/ckptlink_ho30_band10-25_r8 etc.; never touches the checkpoint files
themselves, nothing published is modified):
  ckptlink_pilot_ccip     -> run_pilot_ccip/step_1500   (--loss_ccip 0.05)
  ckptlink_pilot_perc     -> run_pilot_perc/step_1500   (--loss_perc 0.1)
  ckptlink_pilot_baseline -> run_pilot_baseline/step_1500 (clean mix3 mse-only)

Pipeline:
  1. symlink the three checkpoints under their ckptlink_pilot_* names.
  2. probe/46_devprobe.py --stage gen --arm lora --adapter <link>, once per
     checkpoint (12 uid x 2 seed = 24 images each, 72 total) -- generic
     CAP_REF_TXT, no --cap_ref override, devprobe12_tags.json prompt_B same
     as the rest of the devprobe12/band-ablation family, for comparability.
  3. probe/38_gemini_judge.py --set devprobe_lora per checkpoint (defect-aware
     HOLDOUT_PROMPT instrument, same one the band-ablation arms use) ->
     renamed to data/gemini_judge_devprobe_lora_pilot_<arm>.json.
  4. CCIP-torch cosine similarity (generation vs reference) for all 72
     images, reusing get_ccip_torch()/ccip_embed()/load_pixels() from
     45_train_lora.py VERBATIM (same F.interpolate-384 + CLIP-norm
     preprocessing the --loss_ccip term itself trains against -- NOT
     imgutils' PIL-based preprocessing, which would not measure the same
     thing the loss saw). ref-vs-ref cosine sanity-checked ~1.0 first.
  5. Smoke-gate report per arm vs baseline: ENGINEERING (train_log.jsonl NaN
     check, windowed-mean mse @1500 vs baseline), KILL-CHECK (defect rate vs
     baseline on the same 24-img probe), DIRECTION (judge same_character
     rate + CCIP-torch cosine sim vs baseline). n=24/arm, explicitly flagged
     pilot, not paper evidence -- see report["protocol"].

Run (after both real arms hit step 1500):
  source ../env.sh && $PY 68_pilot2_eval.py
"""
import os, sys, json, time, math, subprocess, importlib.util
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TRAINLIGHT = "/mnt/local/cherry_out/trainlight"
DEVPROBE_ROOT = "/mnt/local/cherry_out/devprobe"
REF_IMG_DIR = "/mnt/local/smoke_anime/anime_260623/img"

# env.sh's default REF (revref_30k) points at a directory that does not
# exist (see probe/58_band_ablation_run.sh's header, probe/56's docstring).
# 38_gemini_judge.py hard-requires os.environ["REF"] and 46_devprobe.py
# reads/re-normalizes it too -- both are invoked as subprocesses below and
# inherit THIS process's environment, so fix it here unconditionally rather
# than trusting the caller's shell to have overridden it after `source
# env.sh` (same fix 58/54 apply at the shell level).
os.environ["REF"] = REF_IMG_DIR

SUMMARY = os.path.join(DATA, "pilot2_summary.txt")  # same orchestration-progress file
                                                      # probe/67_pilot2_run.sh's log()
                                                      # tees into throughout training;
                                                      # this script is the LAST stage of
                                                      # the pilot pipeline and is
                                                      # responsible for the file's final
                                                      # "=== pilot2 all done ===" sentinel
                                                      # (see main()).

ARMS = {
    "ccip": dict(ckpt=f"{TRAINLIGHT}/run_pilot_ccip/step_1500", weight=0.05),
    "perc": dict(ckpt=f"{TRAINLIGHT}/run_pilot_perc/step_1500", weight=0.1),
    "baseline": dict(ckpt=f"{TRAINLIGHT}/run_pilot_baseline/step_1500", weight=None),
}
LINK_NAME = {"ccip": "ckptlink_pilot_ccip", "perc": "ckptlink_pilot_perc",
             "baseline": "ckptlink_pilot_baseline"}


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------
# 1. symlinks
# --------------------------------------------------------------------------
def make_links():
    links = {}
    for arm, spec in ARMS.items():
        ckpt = spec["ckpt"]
        if not os.path.isdir(ckpt):
            raise RuntimeError(f"checkpoint missing for arm={arm}: {ckpt} (train it first)")
        link = os.path.join(os.path.dirname(ckpt), LINK_NAME[arm])
        target = os.path.realpath(ckpt)
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            raise RuntimeError(f"{link} exists and is not a symlink, refusing to overwrite")
        os.symlink(target, link)
        links[arm] = link
        log(f"[link] {arm}: {link} -> {target}")
    return links


# --------------------------------------------------------------------------
# 2. devprobe12 generation (subprocess per checkpoint, reuses probe/46 verbatim)
# --------------------------------------------------------------------------
def gen_devprobe(links):
    py = os.environ.get("PY", sys.executable)
    for arm, link in links.items():
        ckpt_name = os.path.basename(os.path.normpath(link))
        out_dir = f"{DEVPROBE_ROOT}/lora_{ckpt_name}"
        n_expected = 24
        n_have = len([f for f in os.listdir(out_dir) if f.endswith(".png")]) if os.path.isdir(out_dir) else 0
        if n_have >= n_expected:
            log(f"[gen] arm={arm}: {n_have}/{n_expected} images already present at {out_dir}, skipping")
            continue
        log(f"[gen] arm={arm}: generating via probe/46_devprobe.py --adapter {link} "
            f"({n_have}/{n_expected} already present)")
        cmd = [py, os.path.join(HERE, "46_devprobe.py"), "--stage", "gen", "--arm", "lora", "--adapter", link]
        r = subprocess.run(cmd, cwd=HERE, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"[gen] FAILED arm={arm} rc={r.returncode}")
        n_have = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
        log(f"[gen] arm={arm}: {n_have}/{n_expected} images at {out_dir}")
        if n_have < n_expected:
            log(f"[gen] WARNING arm={arm}: only {n_have}/{n_expected} images -- some (uid,seed) "
                f"samples must have failed silently, see probe/46's own per-call subprocess logging")


# --------------------------------------------------------------------------
# 3. Gemini judge (subprocess per checkpoint, reuses probe/38 --set devprobe_lora verbatim)
# --------------------------------------------------------------------------
def judge_devprobe(links):
    py = os.environ.get("PY", sys.executable)
    out_paths = {}
    for arm, link in links.items():
        ckpt_name = os.path.basename(os.path.normpath(link))
        dest = os.path.join(DATA, f"gemini_judge_devprobe_lora_pilot_{arm}.json")
        if os.path.exists(dest):
            log(f"[judge] arm={arm}: {dest} already exists, skipping")
            out_paths[arm] = dest
            continue
        log(f"[judge] arm={arm}: judging via probe/38_gemini_judge.py --set devprobe_lora "
            f"(DEVPROBE_LORA_DIR=lora_{ckpt_name})")
        env = dict(os.environ)
        env["DEVPROBE_LORA_DIR"] = f"{DEVPROBE_ROOT}/lora_{ckpt_name}"
        cmd = [py, os.path.join(HERE, "38_gemini_judge.py"), "--set", "devprobe_lora"]
        r = subprocess.run(cmd, cwd=HERE, env=env, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"[judge] FAILED arm={arm} rc={r.returncode}")
        src = os.path.join(DATA, "gemini_judge_devprobe_lora.json")
        os.replace(src, dest)
        out_paths[arm] = dest
        log(f"[judge] arm={arm}: wrote {dest}")
    return out_paths


# --------------------------------------------------------------------------
# 4. CCIP-torch similarity (reuses 45_train_lora.py's get_ccip_torch/ccip_embed/
#    load_pixels VERBATIM -- exact same preprocessing --loss_ccip trains against)
# --------------------------------------------------------------------------
def ccip_scores(links):
    t45 = _load("t45_pilot2eval", "45_train_lora.py")

    ccip_model = t45.get_ccip_torch(4)   # bs=4-locked; 24 images/arm divides evenly,
                                          # reuses the SAME cached onnx the real training
                                          # runs already built under OUT_DIR/aux_models_cache/

    # ref-vs-ref sanity check: cosine(ref, ref) must be ~1.0 or something is
    # wrong with the preprocessing/model before any other number is trusted.
    # Loaded at 1024 (not pre-shrunk to 384): ccip_embed() itself downsamples
    # to 384 via F.interpolate, and that's exactly what forward_backward_aux()
    # does to the (1024px) VAE-decoded x0_hat during training -- loading here
    # at 1024 makes the eval measure the identical pipeline --loss_ccip
    # trained against, not a separately-downsampled variant of it.
    tags = json.load(open(os.path.join(DATA, "devprobe12_tags.json")))
    uids = [u for u in tags if not u.startswith("_")]
    sanity_refs = [f"{REF_IMG_DIR}/u{uids[i % len(uids)]}_ref.png" for i in range(4)]
    with torch.no_grad():
        px = t45.load_pixels(sanity_refs, 1024)
        emb = t45.ccip_embed(ccip_model, px)
        cos_self = F.cosine_similarity(emb, emb, dim=-1).mean().item()
    log(f"[ccip] ref-vs-ref self-cosine sanity check: {cos_self:.6f} (expect ~1.0)")
    if abs(cos_self - 1.0) > 1e-3:
        raise RuntimeError(f"CCIP-torch self-cosine sanity check FAILED: {cos_self} "
                            f"(expected ~1.0) -- do not trust ccip scores below, fix preprocessing first")

    results = {}
    for arm, link in links.items():
        ckpt_name = os.path.basename(os.path.normpath(link))
        gen_dir = f"{DEVPROBE_ROOT}/lora_{ckpt_name}"
        pairs = []
        for uid in uids:
            for s in [0, 1]:
                gen_p = f"{gen_dir}/u{uid}_lora_s{s}.png"
                ref_p = f"{REF_IMG_DIR}/u{uid}_ref.png"
                if os.path.exists(gen_p) and os.path.exists(ref_p):
                    pairs.append((f"{uid}_s{s}", gen_p, ref_p))
        log(f"[ccip] arm={arm}: {len(pairs)} gen/ref pairs found")
        per_image = {}
        for i in range(0, len(pairs), 4):
            grp = pairs[i:i + 4]
            if len(grp) < 4:
                log(f"[ccip] arm={arm}: dropping ragged tail group of {len(grp)} (<4, model is "
                    f"bs=4-locked): {[g[0] for g in grp]}")
                continue
            with torch.no_grad():
                gen_px = t45.load_pixels([g[1] for g in grp], 1024)  # same 1024->384
                ref_px = t45.load_pixels([g[2] for g in grp], 1024)  # rationale as the
                                                                       # sanity check above
                gen_emb = t45.ccip_embed(ccip_model, gen_px)
                ref_emb = t45.ccip_embed(ccip_model, ref_px)
                sims = F.cosine_similarity(gen_emb, ref_emb, dim=-1).cpu().tolist()
            for (pid, _, _), sim in zip(grp, sims):
                per_image[pid] = sim
        mean_sim = sum(per_image.values()) / len(per_image) if per_image else None
        results[arm] = dict(per_image=per_image, mean_cosine_sim=mean_sim, n=len(per_image))
        log(f"[ccip] arm={arm}: mean cosine sim={mean_sim} (n={len(per_image)})")
    return results, cos_self


# --------------------------------------------------------------------------
# 5. smoke-gate report
# --------------------------------------------------------------------------
def windowed_mean(train_log_path, around_step, window=50, field="mse"):
    """Mean of `field` over steps in [around_step-window, around_step]. Plain
    mse-only arms (baseline) only ever log `loss` (which IS mse there -- see
    45_train_lora.py's forward_backward()); falls back to `loss` if `field`
    is absent from a row."""
    if not os.path.exists(train_log_path):
        return None
    vals = []
    with open(train_log_path) as f:
        for line in f:
            d = json.loads(line)
            if around_step - window <= d["step"] <= around_step:
                v = d.get(field, d["loss"])
                if v is not None:
                    vals.append(v)
    return sum(vals) / len(vals) if vals else None


def check_nan(train_log_path):
    if not os.path.exists(train_log_path):
        return None
    n_nan, n_total = 0, 0
    for line in open(train_log_path):
        d = json.loads(line)
        n_total += 1
        if any(isinstance(d.get(k), float) and math.isnan(d.get(k)) for k in ("loss", "mse", "perc", "ccip")):
            n_nan += 1
    return dict(n_total=n_total, n_nan=n_nan)


def defect_rate(judge_json_path):
    d = json.load(open(judge_json_path))
    same = Counter(v.get("same_character") for v in d.values())
    defect = Counter(v.get("visible_defect") for v in d.values())
    n = len(d)
    return dict(n=n,
                defect_yes=defect.get("yes", 0), defect_rate=defect.get("yes", 0) / n if n else None,
                same_yes=same.get("yes", 0), same_partial=same.get("partial", 0),
                same_no=same.get("no", 0), same_error=same.get("error", 0),
                same_yes_rate=same.get("yes", 0) / n if n else None)


def build_report(judge_paths, ccip_results, ccip_self_sanity):
    train_logs = {
        "ccip": f"{TRAINLIGHT}/run_pilot_ccip/train_log.jsonl",
        "perc": f"{TRAINLIGHT}/run_pilot_perc/train_log.jsonl",
        "baseline": f"{TRAINLIGHT}/run_pilot_baseline/train_log.jsonl",
    }
    baseline_mse = windowed_mean(train_logs["baseline"], 1500, field="loss")
    baseline_defect = defect_rate(judge_paths["baseline"])
    baseline_ccip = ccip_results["baseline"]["mean_cosine_sim"]

    report = dict(
        protocol="PILOT (pre-registered smoke gate, NOT final evidence) -- n=24 images/arm "
                 "(12 subjects x 2 seeds, devprobe12 dev-set), n=1 training seed/arm, 1500 steps "
                 "only. Numbers below are directional smoke-gate signals, not paper evidence.",
        ccip_self_sanity_cosine=ccip_self_sanity,
        baseline=dict(ckpt="run_pilot_baseline/step_1500", windowed_mean_mse_at_1500=baseline_mse,
                       kill_check=baseline_defect, ccip_cosine_sim=baseline_ccip,
                       note="CLEAN config-matched baseline (2026-07-18, probe/69_pilot_real.sh): "
                            "identical pilot command (band 10-25, r8, alpha8, lr1e-4, seed 0, mix3, "
                            "1500 steps) with NO aux flag -> arms differ ONLY in the aux loss. "
                            "Replaces the old run_mix1/step_1500 baseline, which trained on mix2 "
                            "(no ref_caption field, pre-53_ref_caption_wd14.py) and so used a "
                            "different ref-slot caption regime than the mix3 pilot arms -- a real "
                            "confound the advisor flagged, now removed."),
        arms={},
    )
    for arm in ("ccip", "perc"):
        tl = train_logs[arm]
        nan_check = check_nan(tl)
        arm_mse = windowed_mean(tl, 1500, field="mse")
        arm_defect = defect_rate(judge_paths[arm])
        arm_ccip = ccip_results[arm]["mean_cosine_sim"]
        defect_worse = None
        if arm_defect["defect_rate"] is not None and baseline_defect["defect_rate"] is not None:
            defect_worse = arm_defect["defect_rate"] > baseline_defect["defect_rate"]
        report["arms"][arm] = dict(
            weight=ARMS[arm]["weight"],
            engineering=dict(
                nan_check=nan_check,
                windowed_mean_mse_at_1500=arm_mse,
                baseline_windowed_mean_mse_at_1500=baseline_mse,
                mse_delta_vs_baseline=(arm_mse - baseline_mse)
                    if (arm_mse is not None and baseline_mse is not None) else None,
            ),
            kill_check=dict(
                arm=arm_defect, baseline=baseline_defect,
                defect_rate_worse_than_baseline=defect_worse,
            ),
            direction=dict(
                judge_same_character_yes_rate=arm_defect["same_yes_rate"],
                baseline_judge_same_character_yes_rate=baseline_defect["same_yes_rate"],
                ccip_cosine_sim=arm_ccip,
                baseline_ccip_cosine_sim=baseline_ccip,
                ccip_delta_vs_baseline=(arm_ccip - baseline_ccip)
                    if (arm_ccip is not None and baseline_ccip is not None) else None,
            ),
        )
    return report


def main():
    log("=== pilot2 eval start ===")
    links = make_links()
    gen_devprobe(links)
    judge_paths = judge_devprobe(links)
    ccip_results, ccip_self_sanity = ccip_scores(links)
    report = build_report(judge_paths, ccip_results, ccip_self_sanity)

    out_json = os.path.join(DATA, "pilot2_summary.json")
    with open(out_json, "w") as f:
        json.dump(dict(report=report, ccip_raw=ccip_results), f, indent=2)
    log(f"wrote {out_json}")
    print(json.dumps(report, indent=2))

    # final digest into the SAME orchestration-progress file
    # probe/67_pilot2_run.sh has been appending to throughout training --
    # this is the last stage of the pilot pipeline, so it also owns the
    # file's closing sentinel line (required verbatim: "=== pilot2 all done ===").
    log("=== pilot2 smoke-gate digest ===")
    log(f"baseline (run_pilot_baseline/step_1500, clean mix3 mse-only): windowed_mean_mse={report['baseline']['windowed_mean_mse_at_1500']} "
        f"defect_rate={report['baseline']['kill_check']['defect_rate']} "
        f"same_char_yes_rate={report['baseline']['kill_check']['same_yes_rate']} "
        f"ccip_cosine_sim={report['baseline']['ccip_cosine_sim']}")
    for arm in ("ccip", "perc"):
        a = report["arms"][arm]
        log(f"arm={arm} weight={a['weight']}: "
            f"ENGINEERING mse={a['engineering']['windowed_mean_mse_at_1500']} "
            f"(delta_vs_baseline={a['engineering']['mse_delta_vs_baseline']}) "
            f"nan_count={a['engineering']['nan_check']['n_nan'] if a['engineering']['nan_check'] else 'n/a'} | "
            f"KILL-CHECK defect_rate={a['kill_check']['arm']['defect_rate']} "
            f"worse_than_baseline={a['kill_check']['defect_rate_worse_than_baseline']} | "
            f"DIRECTION same_char_yes_rate={a['direction']['judge_same_character_yes_rate']} "
            f"ccip_cosine_sim={a['direction']['ccip_cosine_sim']} "
            f"(delta_vs_baseline={a['direction']['ccip_delta_vs_baseline']})")
    log(f"protocol reminder: {report['protocol']}")
    log(f"full structured report: {out_json}")
    log("=== pilot2 eval done ===")
    log("=== pilot2 all done ===")


if __name__ == "__main__":
    main()
