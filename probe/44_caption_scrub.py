#!/usr/bin/env python
"""
44_caption_scrub.py -- training-light data prep, step 1: caption source + cleaning
(docs/design/2026-07-11_traininglight_design.md §2.2-2.6, §5.1; red line #4).

Motivation: `training_triples.jsonl`'s own `prompt` field is 99.8% poisoned with
"keep exact identity from the attached reference (...)" phrasing (design doc §2.1)
-- unusable, and its `prompt` values are never read by this script (only its `uid`
field, to define the current 7.5k-tier image pool; red line #4 concerns caption
*content*, not uid bookkeeping). `anime_30k_recap/metadata.jsonl`'s `t2i_instruction`
field (22,734 records) has 0 hits on the 8 known leak phrases (§2.2) -- but literal
absence of "keep identity" wording isn't suffient: it still writes distinctive
marks/accessories (masks, chokers, tattoos, animal ears, heterochromia, ...) and
face-geometry words straight into the text, which gives a strong T2I model enough to
paint them from text alone and starves the ref-channel's gradient (design doc §2.3,
echoing README Wave-3's "unstripped caption -> zero FM loss pressure on ref slot").

Cleaning principle (§2.3): KEEP gender / names / coarse scene-action-composition /
colour attributes (hair, eye, skin colour, and "colour + generic garment" like
"white t-shirt" -- colour is a text-channel-reliable signal per Wave-1 1a/1c,
`probe/31_tagprior.py` HAIR_COLOR/EYE_COLOR). STRIP distinctive marks/accessories
(mask, goggles, scar, eyepatch, ear/hair ornaments, choker, bell, bandage, animal
ears/tail/horns, tattoo, mole, heterochromia) + face-geometry wording -- these are
exactly the identity cues a ref-conditioned adapter should have to learn from the
image, not read off the caption. Word roots are lifted from `31_tagprior.py`'s
ACCESSORY / WEAR(mask,eyepatch only) / FACE(scar,mole only) / EYE_COLOR(heterochromia
carved out) dicts, then extended with natural-language synonyms actually observed in
`t2i_instruction` (goggles, bell, bandage, animal-ear/tail variants, tattoo,
mismatched/different-coloured eyes, hair clip/ornament, ...). Deliberately NOT
stripped, despite being nominally accessories: glasses/sunglasses/hat/hood/beard/
mustache/freckles -- the design doc's explicit strip list names 护目镜(goggles) but
not 眼镜/墨镜(glasses/sunglasses), and these are common enough (near-universal, not
signature) that they read more like clothing than an identity anchor. Also
deliberately NOT stripped: bare "bow" -- in this corpus it's overwhelmingly a school-
uniform "bow tie" (clothing, ~as generic as a necktie) or, worse, a weapon ("aiming a
large bow") or the verb "to bow"; stripping it had a real false-positive rate on a
1500-sample dry run, so only the unambiguous "hair bow" phrase is stripped.

Two classes added after the 20-sample human review (coordinator decision,
2026-07-11): (1) face-area markings / face paint ("facial markings", "spiral
markings on her forehead", "a red diamond painted on his cheek", "clown face
paint") -- face-area ONLY; body/scene "markings" and transient action marks
("slap mark", "lipstick kiss marks", non -ing spellings in the "on X" form) stay;
(2) pointed/elf-type ears ("pointed ears", "pointy ears", "elf/elven/elf-like
ears") -- same identity-anchor class as animal ears; race nouns themselves ("an
elf", "elven woman") are grammatical subjects and are kept.

Cleaning is short-phrase/clause-level excision, not whole-sentence deletion (spec
requirement) implemented as: split `t2i_instruction` on top-level commas into
clauses; for each clause that contains a strip term, excise just the local noun
phrase around the match (bounded regex, article + up to 3 modifier words, explicitly
NOT crossing "with/and/prepositions" -- keeps it from swallowing an unrelated
downstream verb clause) plus any bounded "with/wrapped in/covered by X" trailing
prepositional phrase whose object is a strip term; everything else in the clause is
left untouched. A clause/list item that turns out to be *only* about the accessory
(e.g. "black choker") excises to empty and the whole clause is dropped. A governing
verb ("wearing"/"with") whose own list item gets fully dropped is reattached to the
next surviving bare list item in the same run (fixes "wearing X[dropped], Y" ->
"wearing Y", not "X[dropped], Y"), tracked via a small state machine across clauses.
Colour-bearing clauses ("blonde woman", "green eyes") are never sacrificed even if a
strip term rides along in the same clause -- only the accessory sub-phrase goes.

Validated: 18,000 corpus lines dry-run across 6 random seeds (with the 2026-07-11
additions in) -> 0 residual strip-term hits in scrubbed text, ~34% of captions
touched (avg ~1.28 phrases deleted per touched caption), 0.15% cosmetic "dangling
and/wearing before punctuation" residual (27/18000 -- harmless coordination
artifacts, none attributable to the added rules, not leaked identity words or lost
KEEP content) after several rounds of stress-testing against pathological
multi-sentence clauses (bug classes eliminated along the way: whole-sentence loss
when a strip word lands on an unrelated background entity, "tail-gun"/"piercing
eyes" false positives, dropped clothing when an accessory shares a list item with
it, dangling "wearing over his left eye" after eyepatch removal, dangling "and"
when a deleted list item was followed by a participle).

Data facts (verified, see docs/design/2026-07-11_traininglight_design.md §2.4-2.6):
  IMG_DIR has u{uid}_ref.png / _whiteref.png / _target.png. ref+target-both-present,
  intersected with training_triples.jsonl's uid space (28,726 uids -- the "current
  7.5k tier"), is exactly 7,515 uids. Minus EXCLUDE (47 uids, 46 of which fall inside
  that uid space) -> 7,489. Intersected with anime_30k_recap's 22,734 captioned uids
  -> 5,945 "uid has image pair + caption + not excluded" rows -- this is the actual
  buildable set for --stage build (data/train_manifest_7k5.jsonl is a tier-name, not
  a literal row count).

Two stages:
  --stage review  Sample N=20 (fixed seed) uids from the 3-condition set, scrub each,
                   write data/caption_scrub_review20.md for human sign-off. Cheap,
                   meant to actually be run.
  --stage build    Full 5,945-row data/train_manifest_7k5.jsonl + leak-phrase scan +
                   summary stats. Written but NOT run by this task -- gated on the
                   review stage passing human review first (main session runs it).

EXCLUDE provenance (also written to data/exclude_uids.json by this script, so the
JSON on disk and this script never drift apart):
  capstone (6)      probe/31_tagprior.py:39 SUBJECTS list, "# capstone 6" comment;
                     literal set corroborated by probe/41_holdout_eval.py:50 CAPSTONE.
  gen10_devset (11) data/gen10_tags.json keys (11, not the naively-assumed 10 --
                     verified against probe/41_holdout_eval.py:51-52 GEN10_DEVSET,
                     identical 11 uids).
  holdout30 (30)    data/holdout30_tags.json keys -- the frozen terminal-eval set
                     from probe/41_holdout_eval.py --stage tags (random.Random(2024)).

Run: source ../env.sh && $PY 44_caption_scrub.py --stage review
     $PY 44_caption_scrub.py --stage build   # NOT run by this task; main session's call
"""
import os
import re
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

IMG_DIR = "/mnt/local/smoke_anime/anime_260623/img"
CAPTION_JSONL = "/mnt/local/smoke_anime/anime_30k_recap/metadata.jsonl"
# uid field ONLY, to define the current 7.5k-tier image pool -- `prompt` field of
# this file is never read (red line #4: it's 99.8% identity-leak phrasing, design
# doc §2.1). Do not add a `prompt`/`t2i_instruction` read against this file.
TRAINING_TRIPLES_JSONL = "/mnt/local/smoke_anime/training_triples.jsonl"

EXCLUDE_JSON = os.path.join(DATA, "exclude_uids.json")
REVIEW_MD = os.path.join(DATA, "caption_scrub_review20.md")
MANIFEST_JSONL = os.path.join(DATA, "train_manifest_7k5.jsonl")

REVIEW_N = 20
REVIEW_SEED = 44  # fixed; distinct from other seeds already used in this project
                   # (7=training_triples audit, 42=gen10, 2024=holdout30,
                   # 2025=dev-probe, 2026=frozen-acceptance per design doc §5)

LEAK_PHRASES = [
    "keep exact identity", "keep the identity", "from the attached reference",
    "keeping their recognizable identity", "maintain identity", "same identity as",
    "reference image", "from the reference",
]

# ============================================================================
# EXCLUDE set (capstone + gen10 devset + holdout30) -- §2.5 hard rule: training
# uids must never include a uid used by any evaluation set.
# ============================================================================

# probe/31_tagprior.py:39  SUBJECTS = [...]  # capstone 6
# corroborated verbatim by probe/41_holdout_eval.py:50 CAPSTONE = {...}
CAPSTONE = sorted(["10000", "18522", "20582", "29998", "4238", "9172"])


def build_exclude():
    """Load gen10/holdout30 uids from their authoritative JSON files (not
    hardcoded), assemble EXCLUDE, and (re)write data/exclude_uids.json with
    provenance so the on-disk file and this script can never drift apart."""
    gen10 = json.load(open(os.path.join(DATA, "gen10_tags.json")))
    holdout30 = json.load(open(os.path.join(DATA, "holdout30_tags.json")))
    gen10_uids = sorted(gen10.keys())
    holdout30_uids = sorted(holdout30.keys())
    if len(gen10_uids) != 11:
        print(f"[WARN] gen10_tags.json has {len(gen10_uids)} uids, design doc "
              f"expects 11 -- check probe/41_holdout_eval.py GEN10_DEVSET", flush=True)
    if len(holdout30_uids) != 30:
        print(f"[WARN] holdout30_tags.json has {len(holdout30_uids)} uids, "
              f"expected 30", flush=True)

    union = sorted(set(CAPSTONE) | set(gen10_uids) | set(holdout30_uids))
    out = {
        "_purpose": (
            "EXCLUDE set for training-light caption/data prep (probe/44_caption_scrub.py). "
            "Any uid here must never enter the training manifest, per "
            "docs/design/2026-07-11_traininglight_design.md §2.5: capstone/gen10/holdout30 "
            "uids overlap the training_triples uid space and, worse, some are physically the "
            "same images as the 7515-pair training pool -- must be subtracted before building "
            "data/train_manifest_7k5.jsonl."
        ),
        "capstone": {
            "uids": CAPSTONE,
            "count": len(CAPSTONE),
            "provenance": (
                "probe/31_tagprior.py:39 SUBJECTS = ['10000','18522','20582','29998','4238','9172']"
                "  # capstone 6 -- hardcoded literal, corroborated verbatim by "
                "probe/41_holdout_eval.py:50 CAPSTONE = {...} (same 6 uids, same source set)."
            ),
        },
        "gen10_devset": {
            "uids": gen10_uids,
            "count": len(gen10_uids),
            "provenance": (
                "data/gen10_tags.json keys (loaded dynamically, not hardcoded) -- matches "
                "probe/41_holdout_eval.py:51-52 GEN10_DEVSET literal set exactly. Design doc "
                "§2.5 calls this \"gen10 devset 11个\"."
            ),
        },
        "holdout30": {
            "uids": holdout30_uids,
            "count": len(holdout30_uids),
            "provenance": (
                "data/holdout30_tags.json keys -- the frozen terminal-eval set built by "
                "probe/41_holdout_eval.py --stage tags (random.Random(2024), disjoint from "
                "capstone/gen10 by construction)."
            ),
        },
        "union": union,
        "union_count": len(union),
    }
    with open(EXCLUDE_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return set(union)


# ============================================================================
# Pool construction: ref+target pairs, intersected with training_triples' uid
# space (defines the current "7.5k tier"), minus EXCLUDE, intersected with
# captioned uids.
# ============================================================================

def build_pool():
    files = os.listdir(IMG_DIR)
    ref_ids, target_ids = set(), set()
    pat = re.compile(r"^u(\d+)_(ref|target|whiteref)\.png$")
    for fn in files:
        m = pat.match(fn)
        if not m:
            continue
        uid, kind = m.group(1), m.group(2)
        if kind == "ref":
            ref_ids.add(uid)
        elif kind == "target":
            target_ids.add(uid)
    both = ref_ids & target_ids

    tt_uids = set()
    with open(TRAINING_TRIPLES_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)  # `prompt` field, if present, is never accessed below
            tt_uids.add(str(d["uid"]))

    pool_7515 = both & tt_uids

    captions = {}
    with open(CAPTION_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("t2i_instruction", "")
            if t:
                captions[str(d["uid"])] = t

    exclude = build_exclude()
    pool_minus_exclude = pool_7515 - exclude
    three_cond = pool_minus_exclude & captions.keys()

    print(f"[pool] ref+target pairs (in training_triples uid space) = {len(pool_7515)}",
          flush=True)
    print(f"[pool] EXCLUDE union = {len(exclude)}, hits inside pool_7515 = "
          f"{len(pool_7515 & exclude)}", flush=True)
    print(f"[pool] pool_7515 - EXCLUDE = {len(pool_minus_exclude)}", flush=True)
    print(f"[pool] captioned uids (t2i_instruction) = {len(captions)}", flush=True)
    print(f"[pool] 3-condition intersection = {len(three_cond)}", flush=True)

    return pool_minus_exclude, captions, exclude, three_cond


# ============================================================================
# Caption scrubber
# ============================================================================

# distinctive marks/accessories + face-geometry word roots. Sourced from
# probe/31_tagprior.py's ACCESSORY dict (hair_ribbon/ribbon/hair_bow/earrings/
# ear_piercing/nose_piercing/hairpin/hair_flower/headband/choker/necklace --
# all present below) and the WEAR/FACE dicts narrowed to just the entries the
# design doc explicitly names (mask, eyepatch, scar, mole -- NOT glasses/
# sunglasses/hat/hood_up/beard/mustache/freckles, which read as generic
# clothing rather than a signature identity mark), plus natural-language
# synonyms observed directly in t2i_instruction (goggles, bell, bandage,
# animal-ear/tail variants, tattoo, mismatched/different-coloured eyes, hair
# clip/ornament/accessory) that aren't wd14 tag names but are common English
# phrasing for the same concepts.
STRIP_PATTERNS = [
    r"\b(?:face\s+)?masks?\b",
    r"\bgoggles\b",
    r"\beye[- ]?patch(?:es)?\b",
    r"\bscars?\b",
    r"\bscarr(?:ed|ing)\b",
    r"\bchokers?\b",
    r"\bbells?\b",
    r"\bbandag(?:e|es|ed)\b",
    r"\bear[- ]?rings?\b",
    r"\bear[- ]piercings?\b",
    r"\bnose[- ](?:piercings?|rings?)\b",
    r"\b(?:facial|face|lip|eyebrow|nipple|tongue|navel|body|multiple|numerous)\s+piercings?\b",
    r"\bpiercings\b(?!\s+(?:eyes?|gaze|stare|look|glare))",  # exclude "piercing eyes/gaze" idiom
    r"\bhair[- ]?pins?\b",
    r"\bhair clips?\b",
    r"\bhair ornaments?\b",
    r"\bhair accessor(?:y|ies)\b",
    r"\bhair flowers?\b",
    r"\bheadbands?\b",
    r"\bnecklaces?\b",
    r"\bribbons?\b",
    r"\bhair bows?\b",  # bare "bow" deliberately NOT stripped -- see module docstring
    r"\b(?:animal|cat|fox|wolf|dog|bunny|rabbit|canine|feline|mouse|bear|deer)[- ]ears?\b",
    # pointed/elf ears (coordinator-approved add-on, 2026-07-11): fantasy ear
    # shape is the same class of ref-channel identity anchor as animal ears.
    # Head noun "ears" is required, so "pointed hat/tail/shoulders/locks" and
    # verb usages ("pointed forward") are structurally unreachable (verified:
    # 40 such non-ear "pointed" usages in corpus, 0 matched).
    r"\b(?:pointed|pointy)\s+(?:elf[- ]?like\s+|elven\s+|elf\s+)?ears?\b",
    r"\b(?:elf[- ]?like|elven|elf)\s+ears?\b",
    r"\bhorns?\b",
    r"(?<!twin )(?<!twin-)\btails?\b(?!-)",  # exclude "twin tail(s)" (hairstyle) and "tail-X" compounds
    r"\btattoos?\b",
    r"\bmoles?\b",
    r"\bheterochromia\b",
    r"\bmismatched eyes\b",
    r"\bdifferent[- ]colou?red eyes\b",
    r"\bodd[- ]eyes\b",
    r"\btwo different eye colors?\b",
    # facial markings / face paint (coordinator-approved add-on, 2026-07-11):
    # painted-on/innate face-area marks are signature identity features
    # (Sukuna-style facial markings, clown/flag face paint, cheek diamonds).
    # Scope is face-area ONLY: bare/body "markings" ("gold body markings",
    # "glowing markings", 177 corpus lines) stay untouched, and the "markings
    # on X" form requires the -ing spelling so transient scene marks ("red
    # slap mark on his cheek", "lipstick kiss marks on his face" -- action
    # consequences, not identity) never match. Verified against corpus:
    # 0 road/ground/scene "markings", 0 "face painting" activity usages, and
    # all 13 non-face "painted on" lines (ceiling/wall/balaclava) unreachable
    # because the object here must be face/cheek/forehead/nose/chin.
    r"\b(?:facial|face|cheek|forehead)\s+mark(?:ing)?s?\b",
    r"\bmarkings\s+on\s+(?:his|her|their|its)?\s*(?:face|cheeks?|forehead|neck|chin)"
    r"(?:\s+and\s+(?:face|cheeks?|forehead|neck|chin|chest|body|arms?))?\b",
    r"\bpainted\s+on\s+(?:his|her|their|its)\s+(?:face|cheeks?|forehead|nose|chin)\b",
    r"\bface\s*paint(?:ing)?s?(?:\s+markings?)?\b",
    # face geometry / structure (rare in this corpus, kept for completeness)
    r"\b(?:round|oval|square|heart[- ]shaped|diamond[- ]shaped|oblong|narrow|angular)\s+face\b",
    r"\b(?:sharp|strong|square|chiseled)\s+jaw(?:line)?\b",
    r"\b(?:high|prominent)\s+cheekbones\b",
    r"\b(?:button|sharp|pointed)[- ]?nose\b",
]
STRIP_CORE = "|".join(STRIP_PATTERNS)
STRIP_RE = re.compile(STRIP_CORE, re.I)

# words a local-phrase-excision lookback must never cross (prepositions/verbs/
# articles) so it can't reach past a genuine phrase boundary and eat unrelated
# content.
STOP_FOR_LOOKBACK = [
    "with", "and", "who", "that", "which", "while", "over", "under", "near",
    "behind", "in", "on", "at", "from", "by", "is", "are", "was", "were",
    "has", "have", "having", "wearing", "featuring", "a", "an", "the",
]
_STOP_ALT = "|".join(STOP_FOR_LOOKBACK)

# local noun-phrase excision: optional article + up to 3 preceding modifier
# words (excluding STOP_FOR_LOOKBACK, so it can't cross "with"/"and"/a verb)
# + the strip term itself. Handles "black choker", "a silver necklace",
# "large purple canine ears".
LOCAL_PHRASE_RE = re.compile(
    r"\b(?:a\s+|an\s+|the\s+)?"
    r"(?:(?!\b(?:" + _STOP_ALT + r")\b)[a-zA-Z]+[\s-]+){0,3}?"
    r"(?:" + STRIP_CORE + r")"
    # optional bounded trailing PP directly describing the just-matched strip
    # term ("eyepatch OVER HIS LEFT EYE") -- without this, removing just the
    # noun leaves a dangling preposition with nothing left to attach to.
    r"(?:\s+(?:over|on|around|under|across|near)\s+"
    r"(?:(?!\b(?:" + _STOP_ALT + r")\b)[a-zA-Z'-]+\s*){1,4})?",
    re.I,
)

_PP_STOP = (r"and|who|that|which|while|is\b|are\b|was\b|were\b|stands?\b|sits?\b|"
            r"kneels?\b|leans?\b|walks?\b|gazes?\b|reaches?\b|stares?\b|holds?\b|looks?\b|"
            r"being\b|has\b|have\b")

# bounded "with/having/featuring X" trailing phrase -- capped at 6 words and
# stops before a verb, so it can never swallow the sentence's real predicate
# (this was the "tail-gun stands turned away" / "necklace who is pulling him"
# catastrophic-loss bug class in earlier iterations). Only removed wholesale
# if X itself contains a strip term.
WITH_PHRASE_RE = re.compile(
    r"\s+(?:with|having|featuring)\s+"
    r"(?:(?!\b(?:" + _PP_STOP + r")\b)[a-zA-Z'-]+\s*){1,6}",
    re.I,
)

# "wrapped in bandages" / "covered in tattoos" / "bound by chains" style: a
# participle is REQUIRED before the preposition, so bare scene prepositions
# ("standing in a park") are never touched -- only removed if the bounded
# phrase itself contains strip content.
PARTICIPLE_PP_RE = re.compile(
    r"\s+(?:wrapped|covered|bound|tied|accompanied|surrounded|draped|dressed)\s+"
    r"(?:in|by|near|behind|on|around|under|with)\s+"
    r"(?:(?!\b(?:" + _PP_STOP + r")\b)[a-zA-Z'-]+\s*){1,6}",
    re.I,
)

# hair/eye/skin colour palette -- kept for text-channel per Wave-1 1a/1c
# (31_tagprior.py HAIR_COLOR/EYE_COLOR), extended with common English
# synonyms not in the wd14 tag vocabulary (auburn, hazel, platinum, ...).
COLOR_WORDS = [
    "white", "silver", "grey", "gray", "black", "brown", "blonde", "blond", "red", "pink",
    "purple", "blue", "aqua", "green", "orange", "multicolored", "multi-colored",
    "auburn", "chestnut", "platinum", "strawberry blonde", "maroon", "crimson",
    "burgundy", "violet", "lavender", "teal", "amber", "hazel", "yellow", "ash blond",
    "dark red", "light purple", "light blue", "dark brown", "light brown",
]
KEEP_COLOR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in COLOR_WORDS) + r")\b[\s-]*"
    r"(?:hair|haired|eyes?|eye|skin(?:ned)?|complexion|women|woman|men|man|girl|boy|"
    r"lady|person|character|guy|female|male)",
    re.I,
)

GOV_RE = re.compile(r"^(wearing|with|having|featuring)\s+(.*)$", re.I)
LEADING_AND_RE = re.compile(r"^and\s+", re.I)
DANGLING_CONNECTOR_RE = re.compile(
    r"\b(?:with|wearing|having|featuring|and)\s*(?=\.|$)", re.I,
)
ORPHAN_TOKENS = {"one", "another", "others", "some", "it", "he", "she", "they",
                 "a", "an", "the", "this", "that", "these", "those"}

# marks the exact position of an excised phrase during cleanup, so repairs can
# distinguish "and" adjacent to a deletion from a naturally-occurring "and".
DELETION_SENTINEL = "\x00"

ACTION_VERBS = [
    "sitting", "standing", "kneeling", "holding", "looking", "leaning", "walking", "gazing",
    "reaching", "staring", "embracing", "embraced", "grappling", "named", "surrounded",
    "smiling", "blushing", "talking", "reading", "engaged", "extending", "pulling",
    "cuddling", "snuggling", r"is\b", r"are\b", r"stands?\b", r"sits?\b", r"kneels?\b",
    r"leans?\b", r"walks?\b", r"gazes?\b", r"reaches?\b", r"stares?\b", r"holds?\b",
    r"looks?\b", "being", "stood", "sat", "knelt", "leaned", "walked", "gazed", "reached",
    "stared", "held", "laughing", "crying", "screaming", "whispering", "gasping",
    "sighing", "nodding", "shaking", "trembling", "flushing", "giggling", "chuckling",
    "grinning", "frowning", "glaring", "winking", "kissing", "hugging", "touching",
    "grabbing", "pinning", "restraining", "comforting", "wrapped", "draped", "perched",
    "seated", "resting", "lying", "reclining", "posed", "posing", "showing", "revealing",
    "lifting", "raising", "turning", "tilting", "bowing", "waving", "pointing",
    "gesturing", "examining", "working", "cooking", "writing", "drawing", "playing",
    "dancing", "running", "jumping", "climbing", "swimming", "driving", "riding",
    "flying", "offering", "aiming", "depict", "depicting", r"stand\b",
]
ACTION_VERB_RE = re.compile(r"\b(?:" + "|".join(ACTION_VERBS) + r")", re.I)

# generic -ing words that are nouns/adjectives, not the scene's action verb
# (would otherwise false-trigger the gerund-based action detector below).
GERUND_BLOCKLIST = {
    "clothing", "nothing", "something", "anything", "everything", "morning",
    "evening", "ring", "rings", "earring", "earrings", "piercing", "piercings",
    "lightning", "ceiling", "building", "wedding", "stocking", "stockings",
    "legging", "leggings", "icing", "siding", "outfit", "styling", "warring",
}


def has_gerund_action(clause):
    for w in re.findall(r"\b(\w+ing)\b", clause):
        if w.lower() not in GERUND_BLOCKLIST:
            return True
    return False


def clause_has_action(raw):
    """True if `raw` looks like it carries its own predicate (scene/action),
    as opposed to being a bare appearance-descriptor list item. Used to decide
    when a pending governing verb ("wearing") should stop looking for a later
    list item to reattach to."""
    if raw.rstrip().endswith("."):
        return True
    if ACTION_VERB_RE.search(raw):
        return True
    return has_gerund_action(raw)


def excise(content):
    """Remove strip-noun-phrases from `content`. Returns (cleaned, deleted_list).
    If nothing matched, returns the ORIGINAL content untouched (verbatim) so
    unmodified text is never needlessly rewritten."""
    deleted = []

    def pp_repl(m):
        whole = m.group(0)
        if STRIP_RE.search(whole):
            deleted.append(whole.strip())
            return ""
        return whole

    out = PARTICIPLE_PP_RE.sub(pp_repl, content)
    out = WITH_PHRASE_RE.sub(pp_repl, out)

    def local_repl(m):
        deleted.append(m.group(0).strip())
        return DELETION_SENTINEL

    out = LOCAL_PHRASE_RE.sub(local_repl, out)

    if not deleted:
        return content, []

    # deletion-point-aware repair BEFORE the sentinel is erased: a coordinating
    # "and" whose right conjunct was just deleted and whose continuation is a
    # participle re-binds wrongly ("with blond hair and [deleted] wearing a hat"
    # -> "with blond hair and wearing a hat"); drop that "and" only when it sits
    # immediately at a deletion point, so naturally-occurring "and wearing"
    # elsewhere in the clause is never touched.
    out = re.sub(
        r"\band\s+" + DELETION_SENTINEL + r"\s*(?=(?:wearing|holding|carrying|having|with)\b)",
        " ", out, flags=re.I)
    out = out.replace(DELETION_SENTINEL, " ")

    out = re.sub(r"\s*,\s*,+", ", ", out)
    out = re.sub(r"^\s*,\s*", "", out)
    out = re.sub(r"\s*,\s*$", "", out)
    out = re.sub(r"^\s*and\s+", "", out, flags=re.I)
    out = re.sub(r",\s*and\s*,", ",", out, flags=re.I)
    # "wearing [removed] and Y" -> "wearing and Y" -> "wearing Y": the same
    # collapse as the leading-"and" strip above, but for when the removed
    # item wasn't at the very start of `content` (e.g. "a man with X hair
    # wearing [eyepatch removed] and a black suit").
    out = re.sub(r"\b(wearing|with|having|featuring)\s+and\s+", r"\1 ", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    out = DANGLING_CONNECTOR_RE.sub("", out).strip()
    out = re.sub(r"\s+\.", ".", out)  # stray space left before a period
    out = out.strip(" ,")
    # a lone pronoun/quantifier/article (optionally followed by a still-attached
    # trailing sentence) left after excision carries no content of its own --
    # e.g. "one wearing an orange mask." -> "one." -> drop the "one", keep any
    # trailing sentence that was glued onto the same clause.
    bare = re.split(r"[.,]", out, maxsplit=1)[0].strip()
    if bare.lower() in ORPHAN_TOKENS:
        out = out[len(bare):].lstrip(" .,")
    if not re.search(r"[a-zA-Z]", out):
        out = ""  # nothing but stray punctuation/whitespace survived
    return out, deleted


def looks_like_color_descriptor(clause):
    return bool(KEEP_COLOR_RE.search(clause)) and not bool(STRIP_RE.search(clause))


def scrub(text):
    """Clause-level scrub of one t2i_instruction string. Returns
    (scrubbed_text, deleted_phrases). Splits on top-level commas; tracks a
    pending governing verb ("wearing"/"with") across a run of bare list-item
    clauses so that dropping the FIRST item of an Oxford list doesn't also
    drop the verb for the items after it (e.g. "wearing choker[dropped],
    earrings[dropped], and a top" -> "wearing a top", not just "a top")."""
    clauses = [c for c in re.split(r",\s*", text) if c.strip()]
    emitted = []
    deleted_all = []
    active_gov = None
    active_gov_used = True

    for raw in clauses:
        has_strip = bool(STRIP_RE.search(raw))
        stripped_and = LEADING_AND_RE.sub("", raw)
        m = GOV_RE.match(stripped_and)

        if m:
            gov, content = m.group(1), m.group(2)
            cleaned, deleted = excise(content) if has_strip else (content, [])
            deleted_all.extend(deleted)
            active_gov = gov
            if cleaned:
                if not deleted:
                    emitted.append(raw)
                elif cleaned.startswith("."):
                    # excision ate through a sentence boundary; what's left is
                    # the start of the NEXT sentence, not something this gov
                    # word governs.
                    emitted.append(cleaned.lstrip(". "))
                else:
                    emitted.append(f"{gov} {cleaned}")
                active_gov_used = True
            else:
                active_gov_used = False  # pending: may reattach to a later bare item
            continue

        has_action = clause_has_action(raw)
        cleaned, deleted = excise(stripped_and) if has_strip else (stripped_and, [])
        deleted_all.extend(deleted)

        if has_action:
            if cleaned:
                if not deleted:
                    emitted.append(raw)
                elif cleaned.startswith("."):
                    emitted.append(cleaned.lstrip(". "))
                else:
                    text_out = cleaned
                    if raw.lower().startswith("and ") and not text_out.lower().startswith("and "):
                        text_out = "and " + text_out
                    emitted.append(text_out)
            active_gov = None
            active_gov_used = True
            continue

        # bare continuation item: no gov word, no action verb of its own
        if cleaned:
            if cleaned.startswith("."):
                emitted.append(cleaned.lstrip(". "))
                active_gov = None
                active_gov_used = True
            else:
                eligible = (active_gov is not None and not active_gov_used
                            and not looks_like_color_descriptor(raw))
                if eligible:
                    emitted.append(f"{active_gov} {cleaned}")
                    active_gov_used = True
                else:
                    emitted.append(raw if not deleted else cleaned)
        # else: nothing survived, emit nothing; active_gov stays pending as-is

    out = ", ".join(emitted)
    out = re.sub(r"^\s*and\s+", "", out)
    out = re.sub(r",\s*,+", ", ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    out = out.strip(" ,")
    return out, deleted_all


# ============================================================================
# stage: review
# ============================================================================

def stage_review():
    _, captions, _, three_cond = build_pool()
    candidates = sorted(three_cond)  # sort BEFORE seeding -- set order is not
                                      # stable across runs under hash randomization
    rng = random.Random(REVIEW_SEED)
    rng.shuffle(candidates)
    sample = sorted(candidates[:REVIEW_N], key=int)

    lines = [
        "# Caption scrub review (20 samples)",
        "",
        f"Sampled from the 3-condition intersection (n={len(three_cond)}), "
        f"random.Random({REVIEW_SEED}).shuffle() on the sorted uid list, first "
        f"{REVIEW_N} taken, then re-sorted by uid for readability.",
        "",
    ]
    for uid in sample:
        raw = captions[uid]
        scrubbed, deleted = scrub(raw)
        print(f"u{uid}: {len(deleted)} phrase(s) deleted", flush=True)
        lines.append(f"## uid {uid}")
        lines.append("")
        lines.append(f"**RAW**: {raw}")
        lines.append("")
        lines.append(f"**SCRUBBED**: {scrubbed}")
        lines.append("")
        lines.append(f"**DELETED**: {deleted if deleted else '(none)'}")
        lines.append("")
    with open(REVIEW_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {REVIEW_MD} ({len(sample)} samples)", flush=True)


# ============================================================================
# stage: build (NOT run by this task -- written for the main session to run
# after --stage review passes human sign-off)
# ============================================================================

def stage_build():
    pool_minus_exclude, captions, exclude, three_cond = build_pool()
    uids = sorted(three_cond, key=int)

    n_deleted_total = 0
    scrub_len_total = 0
    rows = []
    for uid in uids:
        raw = captions[uid]
        scrubbed, deleted = scrub(raw)
        n_deleted_total += len(deleted)
        scrub_len_total += len(scrubbed)
        rows.append({
            "uid": uid,
            "ref": os.path.join(IMG_DIR, f"u{uid}_ref.png"),
            "target": os.path.join(IMG_DIR, f"u{uid}_target.png"),
            "caption_raw": raw,
            "caption_scrubbed": scrubbed,
        })

    with open(MANIFEST_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {MANIFEST_JSONL} ({len(rows)} rows)", flush=True)
    print(f"total deleted phrases = {n_deleted_total}, "
          f"avg per row = {n_deleted_total / max(1, len(rows)):.3f}", flush=True)
    print(f"avg scrubbed caption length (chars) = "
          f"{scrub_len_total / max(1, len(rows)):.1f}", flush=True)

    print("\nleak-phrase scan (must be 0 hits on all 8):", flush=True)
    hit_total = 0
    for phrase in LEAK_PHRASES:
        hits = sum(1 for r in rows if phrase in r["caption_scrubbed"].lower())
        hit_total += hits
        print(f"  {'OK ' if hits == 0 else 'FAIL'} {hits:5d}  {phrase!r}", flush=True)
    if hit_total == 0:
        print("leak scan: 0/8 phrases hit, 0 total occurrences -- clean.", flush=True)
    else:
        print(f"leak scan: {hit_total} total occurrences -- INVESTIGATE before training.",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["review", "build"])
    a = ap.parse_args()
    if a.stage == "review":
        stage_review()
    else:
        stage_build()
