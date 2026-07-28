#!/usr/bin/env python
"""
49_mixed_manifest.py -- build a "solo-balanced mixed" training manifest out of
train_manifest_7k5.jsonl (5945 rows), as a follow-on to probe/48_clean_manifest.py.

Background: 48_clean_manifest.py's strict solo-only filter (WD14 solo>=0.5 +
all multi-person tags <0.35 + no F/M-pair tag pair + caption doesn't read as
multi/intimate) keeps only ~369/5945 rows (data/train_manifest_7k5_clean.jsonl)
-- 91% of the corpus is genuinely two-person target images (couples, embraces,
confrontations, bedroom scenes, ...). 369 rows is too thin to train on alone.
The fix isn't "train solo-only on 369 rows" or "train on all 5945 rows
including caption/image mismatches" -- it's a mix: solo rows (oversampled to
compensate for their small count) PLUS the subset of two-person rows whose
caption_scrubbed *faithfully describes* the second person, so the model has
positive supervision for "when the caption says there's someone else, render
someone else" instead of being punished for a second person it was never
told about. Rows where the image is two-person but the caption stays silent
about it ("poison": caption reads like a solo portrait, WD14 says otherwise)
are dropped outright -- keeping them would teach the ref-conditioned model to
hallucinate an uncaptioned second character.

WD14 tags for every one of the 5945 targets are already cached (produced by
48_clean_manifest.py's run) at
/mnt/local/cherry_out/trainlight/wd14_target_cache.jsonl -- one JSON line per
uid, {"uid", "rating", "tags"}, where `tags` only carries the person-count
tags (solo/1girl/1boy/2girls/2boys/multiple_girls/multiple_boys) at
general_threshold=0.3. This script is READ-ONLY against that cache (all 5945
uids are present, verified before writing this script) -- no imgutils / WD14
/ GPU calls here, just regex over caption_scrubbed and cached tag lookups.

Four-way per-row classification (mutually exclusive, in this priority order):

  1. nsfw (dropped) -- rating['explicit']>=0.5 or rating['questionable']>=0.5.

  2. solo_clean (kept, oversampled) -- tags['solo']>=0.5, AND
     max(2girls,2boys,multiple_girls,multiple_boys) < 0.35, AND NOT
     (1girl>=0.8 AND 1boy>=0.8) [a tighter, deliberately different bar than
     48_clean_manifest.py's own min(1girl,1boy)>=0.35 -- this script defines
     its own solo_clean per the task spec, not a re-import of 48's decision],
     AND caption_scrubbed does NOT match 48_clean_manifest.CAPTION_MULTI_RE
     (imported verbatim from probe/48_clean_manifest.py -- same "does this
     caption read as multi-person/intimate" gate, reused rather than copied
     so the two scripts can never drift out of sync). Yields 371 rows here
     (vs 48's 369 -- close, expected, different threshold on the 1girl/1boy
     condition).

  3. multi_faithful (kept, used as-is) -- not solo_clean, not nsfw, AND
     caption_scrubbed passes the FAITHFUL detector below (some independent
     signal says the caption itself, not just WD14, describes >=2 people).

  4. multi_poison (dropped) -- not solo_clean, not nsfw, caption fails the
     FAITHFUL detector -- i.e. WD14 says multi-person (or the caption itself
     already tripped 48's multi/intimate gate) but the caption text gives no
     independent textual evidence of a second person. This is exactly the
     "image says two people, caption says one" mismatch that would poison a
     single-identity ref-conditioned LoRA.

FAITHFUL detector -- design + tuning story: started from the task's three
suggested signal families ((a) reuse 48's CAPTION_MULTI_RE, (b) opposite-
gender word co-occurrence, (c) contact/proximity verbs with a person object),
then iterated twelve rounds against manual reads of 25-sample faithful/
poison pulls (7 full audit rounds, seeds 42/7/99/555/777 among them; see
below for the final read) until faithful precision was clean across every
round and the poison pile had no glaring "should be saved" case left.
Concretely, a caption is `multi_faithful` if ANY of:
  - BASE48_NO_SURROUND: 48's own multi/intimate regex, MINUS its bare
    "surrounded by" clause -- audit round 3 caught uid=2951 ("playing a
    game, surrounded by monitors and clutter") being promoted to faithful
    on a caption that never mentions a second person at all. "surrounded
    by" is instead its own clause (SURROUND_RE) gated on the surrounded-by
    object actually being a person noun ("surrounded by several other
    girls" yes, "surrounded by monitors" no).
  - CONTACT_RE: embrace/kiss/carry/straddle/pin/grip/press + "being held" +
    "holding her/him/his hand/her hand" + "holding a/an/the ... person/
    figure" (48's own base pattern already covers "holding a woman/man/
    girl/boy" -- this adds the person/figure object 48 didn't need) +
    "holding hands with" + a
    proximity preposition (against/beside/next to/toward/towards/facing/
    close to) whose object is explicitly a man/woman/girl/boy/person/
    figure (NOT bare "leaning against" -- 48's own docstring already
    flagged that as mostly walls/doorways, so the object noun is required)
    + talking/chatting/arguing/etc "to/with a/an/the ...".
  - gender_pair: an opposite-gender word from MALE_RE and one from
    FEMALE_RE both present. Extended past the task's suggested word lists
    with plurals (men/women/girls/boys) and title nouns that this corpus's
    fantasy/romance captions lean on instead of common nouns (prince/king/
    lord, princess/queen/lady/demoness/goddess/empress/duchess/priestess/
    sorceress/huntress) -- audit round 2/3 caught real misses like "Prince
    Arthur... talking to a brown-haired woman" and "Azraella, a demoness...
    a man in a white shirt".
  - repeat_male_noun / repeat_female_noun: >=2 occurrences (anywhere,
    pronouns he/she/his/her excluded) of ANY noun from the male or female
    word family respectively -- catches two DIFFERENT nouns from the same
    gender family too ("a pale male... a shirtless man" -- male + man,
    not the same literal word, gender_pair can't see it since there's no
    opposite-gender word). Strictly broader than an earlier per-exact-word
    "repeat:man"/"repeat:woman" version that missed this.
  - repeat:{person,figure,individual,character} -- the SAME singular noun
    occurring >=2x almost always means two different entities were
    introduced with it (a caption's own subject is typically named once
    via that noun then referred to by pronoun after; a second "a figure"/
    "another character"-shaped mention is a second entity). Catches
    same-gender pairs gender_pair structurally can't ("a woman... behind a
    seated woman" -- covered by repeat_female_noun above; "a hyena
    character... a hyena character" -- covered by repeat:character here).
  - marker: "another/a second/a third/a smaller/the other/second/other"
    directly followed by man/woman/girl/boy/person/figure/individual/
    character(s) -- explicit second-entity language ("another dark-haired
    man", "the other woman", "a smaller human figure").
  - plural (PLURAL_RE / BARE_PLURAL_WORDS): a bare plural or inherently-
    multiple person-noun (men/women/girls/boys/people/students/guards/
    soldiers/nurses/doctors/officers/twins/siblings/princesses/kings/... --
    the plural of MALE_RE/FEMALE_RE's title nouns is included here too,
    same "plural is unambiguous" logic) anywhere in the caption; plural
    already implies >1 person independent of what the MAIN
    subject's own gender is ("watched by three men", "a line of women",
    "Two other men stand nearby"). Deliberately does NOT include the
    singular of the 8 "core" words (man/woman/girl/boy/person/figure/
    individual/male/female) -- audit round v9 caught a real regression
    here: an earlier version's word list used "wom[ae]n" (matching BOTH
    "woman" and "women") and bare "man" to feed both this bare-anywhere
    check and the "surrounded by X" object check off one shared list, and
    reusing it for the *bare* check let ordinary solo self-descriptions
    ("A person with... " / "a man with...") trigger through the back door,
    faithful count spiked from ~5000 to ~5390 on bogus grounds. Fixed by
    giving "surrounded by" its own separate, more permissive object-word
    list (SURROUND_OBJECT_WORDS -- safe to include singular core nouns
    there specifically because "surrounded by" is structurally never
    self-referential) and keeping the bare-anywhere PLURAL_RE strictly to
    unambiguous plurals / collective-noun roles.
  - characters / staff / family: three words that carry real "second
    person" signal in this corpus but ALSO have a demonstrated non-person
    usage here, so each gets a narrow guarded pattern instead of joining
    the bare word list: "characters" excludes the (?<!japanese )(?<!chinese
    )(?<!korean )(?<!kanji ) case (on-screen text, "Visible text: ...
    Japanese characters ..." -- ~1/6 of all "characters" hits in this
    corpus); "staff" requires "medical/kitchen/hotel/... staff" or "staff
    members/stands/observes/..." (bare "staff" is a magic/fantasy STAFF,
    the weapon, in roughly half its hits here -- "holding a wooden
    staff"); "family" requires "family members/gathers/watches/...", "with
    his/her/their family", or "family stands/sits" (bare "family" also
    hits depicted OBJECTS here -- "a framed family photograph", "a family
    drawing" -- and visible-text/proper-noun uses -- "FAMILY ESTATE").
  - relational: someone/stranger/companion/subordinate -- words that in
    this corpus's caption style are essentially never used for the
    caption's own opening subject, only for someone else in frame.
  - single_noun (SINGLE_NOUN_RE): a SINGLE occurrence of person/figure/
    individual/man/woman/girl/boy is only trusted as evidence of a second
    entity if it is introduced by a relational cue (to/toward/at/behind/
    before/beside/near/with/against/of/where/while/as/who/whom/nearby/in
    front of/flanked by/accompanied by + a/an/the + <=4 modifier words +
    noun) -- e.g. "the collar of a young man", "looking at a dark-haired
    person". A bare opening "A person with spiky hair, leaning against a
    door" (no relational cue before it) does NOT count -- audit round 2
    caught two hard false positives this way (uid=3936, uid=8110:
    captions that describe a SINGLE character via "a person"/"a shadowy
    figure" as their own opening self-description, no second person
    anywhere in the text) and a bare-occurrence trigger would have
    silently promoted both (and everything like them) to faithful. Man/
    woman/girl/boy were folded into this same cue-gated rule rather than
    given their own bare-single-mention path for the same reason (they're
    this corpus's single most common solo self-description vocabulary).
    Not triggered by possessive "her/his/their figure" either (a
    body-shape idiom here -- "showing off her figure" -- not a person).

  Known accepted misses (residual multi_poison rows that arguably describe
  a second person but the regex doesn't catch, left as-is because each
  fix attempted here traded away more precision than the recall it bought,
  or because a real fix would need actual NLP/coref, not more regex):
  a second person introduced by PROPER NAME only with no common noun at all
  ("Android 17 and Android 18 stand...", "talking happily with Ochaco
  Uraraka") -- would need named-entity recognition, not lexical matching;
  a second entity introduced as the opening subject of a NEW sentence
  ("... Ivan is sitting up in a hospital bed. A person with blue spiky
  hair is kneeling beside the bed, hugging Ivan...") -- indistinguishable,
  from single_noun's cue-gate alone, from the same construction used for a
  truly solo caption's OWN subject, so intentionally not trusted (mirror
  image of the uid=3936/8110 fix above -- the same design choice that
  fixes those false positives costs these false negatives); creature/
  monster second subjects (symbiote, flame atronach, vampire -- captioned
  but not a person-noun); occupational role nouns used only once ("the
  driver", "police officers", "military personnel" as the ONLY cue --
  many of these words are also how captions self-describe their OWN
  subject, so they aren't safe as a bare trigger); a handful of
  caption_scrubbed text-corruption artifacts from the upstream scrub step
  gluing words together ("girland", "manand", "womanwoman") that break
  \\b-anchored regex by construction.

Output construction (data/train_manifest_mix.jsonl):
  - solo_clean rows: caption_scrubbed gets "solo, single character, "
    prepended (skipped if the caption already starts with "solo",
    case-insensitively) and the WHOLE row (uid/ref/target/caption_raw/
    caption_scrubbed) is duplicated K times verbatim with that modified
    caption_scrubbed -- caption_raw is left untouched.
  - multi_faithful rows: used as-is, no modification.
  - K = min(8, ceil(n_faithful / n_solo_clean)). If K hits the cap of 8 and
    n_solo_clean*K is still short of n_faithful (this run: 371*8=2968 <<
    4991), multi_faithful is randomly downsampled (random.seed(0)) down to
    exactly n_solo_clean*K so the two sides land at ~1:1.
  - All rows concatenated, random.seed(0) + random.shuffle, then written.

Run (no GPU / imgutils needed -- pure regex + cached-tag lookup):
  source ../env.sh && $PY 49_mixed_manifest.py
"""
import os
import re
import json
import math
import random
import importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MANIFEST_IN = os.path.join(DATA, "train_manifest_7k5.jsonl")
MANIFEST_OUT = os.path.join(DATA, "train_manifest_mix.jsonl")
REPORT_OUT = os.path.join(DATA, "manifest_mix_report.json")

CACHE_PATH = "/mnt/local/cherry_out/trainlight/wd14_target_cache.jsonl"

SOLO_THR = 0.5          # tags['solo'] >= this
MULTI_THR = 0.35        # max(2girls,2boys,multiple_girls,multiple_boys) < this
PAIR_EXTREME_THR = 0.8  # NOT (1girl>=this AND 1boy>=this)
NSFW_THR = 0.5          # rating['explicit'] / rating['questionable'] >= this -> drop
K_CAP = 8
SEED = 0
SOLO_PREFIX = "solo, single character, "


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m48 = _load("m48", "48_clean_manifest.py")
CAPTION_MULTI_RE = m48.CAPTION_MULTI_RE  # reused verbatim, solo_clean gate only

STYLE_RE = re.compile(r"[Ii]n (?:an? )?([A-Za-z_0-9]+?)[ _]style")

# --- multi_faithful detector -----------------------------------------------
# 48's own multi/intimate regex, minus bare "surrounded by" (see docstring:
# that clause alone has a real non-person false-positive rate in this corpus
# -- "surrounded by monitors and clutter" -- so it's split out below and
# gated on a person-plural object).
BASE48_NO_SURROUND = re.compile(
    "|".join([
        r"intimate", r"kiss", r"embrac", r"cuddl",
        r"in his arms", r"in her arms",
        r"holding a (?:woman|man|girl|boy)",
        r"while a (?:woman|man|girl|boy)",
        r"another (?:woman|man|girl|boy)",
        r"nuzzl", r"caress", r"snuggl", r"spooning", r"nestled against",
        r"romantic", r"wrapp\w+ (?:his|her|their) arms around",
        r"\btwo (?:girls|boys|women|men)\b",
        r"making out", r"entwined",
    ]), re.IGNORECASE,
)

# Object-word list for "surrounded by <X>" specifically -- safe to include
# SINGULAR core nouns (man/woman/girl/boy/figure/character) here because
# "surrounded by" is structurally never self-referential (nobody's own
# opening self-description is "surrounded by a woman"). This list is
# DELIBERATELY separate from BARE_PLURAL_WORDS below (see that constant's
# comment / the docstring's "plural" bullet for why sharing one list
# between the two contexts was a real bug caught in tuning).
SURROUND_OBJECT_WORDS = (
    r"girls?|boys?|wom[ae]n|men|man|people|students?|guests?|classmates?|"
    r"teammates?|crewmates?|friends?|family|colleagues?|guards?|staff|"
    r"onlookers?|characters?|figures?|goblins?|customers?|servants?|nobles?|"
    r"spectators?|adventurers?|companions?|soldiers?|warriors?|knights?|"
    r"priests?|monks?|nuns?|waiters?|maids?|butlers?|gang members?|crowd|"
    r"audience|dancers?|attendants?|officers?|nurses?|doctors?|workers?|"
    r"twins?|siblings?|bystanders?"
)
SURROUND_RE = re.compile(
    r"surrounded by\b(?:[^,.]{0,40}?\b(?:" + SURROUND_OBJECT_WORDS + r")\b)",
    re.IGNORECASE,
)

# Bare-anywhere-in-caption plural / inherently-multiple person-noun trigger.
# MUST NOT include the singular of the 8 "core" words (man/woman/girl/boy/
# person/figure/individual/male/female) -- those are this corpus's default
# vocabulary for describing ANY solo protagonist, so a bare singular hit is
# not evidence of a SECOND person. (Those 8 are covered elsewhere: gender_
# pair / repeat_*_noun / marker / the cue-gated half of single_noun below.)
BARE_PLURAL_WORDS = (
    r"men|women|girls|boys|males|females|persons|figures|individuals|"
    r"people|kids|children|ladies|gentlemen|crowd|audience|"
    r"students?|guests?|classmates?|teammates?|crewmates?|friends?|"
    r"colleagues?|guards?|onlookers?|goblins?|customers?|servants?|nobles?|"
    r"spectators?|adventurers?|soldiers?|warriors?|knights?|priests?|"
    r"monks?|nuns?|waiters?|maids?|butlers?|gang members?|dancers?|"
    r"attendants?|officers?|nurses?|doctors?|workers?|twins?|siblings?|"
    r"bystanders?|"
    # plural forms of the MALE_RE/FEMALE_RE *title* nouns -- singular
    # "princess"/"king"/etc are core-unsafe (bare trigger risk, same as
    # man/woman), but a PLURAL title noun ("princesses", "kings") is just
    # as unambiguous a >1-person signal as "women"/"men" is
    r"princesses|queens|goddesses|empresses|duchesses|priestesses|"
    r"sorceresses|huntresses|kings|lords|princes"
)
PLURAL_RE = re.compile(r"\b(?:" + BARE_PLURAL_WORDS + r")\b", re.IGNORECASE)

# "characters" / "staff" / "family": real second-person signal in this
# corpus, but each ALSO has a demonstrated non-person usage here (grepped
# every occurrence in the 5945-row corpus before writing these), so each
# gets a narrow guarded pattern instead of joining BARE_PLURAL_WORDS.
CHARACTERS_RE = re.compile(
    r"(?<!japanese )(?<!chinese )(?<!korean )(?<!kanji )characters\b",
    re.IGNORECASE,
)
STAFF_RE = re.compile(
    r"\b(?:medical|kitchen|hotel|store|zoo|hospital|security|office)\s+staff\b|"
    r"\bstaff\s+(?:members?|stand\w*|observ\w*|operat\w*|treat\w*|pack\w*|"
    r"sleep\w*)\b",
    re.IGNORECASE,
)
FAMILY_RE = re.compile(
    r"\bfamily\s+(?:members?|gathers?|gathered|watches?|watching|group)\b|"
    r"\bwith\s+(?:his|her|their|my)\s+family\b|"
    r"\bfamily\s+(?:stands?|standing|sits?|sitting)\b",
    re.IGNORECASE,
)

MALE_RE = re.compile(
    r"\b(he|his|him|man|men|male|males|boy|boys|boyfriend|boyfriends|"
    r"prince|king|lord)\b", re.IGNORECASE,
)
FEMALE_RE = re.compile(
    r"\b(she|her|woman|women|female|females|girl|girls|girlfriend|girlfriends|"
    r"princess|queen|lady|demoness|goddess|empress|duchess|priestess|"
    r"sorceress|huntress)\b", re.IGNORECASE,
)
# noun-only variants of the two above (pronouns he/his/him/she/her dropped)
# for the repeat_male_noun / repeat_female_noun signal -- pronoun repeats
# ("He stands... He looks...") say nothing about a second entity, but two
# noun mentions from the same gender family (even two DIFFERENT nouns,
# e.g. "a pale male... a shirtless man") almost always do.
MALE_NOUN_RE = re.compile(
    r"\b(man|men|male|males|boy|boys|boyfriend|boyfriends|prince|king|lord)\b",
    re.IGNORECASE,
)
FEMALE_NOUN_RE = re.compile(
    r"\b(woman|women|female|females|girl|girls|girlfriend|girlfriends|"
    r"princess|queen|lady|demoness|goddess|empress|duchess|priestess|"
    r"sorceress|huntress)\b", re.IGNORECASE,
)

CONTACT_RE = re.compile(
    r"\bembrac\w*|\bkiss\w*|\bcarr(?:ied|ying)\b|\bstraddl\w*|\bpinn(?:ed|ing)\b|"
    r"\bgrip(?:ping|ped)\b|\bpress(?:ing|ed)\b|"
    r"being held|holding (?:her|him|his hand|her hand)\b|"
    r"holding (?:a|an|the)\s+(?:[a-z]+[- ]){0,3}(?:person|figure|guy)\b|"
    r"\bhold(?:ing|s)\s+hands?\s+with\b|"
    r"(?:against|beside|next to|toward|towards|facing|close to)\s+(?:a|an|the)\s+"
    r"(?:[a-z]+[- ]){0,3}(?:man|woman|girl|boy|person|figure|guy)\b|"
    r"\b(?:talking|speaks?|speaking|chatting|conversing|arguing|shouting|"
    r"whispering)\s+(?:to|with)\s+(?:a|an|the)\b",
    re.IGNORECASE,
)

REPEAT_NOUNS = ["person", "figure", "individual", "character"]
REPEAT_RES = {n: re.compile(r"\b" + n + r"\b", re.IGNORECASE) for n in REPEAT_NOUNS}

MARKER_RE = re.compile(
    r"\b(?:another|a second|a third|a smaller|the other|second|other)\s+"
    r"(?:[a-z]+[- ]){0,3}(?:man|woman|girl|boy|person|figure|individual|"
    r"character)s?\b",
    re.IGNORECASE,
)

RELATIONAL_RE = re.compile(r"\b(someone|stranger|companion|subordinate)\b", re.IGNORECASE)

# Relational-cue-prefixed single mention. Applies to person/figure/
# individual AND, by the same safe construction, man/woman/girl/boy
# (catches e.g. "the collar of a young man") -- safe because the cue
# requirement structurally excludes a caption's own opening self-
# description (which never has "of"/"toward"/"beside"/... right before it).
SINGLE_NOUN_RE = re.compile(
    r"\b(?:to|toward|towards|at|behind|before|beside|near|with|against|of|"
    r"where|while|as|who|whom|nearby|in front of|flanked by|accompanied by)"
    r"\s+(?:a|an|the)\s+(?:[a-z]+[- ]){0,4}"
    r"(?:person|figure|individual|man|woman|girl|boy|guy)s?\b",
    re.IGNORECASE,
)


def faithful_reasons(caption):
    """Return the list of independent signals (possibly empty) that make
    `caption` read as faithfully describing >=2 people. Non-empty ->
    multi_faithful; empty (on a row that's already not solo_clean/nsfw) ->
    multi_poison."""
    reasons = []
    if BASE48_NO_SURROUND.search(caption):
        reasons.append("base48")
    if SURROUND_RE.search(caption):
        reasons.append("surrounded_by_people")
    if CONTACT_RE.search(caption):
        reasons.append("contact")
    if MALE_RE.search(caption) and FEMALE_RE.search(caption):
        reasons.append("gender_pair")
    for noun, rex in REPEAT_RES.items():
        if len(rex.findall(caption)) >= 2:
            reasons.append(f"repeat:{noun}")
    if len(MALE_NOUN_RE.findall(caption)) >= 2:
        reasons.append("repeat_male_noun")
    if len(FEMALE_NOUN_RE.findall(caption)) >= 2:
        reasons.append("repeat_female_noun")
    if MARKER_RE.search(caption):
        reasons.append("marker")
    if PLURAL_RE.search(caption):
        reasons.append("plural")
    if CHARACTERS_RE.search(caption):
        reasons.append("characters")
    if STAFF_RE.search(caption):
        reasons.append("staff")
    if FAMILY_RE.search(caption):
        reasons.append("family")
    if RELATIONAL_RE.search(caption):
        reasons.append("relational")
    if SINGLE_NOUN_RE.search(caption):
        reasons.append("single_noun")
    return reasons


def classify_row(row, wd14_rec):
    """Returns (category, reasons) where category is one of
    nsfw/solo_clean/multi_faithful/multi_poison and reasons is the
    faithful_reasons() list (only non-empty for multi_faithful)."""
    rating = wd14_rec.get("rating", {})
    if rating.get("explicit", 0.0) >= NSFW_THR or rating.get("questionable", 0.0) >= NSFW_THR:
        return "nsfw", []

    tags = wd14_rec.get("tags", {})
    caption = row.get("caption_scrubbed", "")
    solo = tags.get("solo", 0.0)
    multi_max = max(
        tags.get("2girls", 0.0), tags.get("2boys", 0.0),
        tags.get("multiple_girls", 0.0), tags.get("multiple_boys", 0.0),
    )
    pair_extreme = tags.get("1girl", 0.0) >= PAIR_EXTREME_THR and tags.get("1boy", 0.0) >= PAIR_EXTREME_THR
    caption_multi = bool(CAPTION_MULTI_RE.search(caption))

    if solo >= SOLO_THR and multi_max < MULTI_THR and not pair_extreme and not caption_multi:
        return "solo_clean", []

    reasons = faithful_reasons(caption)
    if reasons:
        return "multi_faithful", reasons
    return "multi_poison", []


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cache(path):
    cache = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["uid"]] = rec
    return cache


def style_counts(rows):
    c = Counter()
    for row in rows:
        for m in STYLE_RE.finditer(row.get("caption_scrubbed", "")):
            c[m.group(1).lower()] += 1
    return c


def apply_solo_prefix(row):
    row = dict(row)
    cap = row.get("caption_scrubbed", "")
    if not cap.strip().lower().startswith("solo"):
        cap = SOLO_PREFIX + cap
    row["caption_scrubbed"] = cap
    return row


def main():
    print(f"[49_mixed_manifest] reading {MANIFEST_IN}", flush=True)
    rows = load_manifest(MANIFEST_IN)
    n_total = len(rows)
    print(f"[49_mixed_manifest] {n_total} rows", flush=True)

    print(f"[49_mixed_manifest] reading wd14 cache {CACHE_PATH}", flush=True)
    cache = load_cache(CACHE_PATH)
    missing = [row["uid"] for row in rows if row["uid"] not in cache]
    if missing:
        raise SystemExit(
            f"{len(missing)} uids missing from wd14 cache (e.g. {missing[:5]}) -- "
            f"this script is read-only against the cache and does not call the "
            f"tagger; run probe/48_clean_manifest.py first to populate it."
        )
    print(f"[49_mixed_manifest] wd14 cache covers all {n_total} rows, no tagger calls needed", flush=True)

    buckets = {"nsfw": [], "solo_clean": [], "multi_faithful": [], "multi_poison": []}
    faithful_reason_counts = Counter()
    faithful_rows_with_reasons = []

    for row in rows:
        wd14_rec = cache[row["uid"]]
        cat, reasons = classify_row(row, wd14_rec)
        buckets[cat].append(row)
        if cat == "multi_faithful":
            faithful_rows_with_reasons.append((row, reasons))
            for r in reasons:
                faithful_reason_counts[r] += 1

    n_nsfw = len(buckets["nsfw"])
    n_solo = len(buckets["solo_clean"])
    n_faithful = len(buckets["multi_faithful"])
    n_poison = len(buckets["multi_poison"])
    assert n_nsfw + n_solo + n_faithful + n_poison == n_total

    print("\n=== CLASSIFICATION ===")
    print(f"total={n_total}")
    print(f"  nsfw           = {n_nsfw}")
    print(f"  solo_clean     = {n_solo}")
    print(f"  multi_faithful = {n_faithful}")
    print(f"  multi_poison   = {n_poison}")
    print("\nfaithful reason breakdown (overlapping -- a row can hit several):")
    for r, c in faithful_reason_counts.most_common():
        print(f"  {r}: {c}")

    if n_solo == 0:
        raise SystemExit("solo_clean bucket is empty -- cannot balance a mix against zero solo rows")

    # --- oversample solo_clean, (down)sample multi_faithful to balance ----
    raw_K = math.ceil(n_faithful / n_solo)
    K = min(raw_K, K_CAP)
    n_solo_side = n_solo * K
    n_couple_target = min(n_faithful, n_solo_side)
    downsampled = n_couple_target < n_faithful

    random.seed(SEED)
    if downsampled:
        faithful_final_rows = random.sample(buckets["multi_faithful"], n_couple_target)
    else:
        faithful_final_rows = list(buckets["multi_faithful"])
    n_couple_side = len(faithful_final_rows)

    solo_final_rows = []
    for row in buckets["solo_clean"]:
        prefixed = apply_solo_prefix(row)
        solo_final_rows.extend([dict(prefixed) for _ in range(K)])

    mixed = solo_final_rows + faithful_final_rows
    random.seed(SEED)
    random.shuffle(mixed)

    os.makedirs(DATA, exist_ok=True)
    with open(MANIFEST_OUT, "w") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[49_mixed_manifest] wrote {MANIFEST_OUT} ({len(mixed)} rows)", flush=True)

    # --- style-word distribution (on the classification-level sets, i.e.
    # 371 unique solo_clean rows and the full 4991 unique multi_faithful
    # rows pre-downsample -- duplicates from oversampling/downsampling
    # would only rescale counts, not add information) -----------------
    style_solo = style_counts(buckets["solo_clean"])
    style_faithful = style_counts(buckets["multi_faithful"])

    # --- 10 + 10 caption samples, first-N-by-uid-order for reproducibility
    faithful_samples = [
        {"uid": row["uid"], "reasons": reasons, "caption": row["caption_scrubbed"]}
        for row, reasons in faithful_rows_with_reasons[:10]
    ]
    poison_samples = [
        {"uid": row["uid"], "caption": row["caption_scrubbed"]}
        for row in buckets["multi_poison"][:10]
    ]

    report = {
        "manifest_in": MANIFEST_IN,
        "manifest_out": MANIFEST_OUT,
        "wd14_cache": CACHE_PATH,
        "thresholds": {
            "solo_thr": SOLO_THR, "multi_thr": MULTI_THR,
            "pair_extreme_thr": PAIR_EXTREME_THR, "nsfw_thr": NSFW_THR,
        },
        "counts": {
            "total": n_total, "nsfw": n_nsfw, "solo_clean": n_solo,
            "multi_faithful": n_faithful, "multi_poison": n_poison,
        },
        "faithful_reason_counts_overlapping": dict(faithful_reason_counts.most_common()),
        "mix_params": {
            "raw_K_ceil_nfaithful_over_nsolo": raw_K,
            "K_cap": K_CAP,
            "K_used": K,
            "n_solo_clean_unique": n_solo,
            "n_solo_side_after_oversample": n_solo_side,
            "n_faithful_unique": n_faithful,
            "n_couple_side_final": n_couple_side,
            "faithful_downsampled": downsampled,
            "seed": SEED,
            "solo_prefix": SOLO_PREFIX,
        },
        "final": {
            "n_rows": len(mixed),
            "n_solo_side": len(solo_final_rows),
            "n_couple_side": n_couple_side,
            "solo_to_couple_ratio": (
                round(len(solo_final_rows) / n_couple_side, 3) if n_couple_side else None
            ),
        },
        "style_distribution": {
            f"solo_clean (n={n_solo} unique rows)": dict(style_solo.most_common()),
            f"multi_faithful (n={n_faithful} unique rows, pre-downsample)": dict(style_faithful.most_common()),
        },
        "samples": {
            "multi_faithful_10": faithful_samples,
            "multi_poison_10": poison_samples,
        },
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"[49_mixed_manifest] wrote {REPORT_OUT}", flush=True)

    # --- terminal summary ---
    print("\n=== MIX ===")
    print(f"raw_K=ceil({n_faithful}/{n_solo})={raw_K}  K_cap={K_CAP}  K_used={K}")
    print(f"solo side: {n_solo} unique x{K} = {len(solo_final_rows)} rows")
    print(f"couple side: {n_faithful} unique -> "
          f"{'downsampled to ' + str(n_couple_side) if downsampled else 'kept all ' + str(n_couple_side)}")
    print(f"final mixed manifest: {len(mixed)} rows, "
          f"solo:couple = {len(solo_final_rows)}:{n_couple_side} "
          f"(ratio {len(solo_final_rows)/n_couple_side:.3f}:1)")

    print(f"\nstyle distribution -- solo_clean (n={n_solo}):")
    for style, c in style_solo.most_common():
        print(f"  {style}: {c}")
    print(f"\nstyle distribution -- multi_faithful (n={n_faithful}, pre-downsample):")
    for style, c in style_faithful.most_common():
        print(f"  {style}: {c}")

    print("\n=== 10 multi_faithful samples ===")
    for s in faithful_samples:
        print(f"uid={s['uid']} reasons={s['reasons']}")
        print(f"  {s['caption']}")
    print("\n=== 10 multi_poison samples ===")
    for s in poison_samples:
        print(f"uid={s['uid']}")
        print(f"  {s['caption']}")


if __name__ == "__main__":
    main()
