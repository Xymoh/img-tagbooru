from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None

from backend.tag_index import TagFrequencyIndex, get_tag_index
from backend.content_policy import ADULTS_ONLY_RULE, strip_minor_terms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tags considered "actor/act" for the action-only copy feature
_ACTOR_ACT_TAGS: set[str] = {
    "1girl", "2girls", "3girls", "multiple_girls",
    "1boy", "2boys", "3boys", "multiple_boys",
    "fellatio", "blowjob", "oral", "deepthroat", "cunnilingus",
    "sex", "vaginal", "anal", "anal_sex", "paizuri", "handjob", "footjob",
    "doggystyle", "missionary", "cowgirl_position", "reverse_cowgirl",
    "sex_from_behind", "riding", "straddling",
    "gangbang", "group_sex", "rape", "forced", "double_penetration",
    "69", "masturbation", "fingering", "fisting",
    "squirting", "orgasm",
    "kneeling", "bent_over", "on_all_fours", "on_back", "legs_up",
    "spread_legs", "spread_pussy", "lying",
    "penis", "erection", "vagina", "pussy", "clitoris",
    "breasts", "large_breasts", "bare_breasts", "nipples",
    "ass", "bare_ass", "spread_ass",
    "open_mouth", "tongue_out", "saliva", "drooling", "gagging",
    "penis_in_mouth", "cum_in_mouth",
    "cum", "cumshot", "cum_on_face", "cum_on_body", "creampie", "cum_inside",
    "facial",
    "ahegao", "moaning", "panting", "blush", "tears",
    "submission", "dominance", "power_dynamic", "restraints", "bondage",
}

# ---------------------------------------------------------------------------
# Semantic deduplication map
#   canonical_tag → [synonyms that should be collapsed into canonical]
# ---------------------------------------------------------------------------

_SEMANTIC_DEDUP_MAP: dict[str, list[str]] = {
    "fellatio":         ["blowjob", "oral", "bj", "sucking_dick", "deepthroat"],
    "cum_in_mouth":     ["cum_on_lips", "cum_on_tongue", "cum_in_throat"],
    "facial":           ["cum_on_face", "cum_on_hair", "cum_on_forehead"],
    "cum_on_body":      ["cum_on_chest", "cum_on_stomach", "cum_on_hands",
                         "cum_on_feet", "cumshot_on_body"],
    "night":            ["darkness", "dim_lighting", "dark", "low_light", "dark_ambience"],
    "alley":            ["alleyway", "back_alley"],
    "moaning":          ["panting", "heavy_breathing"],
    "tongue_out":       ["tongue"],
    "erection":         ["erect_penis", "hard"],
    "street":           ["road", "sidewalk", "pavement", "cobblestone"],
    "urban":            ["city", "cityscape", "downtown"],
    "saliva":           ["drooling", "drool"],
    "kneeling":         ["on_knees", "kneel"],
    "nun":              ["nun_outfit", "religious_sister", "nun_headdress", "traditional_nun"],
}

# Build reverse index for fast lookup: synonym → canonical
_SYNONYM_TO_CANONICAL: dict[str, str] = {}
for _canon, _syns in _SEMANTIC_DEDUP_MAP.items():
    for _s in _syns:
        _SYNONYM_TO_CANONICAL[_s] = _canon


# ---------------------------------------------------------------------------
# Pose / framing / viewpoint conflict groups
#   Tags in the same group are mutually exclusive — keep highest post_count.
# ---------------------------------------------------------------------------

_CONFLICT_GROUPS: list[set[str]] = [
    # Body pose
    {"standing", "sitting", "kneeling", "lying", "squatting",
     "crouching", "on_all_fours", "bent_over"},
    # Viewpoint
    {"from_above", "from_below", "from_side", "from_behind", "from_front"},
    # Framing
    {"portrait", "upper_body", "cowboy_shot", "full_body", "close-up"},
    # Mouth
    {"open_mouth", "closed_mouth", "parted_lips"},
    # Time-of-day
    {"day", "night", "sunset", "twilight", "dawn", "dusk"},
    # Weather (soft — only one should win)
    {"rain", "snow", "fog", "clear_sky", "cloudy_sky"},
]

# Tag → group index lookup for O(1) conflict resolution
_TAG_TO_CONFLICT_GROUP: dict[str, int] = {
    tag: idx
    for idx, group in enumerate(_CONFLICT_GROUPS)
    for tag in group
}


# ---------------------------------------------------------------------------
# Concept / archetype expansion map
#   archetype_keyword → list of Danbooru tags that are visually plausible
#   even without being named in the description. Used by the relevance gate
#   so e.g. `nun` in the description permits `veil` in the output.
# ---------------------------------------------------------------------------

_CONCEPT_EXPANSIONS: dict[str, list[str]] = {
    # Archetypes / characters
    "nun":        ["veil", "cross", "habit", "rosary", "robe", "praying"],
    "priest":     ["cross", "robe", "rosary", "cassock", "1boy"],
    "witch":      ["witch_hat", "broom", "robe", "magic", "pointy_hat", "cauldron"],
    "knight":     ["armor", "sword", "shield", "cape", "helmet", "gauntlets"],
    "maid":       ["maid", "maid_headdress", "apron", "frills", "maid_outfit"],
    "catgirl":    ["cat_ears", "cat_tail", "animal_ears", "tail"],
    "neko":       ["cat_ears", "cat_tail", "animal_ears", "tail"],
    "foxgirl":    ["fox_ears", "fox_tail", "animal_ears", "tail"],
    "elf":        ["pointy_ears", "long_hair", "dark_elf", "elf_ears"],
    "dwarf":      ["beard", "short"],
    "goblin":     ["green_skin", "goblin", "small"],
    "orc":        ["orc", "muscular_male", "dark_skin", "tusks", "male_orc", "1boy"],
    "angel":      ["wings", "halo", "angel_wings", "feathered_wings"],
    "demon":      ["horns", "demon_tail", "demon_wings", "demon_girl"],
    "zombie":     ["torn_clothes", "blood", "pale_skin", "undead"],
    "vampire":    ["fangs", "pale_skin", "red_eyes", "blood"],
    "succubus":   ["succubus", "horns", "wings", "demon_tail", "demon_girl",
                   "demon_horns", "heart", "seductive_smile", "cleavage"],
    "incubus":    ["horns", "demon_tail", "demon_wings", "1boy", "muscular"],
    "mermaid":    ["fish_tail", "seashell", "underwater", "swimming"],
    "ninja":      ["ninja", "mask", "katana", "dark_clothes"],
    "samurai":    ["katana", "armor", "kimono", "topknot"],
    "cowboy":     ["cowboy_hat", "cowboy_shot", "revolver", "jeans", "boots"],
    "pirate":     ["pirate_hat", "eyepatch", "cutlass", "ship"],
    "doctor":     ["labcoat", "stethoscope", "glasses"],
    "nurse":      ["nurse", "nurse_cap", "stethoscope", "syringe"],
    "sailor":     ["sailor_collar", "sailor_hat", "serafuku"],
    "bride":      ["wedding_dress", "veil", "bouquet", "white_dress"],
    "schoolgirl": ["school_uniform", "serafuku", "skirt", "pleated_skirt"],
    # Subcultures / aesthetic
    "emo":        ["pale_skin", "black_hair", "eyeliner", "goth", "dark_clothes",
                   "studded_belt", "choker", "messy_hair", "ear_piercing"],
    "goth":       ["pale_skin", "black_hair", "eyeliner", "black_clothes",
                   "choker", "lace", "lipstick", "ear_piercing"],
    "punk":       ["studded_belt", "leather_jacket", "ear_piercing", "messy_hair",
                   "ripped_clothes", "spiked_hair"],
    "couple":     ["1girl", "1boy", "duo", "looking_at_another", "intimate"],
    "lovers":     ["1girl", "1boy", "duo", "looking_at_another", "intimate",
                   "blush", "holding_hands"],
    "romantic":   ["looking_at_another", "intimate", "blush", "warm_lighting",
                   "soft_lighting"],
    # Settings — interior
    "kitchen":    ["kitchen", "counter", "apron", "cooking", "stove"],
    "baking":     ["baking", "cooking", "apron", "kitchen", "food", "flour"],
    "bedroom":    ["bedroom", "bed", "indoors", "window", "lamp", "pillow"],
    "bathroom":   ["bathroom", "bathtub", "shower", "indoors", "tile", "wet"],
    "classroom":  ["classroom", "school_uniform", "indoors", "desk", "chalkboard"],
    "office":     ["office", "indoors", "desk", "computer", "suit"],
    "library":    ["library", "bookshelf", "book", "indoors"],
    "gym":        ["gym", "indoors", "sportswear", "sweat", "athletic"],
    "cathedral":  ["church", "cathedral", "stained_glass", "cross", "indoors"],
    # Settings — outdoor / nature
    "park":       ["park", "bench", "tree", "outdoors", "grass"],
    "beach":      ["beach", "ocean", "sand", "swimsuit", "sky", "horizon",
                   "wave", "cloud", "sunlight", "outdoors"],
    "lake":       ["lake", "water", "outdoors", "ripples", "reflection",
                   "tree", "shore", "sky", "cloud", "calm_water"],
    "river":      ["river", "water", "outdoors", "rocks", "current", "tree",
                   "sky", "reflection"],
    "ocean":      ["ocean", "water", "wave", "sky", "horizon", "outdoors", "cloud"],
    "sea":        ["ocean", "water", "wave", "sky", "horizon", "outdoors"],
    "mountain":   ["mountain", "outdoors", "sky", "cloud", "snow", "rocks"],
    "field":      ["field", "grass", "outdoors", "sky", "wind", "flower"],
    "meadow":     ["field", "grass", "flower", "outdoors", "sky", "sunlight"],
    "forest":     ["forest", "tree", "outdoors", "grass", "leaves", "foliage",
                   "dappled_light", "sunlight", "shadow", "fog", "mist",
                   "forest_background", "nature", "path", "moss"],
    "garden":     ["garden", "flower", "tree", "outdoors", "grass", "petals"],
    "alley":      ["alley", "brick_wall", "street", "outdoors", "shadow"],
    "city":       ["cityscape", "city", "building", "street", "outdoors", "sky"],
    "street":     ["street", "road", "outdoors", "building", "shadow"],
    "rooftop":    ["rooftop", "outdoors", "sky", "city", "sunset"],
    "train":      ["train_interior", "train_station", "standing", "crowd"],
    "storm":      ["rain", "dark_clouds", "lightning", "wind", "sky"],
    "rain":       ["rain", "wet", "wet_hair", "umbrella", "sky", "puddle"],
    "snow":       ["snow", "winter", "snowing", "scarf", "breath", "sky"],
    "sunset":     ["sunset", "orange_sky", "cloud", "outdoors", "warm_lighting"],
    "night":      ["night", "moonlight", "stars", "dark", "shadow"],
    "winter":     ["snow", "winter", "scarf", "coat", "breath"],
    "summer":     ["summer", "sky", "cloud", "sunlight", "outdoors"],
    "spring":     ["cherry_blossoms", "petals", "spring", "flower", "outdoors"],
    "autumn":     ["autumn_leaves", "leaves", "outdoors", "warm_lighting"],
    # Activities / wear hints
    "swimming":   ["swimsuit", "water", "wet", "wet_hair", "swimming", "splashing"],
    "swimsuit":   ["swimsuit", "bare_shoulders", "cleavage", "midriff", "navel",
                   "thighs", "wet"],
    "bikini":     ["bikini", "swimsuit", "bare_shoulders", "cleavage", "midriff",
                   "navel", "thighs", "wet", "side-tie_bikini"],
    "festival":   ["yukata", "festival", "fireworks", "lantern", "summer_festival"],
    "wedding":    ["wedding_dress", "veil", "bouquet", "white_dress", "indoors"],
}


# ---------------------------------------------------------------------------
# Sex-act / NSFW action expansion map
#   act_keyword → list of anatomy / position / reaction tags that naturally
#   accompany the act. The relevance gate uses this so e.g. a description
#   saying "blowjob" permits `penis`, `fellatio`, `saliva`, `kneeling`, etc.
#   that don't literally appear in the user's text.
# ---------------------------------------------------------------------------

_ACT_EXPANSIONS: dict[str, list[str]] = {
    # Oral
    "blowjob":    ["fellatio", "penis", "open_mouth", "saliva", "oral",
                   "kneeling", "tongue_out", "deepthroat", "drooling",
                   "cum_in_mouth", "erection"],
    "fellatio":   ["fellatio", "penis", "open_mouth", "saliva", "oral",
                   "kneeling", "tongue_out", "deepthroat"],
    "sucking":    ["fellatio", "penis", "open_mouth", "saliva", "oral"],
    "oral":       ["fellatio", "penis", "open_mouth", "saliva"],
    "cunnilingus": ["cunnilingus", "pussy", "oral", "spread_legs", "tongue"],
    "deepthroat": ["deepthroat", "fellatio", "penis", "gagging", "saliva"],
    # Penetration / positions
    "sex":        ["sex", "penetration", "vaginal", "penis", "pussy",
                   "nude", "breasts", "moaning", "blush"],
    "fuck":       ["sex", "penetration", "vaginal", "penis", "pussy",
                   "nude", "breasts", "moaning", "blush"],
    "fucking":    ["sex", "penetration", "vaginal", "penis", "pussy",
                   "nude", "breasts", "moaning", "blush"],
    "penetration": ["sex", "penetration", "vaginal", "penis", "pussy"],
    "missionary": ["missionary", "sex", "penetration", "lying", "on_back"],
    "doggystyle": ["doggystyle", "sex_from_behind", "bent_over", "ass",
                   "penis", "vaginal"],
    "doggy":      ["doggystyle", "sex_from_behind", "bent_over", "ass"],
    "cowgirl":    ["cowgirl_position", "girl_on_top", "straddling", "sex",
                   "vaginal", "penetration"],
    "riding":     ["cowgirl_position", "girl_on_top", "straddling", "sex",
                   "penetration", "penis"],
    "anal":       ["anal", "ass", "penetration", "doggystyle"],
    "gangbang":   ["gangbang", "group_sex", "multiple_penises",
                   "double_penetration", "multiple_boys", "sex"],
    "group_sex":  ["group_sex", "gangbang", "multiple_penises", "multiple_boys"],
    "threesome":  ["threesome", "group_sex", "multiple_boys", "multiple_girls"],
    "spitroast":  ["spitroast", "group_sex", "multiple_penises", "multiple_boys",
                   "fellatio", "vaginal"],
    # Non-consent
    "rape":       ["rape", "forced", "restrained", "crying", "tears", "sex"],
    "forced":     ["rape", "forced", "crying", "tears"],
    "forcing":    ["rape", "forced", "crying", "tears", "restrained"],
    "bondage":    ["bondage", "bdsm", "restrained", "rope", "tied"],
    # Touch / soft
    "kissing":    ["kiss", "french_kiss", "tongue", "eye_contact",
                   "hand_on_another's_cheek"],
    "groping":    ["groping", "breast_grab", "fondling", "grabbing"],
    "seducing":   ["seductive_smile", "cleavage", "arched_back",
                   "looking_at_viewer", "bedroom_eyes", "naughty_face"],
    "seduce":     ["seductive_smile", "cleavage", "looking_at_viewer",
                   "bedroom_eyes"],
    "handjob":    ["handjob", "penis", "erection"],
    "footjob":    ["footjob", "penis", "feet"],
    "paizuri":    ["paizuri", "breasts", "large_breasts", "penis"],
    "titfuck":    ["paizuri", "breasts", "large_breasts", "penis"],
    "titjob":     ["paizuri", "breasts", "large_breasts", "penis"],
    "masturbate": ["masturbation", "fingering", "blush", "solo"],
    "masturbating": ["masturbation", "fingering", "blush", "solo"],
    "masturbation": ["masturbation", "fingering", "blush"],
    "squirt":     ["squirting", "pussy", "female_ejaculation", "orgasm"],
    "squirting":  ["squirting", "pussy", "female_ejaculation", "orgasm"],
    "orgasm":     ["orgasm", "moaning", "ahegao", "trembling"],
    # Fluids / finish (only if user mentions them)
    "cum":        ["cum", "cumshot", "facial", "cum_in_mouth", "cum_on_body"],
    "cumshot":    ["cum", "cumshot", "facial"],
    "creampie":   ["creampie", "cum_in_pussy", "cum_inside"],
    "cumming":    ["cum", "cumshot", "orgasm", "ahegao"],
    # Anatomy keywords (if user explicitly mentions)
    "cock":       ["penis", "erection"],
    "dick":       ["penis", "erection"],
    "pussy":      ["pussy", "spread_pussy"],
    "boobs":      ["breasts", "large_breasts", "nipples"],
    "tits":       ["breasts", "large_breasts", "nipples"],
    "breast":     ["breasts", "nipples"],
    "breasts":    ["breasts", "nipples"],
    "naked":      ["nude", "completely_nude", "breasts", "pussy", "nipples"],
    "nude":       ["nude", "completely_nude", "breasts", "nipples"],
    "busty":      ["large_breasts", "huge_breasts", "breasts", "cleavage"],
    "curvy":      ["large_breasts", "wide_hips", "thick_thighs"],
    "thick":      ["thick_thighs", "wide_hips", "curvy"],
    "muscular":   ["muscular", "abs", "muscular_male"],
    "petite":     ["small_breasts", "flat_chest", "short"],
}

# Tags that are universally plausible in almost any scene — never treated
# as "foreign" even if the description doesn't mention them.
_UNIVERSAL_TAGS: set[str] = {
    "solo", "multiple_girls", "multiple_boys",
    "long_hair", "short_hair", "medium_hair", "very_long_hair",
    "ponytail", "twintails", "braid", "bangs", "side_ponytail",
    "messy_hair", "wavy_hair", "straight_hair", "floating_hair", "hair_blowing",
    "blonde_hair", "brown_hair", "black_hair", "white_hair", "blue_hair",
    "red_hair", "pink_hair", "silver_hair", "grey_hair", "green_hair",
    "purple_hair", "orange_hair", "two-tone_hair",
    "blue_eyes", "brown_eyes", "green_eyes", "red_eyes", "yellow_eyes",
    "purple_eyes", "pink_eyes", "grey_eyes", "black_eyes", "heterochromia",
    "looking_at_viewer", "looking_away", "looking_back", "looking_to_the_side",
    "smile", "blush", "closed_eyes", "half-closed_eyes", "frown",
    "open_mouth", "closed_mouth", "parted_lips", "smirk", "grin",
    "highres", "absurdres", "masterpiece", "best_quality", "detailed",
    "ultra_detailed", "depth_of_field", "soft_focus", "realistic",
    "1girl", "1boy", "2girls", "2boys",
    "solo_focus", "duo",
    "indoors", "outdoors", "day", "night",
    "sky", "cloud", "clear_sky", "cloudy_sky", "blue_sky",
    "simple_background", "blurry_background",
    # Lighting / atmosphere — plausible in any scene
    "sunlight", "moonlight", "shadow", "soft_lighting", "dim_lighting",
    "backlighting", "lens_flare", "warm_lighting", "cool_lighting",
    "dappled_light", "light_rays", "glowing", "sparkle", "ambient_light",
    "sunset", "sunrise", "twilight", "dawn", "dusk", "golden_hour",
    # Nature / environment details — plausible in any outdoor scene
    "tree", "grass", "leaf", "leaves", "flower", "petals", "cherry_blossoms",
    "wind", "breeze", "rain", "snow", "fog", "mist", "water", "reflection",
    "ripples", "wave", "horizon", "shore", "sand",
    "nature", "foliage", "branch", "path", "rocks", "stone",
    # Skin / body details — plausible on any character
    "fair_skin", "pale_skin", "dark_skin", "tan", "skin",
    "thighs", "legs", "arms", "shoulders", "bare_shoulders", "collarbone",
    "navel", "midriff", "cleavage", "small_breasts", "medium_breasts",
    "wet", "wet_hair", "wet_clothes", "sweat",
    # Common pose / expression additions
    "standing", "sitting", "looking_up", "looking_down",
    "leaning_forward", "leaning_back",
    # Common clothing / accessory additions
    "choker", "necklace", "earrings", "ring", "bracelet", "ribbon",
    "jewelry", "piercing", "ear_piercing",
    # Style / quality additions
    "motion_blur", "bokeh", "film_grain", "chromatic_aberration",
    "scenery", "atmospheric", "cinematic_lighting",
}


# ---------------------------------------------------------------------------
# Mature "wildcard" pool — when a Mature-mode description is non-explicit,
# the post-processor injects 1-2 of these to add a tasteful suggestive flair
# that always makes sense regardless of scene.
# ---------------------------------------------------------------------------

_MATURE_WILDCARDS_GENERIC: list[str] = [
    "cleavage", "bare_shoulders", "thighs", "wet", "lipstick",
    "seductive_smile", "looking_at_viewer", "naughty_face",
    "bedroom_eyes", "parted_lips", "blush", "soft_focus",
]

# Setting-aware wildcard pools — used when the description hints at a vibe.
# All entries are tasteful (never explicit) and pair naturally with the cue.
_MATURE_WILDCARDS_BY_CUE: dict[str, list[str]] = {
    "beach":   ["bikini", "side-tie_bikini", "wet", "cleavage", "navel"],
    "lake":    ["wet", "wet_hair", "swimsuit", "bare_shoulders", "thighs"],
    "river":   ["wet", "wet_hair", "swimsuit", "bare_shoulders"],
    "ocean":   ["bikini", "wet", "cleavage", "bare_shoulders"],
    "pool":    ["bikini", "wet", "wet_hair", "cleavage"],
    "bath":    ["wet", "wet_hair", "blush", "steam", "bare_shoulders"],
    "shower":  ["wet", "wet_hair", "blush", "steam", "bare_shoulders"],
    "bedroom": ["bare_shoulders", "blush", "intimate", "dim_lighting", "messy_hair"],
    "night":   ["dim_lighting", "soft_focus", "moonlight", "bedroom_eyes"],
    "rain":    ["wet", "wet_hair", "wet_clothes", "see-through"],
    "forest":  ["sunlight", "thighs", "soft_focus", "messy_hair"],
}


# ---------------------------------------------------------------------------
# Few-shot examples per mode
# ---------------------------------------------------------------------------

_FEWSHOT_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "safe": [
        (
            "a girl, emo, black hair",
            "1girl, solo, long_hair, black_hair, black_eyes, looking_at_viewer, full_body, standing, open_mouth, twintails, hair_over_one_eye, choker, black_choker, black_shirt, t-shirt, long_sleeves, miniskirt, black_skirt, pleated_skirt, boots, black_boots, knee_boots, arm_warmers, black_arm_warmers, belt, nail_polish, hair_ornament, x_hair_ornament, simple_background, red_background",
        ),
        (
            "warrior girl in a forest with a sword",
            "1girl, solo, long_hair, silver_hair, ponytail, red_eyes, looking_at_viewer, serious, full_body, standing, holding, holding_sword, sword, armor, breastplate, gauntlets, cape, red_cape, boots, brown_boots, belt, scabbard, forest, outdoors, tree, grass, dappled_sunlight, wind, hair_blowing, depth_of_field, highres",
        ),
        (
            "two girls at a cafe, school uniforms",
            "2girls, multiple_girls, school_uniform, serafuku, sailor_collar, pleated_skirt, blue_skirt, white_shirt, short_sleeves, sitting, table, cup, coffee, indoors, cafe, window, smile, looking_at_another, brown_hair, ponytail, black_hair, short_hair, bag, school_bag, chair, menu, sunlight",
        ),
    ],
    "creative": [
        (
            "a girl, emo, black hair",
            "1girl, solo, long_hair, black_hair, black_eyes, looking_at_viewer, full_body, standing, open_mouth, twintails, sidelocks, hair_over_one_eye, choker, black_choker, black_shirt, t-shirt, skeleton_print, long_sleeves, miniskirt, plaid_skirt, red_skirt, pleated_skirt, boots, black_boots, knee_boots, arm_warmers, black_arm_warmers, red_arm_warmers, mismatched_arm_warmers, belt, studded_belt, nail_polish, hair_ornament, x_hair_ornament, ear_piercing, simple_background, red_background, v, hand_up, cellphone, holding_phone, selfie, highres",
        ),
        (
            "a girl, blonde hair, at the beach",
            "1girl, solo, blonde_hair, long_hair, ponytail, blue_eyes, looking_at_viewer, smile, teeth, full_body, standing, bikini, white_bikini, side-tie_bikini_bottom, bare_shoulders, navel, collarbone, thighs, wet, wet_hair, barefoot, sand, beach, ocean, wave, sky, cloud, blue_sky, horizon, sunlight, lens_flare, sun, shadow, towel, beach_umbrella, depth_of_field, wind, hair_blowing, highres, absurdres",
        ),
        (
            "boy and girl holding hands in a park, autumn",
            "1boy, 1girl, hetero, holding_hands, outdoors, park, autumn, autumn_leaves, tree, bench, path, grass, standing, smile, looking_at_another, blush, brown_hair, short_hair, long_hair, black_hair, scarf, red_scarf, coat, brown_coat, skirt, pleated_skirt, boots, school_bag, wind, leaves, warm_lighting, depth_of_field, bokeh, from_behind, full_body, highres",
        ),
    ],
    "mature": [
        (
            "a girl, blonde hair, interracial, blowjob",
            "1boy, 1girl, hetero, blonde_hair, sidelocks, grey_eyes, looking_at_another, fellatio, oral, penis, large_penis, dark-skinned_male, dark_skin, very_dark_skin, interracial, on_stomach, lying, feet_up, crossed_ankles, indoors, couch, on_couch, bracelet, earrings, jewelry, necklace, ring, eyeshadow, makeup, red_nails, shoes, red_shoes, strappy_heels, hand_on_another's_head, depth_of_field, blurry_background, highres",
        ),
        (
            "a succubus riding a man, dark cathedral, stained glass",
            "1boy, 1girl, hetero, succubus, demon_girl, horns, demon_horns, demon_tail, demon_wings, purple_hair, long_hair, red_eyes, looking_at_viewer, smile, fangs, nude, breasts, large_breasts, nipples, navel, straddling, girl_on_top, cowgirl_position, sex, vaginal, penis, spread_legs, sweat, blush, indoors, cathedral, stained_glass, candlelight, dim_lighting, cross, pew, stone_floor, depth_of_field, highres, absurdres",
        ),
        (
            "a girl, emo, black hair, bedroom, lingerie",
            "1girl, solo, long_hair, black_hair, red_eyes, looking_at_viewer, parted_lips, blush, lying, on_back, on_bed, bed, pillow, bedroom, indoors, night, dim_lighting, lamp, lingerie, black_lingerie, lace, bra, panties, thighhighs, black_thighhighs, garter_belt, choker, black_choker, nail_polish, black_nail_polish, collarbone, cleavage, navel, thighs, bare_shoulders, messy_hair, arm_up, hand_in_hair, depth_of_field, highres, absurdres",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Few-shot examples for TAG ENRICHMENT
#   Input: a list of seed tags (already-validated Danbooru tags)
#   Output: additional complementary tags that fit the implied scene.
# ---------------------------------------------------------------------------

_ENRICH_FEWSHOT_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "safe": [
        (
            "1girl, beach, volleyball",
            "solo, blonde_hair, ponytail, blue_eyes, smile, teeth, athletic, tan, bikini, white_bikini, sports_bikini, barefoot, jumping, arm_up, reaching, ball, sand, ocean, sky, blue_sky, cloud, sunlight, sweat, motion_blur, full_body, outdoors, day, highres",
        ),
        (
            "1girl, witch_hat, forest",
            "solo, long_hair, purple_hair, green_eyes, smile, witch, robe, black_robe, cape, staff, holding, holding_staff, broom, pointy_hat, boots, brown_boots, belt, pouch, forest, tree, outdoors, night, moonlight, glowing, fog, path, leaves, lantern, full_body, standing, highres",
        ),
        (
            "1girl, maid, kitchen",
            "solo, brown_hair, long_hair, braid, blue_eyes, smile, maid, maid_headdress, apron, white_apron, black_dress, long_sleeves, frills, ribbon, thighhighs, white_thighhighs, shoes, holding, ladle, cooking, indoors, kitchen, counter, stove, pot, steam, window, sunlight, tile_floor, full_body, standing, highres",
        ),
    ],
    "creative": [
        (
            "1girl, beach, volleyball",
            "solo, blonde_hair, ponytail, hair_tie, blue_eyes, looking_at_viewer, smile, teeth, open_mouth, athletic, tan, tanlines, bikini, white_bikini, sports_bikini, side-tie_bikini_bottom, barefoot, jumping, arm_up, reaching, outstretched_arm, ball, volleyball, sand, ocean, wave, splash, sky, blue_sky, cloud, horizon, sunlight, lens_flare, sweat, wet, motion_blur, depth_of_field, full_body, outdoors, day, summer, highres, absurdres",
        ),
        (
            "1girl, 1boy, school_uniform, classroom",
            "hetero, brown_hair, long_hair, ponytail, ribbon, hair_ribbon, brown_eyes, looking_at_another, blush, smile, black_hair, short_hair, school_uniform, serafuku, sailor_collar, pleated_skirt, blue_skirt, white_shirt, short_sleeves, kneehighs, white_kneehighs, loafers, necktie, pants, black_pants, sitting, desk, chair, leaning_forward, hand_on_desk, indoors, classroom, chalkboard, window, curtains, sunlight, afternoon, depth_of_field, highres, absurdres",
        ),
    ],
    "mature": [
        (
            "1girl, 1boy, bedroom",
            "hetero, long_hair, brown_hair, brown_eyes, blush, open_mouth, moaning, nude, breasts, medium_breasts, nipples, navel, sweat, on_back, lying, spread_legs, missionary, sex, vaginal, penis, penetration, pillow, bed_sheet, indoors, bedroom, night, lamp, dim_lighting, messy_hair, tears, saliva, panting, hand_on_another's_chest, depth_of_field, highres, absurdres",
        ),
        (
            "1girl, 1boy, fellatio, nun",
            "hetero, nun, veil, white_veil, cross, cross_necklace, habit, black_habit, robe, kneeling, fellatio, oral, penis, large_penis, open_mouth, tongue, tongue_out, saliva, saliva_trail, blush, tears, looking_up, eye_contact, hand_on_another's_thigh, indoors, cathedral, stained_glass, candlelight, dim_lighting, stone_floor, pew, night, depth_of_field, highres, absurdres",
        ),
        (
            "1girl, maid, library",
            "solo, long_hair, black_hair, red_eyes, looking_back, blush, parted_lips, maid, maid_headdress, apron, white_apron, black_dress, short_sleeves, frills, thighhighs, black_thighhighs, garter_belt, garter_straps, shoes, bent_over, skirt_lift, ass, panties, white_panties, cleavage, large_breasts, indoors, library, bookshelf, book, desk, lamp, dim_lighting, wooden_floor, depth_of_field, highres, absurdres",
        ),
    ],
}


# ---------------------------------------------------------------------------
# DescriptionTagResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptionTagResult:
    tags: list[str]
    raw_response: str
    model: str
    actor_tags: list[str] = None
    scene_tags: list[str] = None

    def __post_init__(self) -> None:
        if self.actor_tags is None:
            actor = [t for t in self.tags if t in _ACTOR_ACT_TAGS]
            scene = [t for t in self.tags if t not in _ACTOR_ACT_TAGS]
            object.__setattr__(self, "actor_tags", actor)
            object.__setattr__(self, "scene_tags", scene)


# ---------------------------------------------------------------------------
# Natural-language prompt few-shot examples
#   Danbooru tag list -> flowing prose prompt.
#   These teach FORMAT (one paragraph, no tags, no underscores); the per-mode
#   rules in _build_prose_system_prompt decide how much content latitude the
#   model has.
# ---------------------------------------------------------------------------

_PROSE_FEWSHOT: dict[str, list[tuple[str, str]]] = {
    "safe": [
        (
            "1girl, solo, long_hair, silver_hair, ponytail, red_eyes, serious, full_body, standing, "
            "holding_sword, sword, armor, breastplate, gauntlets, cape, red_cape, brown_boots, belt, "
            "scabbard, forest, outdoors, tree, grass, dappled_sunlight, wind, hair_blowing, depth_of_field",
            "A young woman stands alone in a sunlit forest clearing, gripping a longsword point-down in "
            "the grass. Her silver hair is tied back in a high ponytail that lifts in the wind, and her "
            "red eyes are set in a serious, level stare. She wears a fitted steel breastplate over "
            "leather, articulated gauntlets, a heavy belt with an empty scabbard at her hip, and a deep "
            "red cape that falls behind her to worn brown boots. Dappled afternoon light filters through "
            "the canopy and scatters across the grass. Full-body shot at eye level, shallow depth of "
            "field, the trees softening into bokeh behind her.",
        ),
        (
            "2girls, multiple_girls, school_uniform, serafuku, pleated_skirt, blue_skirt, white_shirt, "
            "sitting, table, cup, coffee, indoors, cafe, window, smile, looking_at_another, brown_hair, "
            "ponytail, black_hair, short_hair, sunlight",
            "Two schoolgirls sit across from each other at a small cafe table by a tall window, talking "
            "over coffee. One has brown hair pulled into a ponytail, the other a short black bob; both "
            "wear navy sailor uniforms with white blouses and pleated blue skirts. They are turned "
            "toward one another mid-conversation, one smiling as she wraps both hands around her cup. "
            "Warm late-afternoon sunlight comes through the window and washes across the tabletop, "
            "catching the steam. Medium shot from across the table, soft natural light, quiet and "
            "unhurried mood.",
        ),
    ],
    "creative": [
        (
            "1girl, solo, long_hair, black_hair, twintails, hair_over_one_eye, pale_skin, eyeliner, "
            "choker, black_choker, black_shirt, long_sleeves, black_skirt, pleated_skirt, black_boots, "
            "knee_boots, arm_warmers, studded_belt, nail_polish, red_background, simple_background, "
            "looking_at_viewer, standing, depth_of_field",
            "A pale young woman with long black hair in uneven twintails stands facing the camera, one "
            "eye hidden behind a heavy sweep of fringe. Sharp black eyeliner rims her eyes and her "
            "expression is flat and unimpressed. She wears a fitted black long-sleeved shirt, a pleated "
            "black miniskirt over a studded belt slung low on her hips, black fingerless arm warmers, "
            "and scuffed black knee-high boots; a plain black choker sits at her throat and her nails "
            "are painted chipped black. She is lit from the front against a flat crimson backdrop that "
            "falls away into soft shadow. Waist-up portrait, slight depth of field, high-contrast "
            "alternative fashion editorial look.",
        ),
        (
            "1girl, solo, blonde_hair, ponytail, blue_eyes, smile, athletic, tan, white_bikini, barefoot, "
            "jumping, arm_up, reaching, volleyball, sand, ocean, wave, blue_sky, cloud, sunlight, "
            "lens_flare, sweat, motion_blur, full_body, outdoors, day, summer",
            "A tanned, athletic blonde woman leaps off the sand mid-spike, one arm stretched high toward "
            "a volleyball at the top of its arc. Her ponytail whips behind her and she is grinning, "
            "teeth showing, sweat catching the light along her shoulders. She wears a simple white "
            "sports bikini and is barefoot, sand still falling from her heels. Behind her the ocean "
            "breaks in white surf under a bright blue summer sky scattered with cloud. Full-body action "
            "shot from slightly below, hard midday sunlight with lens flare, a touch of motion blur in "
            "her trailing arm.",
        ),
    ],
    "mature": [
        (
            "1girl, solo, long_hair, black_hair, red_eyes, looking_at_viewer, blush, parted_lips, "
            "bedroom_eyes, cleavage, bare_shoulders, thighs, black_lingerie, lace, garter_belt, "
            "thighhighs, lying, on_back, bed, indoors, bedroom, dim_lighting, candlelight, "
            "depth_of_field, cowboy_shot",
            "A dark-haired woman lies back across rumpled bedsheets, propped on one elbow and looking "
            "directly at the viewer through half-lidded red eyes, lips parted and a flush across her "
            "cheeks. She wears black lace lingerie with a garter belt clipped to sheer thigh-highs, "
            "shoulders bare, one knee drawn up. The bedroom around her is dim, lit low and warm by "
            "candles just out of frame that pick out the lace and the curve of her waist. Cowboy shot "
            "from slightly above, shallow depth of field, intimate low-key lighting and a heavy, "
            "unhurried mood.",
        ),
    ],
}


# ---------------------------------------------------------------------------
# NaturalPromptResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalPromptResult:
    """A natural-language prompt built from a Danbooru tag set.

    ``prompt`` is the flowing prose intended for natural-language image models
    (Krea, Flux and similar T5-encoder models). ``tags`` is the intermediate
    tag set stage 1 produced, kept so the UI can show what the prose was
    written from and so the user can still copy the tags.
    """

    prompt: str
    tags: list[str]
    raw_response: str
    model: str
    word_count: int = 0

    def __post_init__(self) -> None:
        if not self.word_count:
            object.__setattr__(self, "word_count", len(self.prompt.split()))


# ---------------------------------------------------------------------------
# DescriptionTagger
# ---------------------------------------------------------------------------


class DescriptionTagger:
    """Generate Danbooru tags from text descriptions using local LLM (Ollama).

    Uses a TagFrequencyIndex backed by danbooru_tags_post_count.csv to:
      - Build vocabulary-grounded system prompts (LLM prefers common tags)
      - Post-process LLM output (strip rare tags, deduplicate synonyms)
    """

    DEFAULT_MODEL = "richardyoung/qwen3-14b-abliterated:latest"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_CREATIVITY = "creative"
    MAX_TAGS = 60  # Allow up to 60 tags for Danbooru-density output
    _NSFW_MODES = {"mature"}
    _VALID_CREATIVITIES = {"safe", "creative", "mature"}
    
    # Model-specific generation hints.
    # For qwen3, "/no_think" in the user message tells the model to skip
    # its <think>...</think> reasoning block, freeing the entire num_predict
    # budget for actual tag output. Huge reliability improvement for tag
    # generation where reasoning wastes tokens.
    _MODEL_PREFILLS = {
        "qwen3_no_think": "/no_think",
    }

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        post_count_threshold: int = 500,
    ) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.tag_index = get_tag_index(min_threshold=post_count_threshold)
        self._system_prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Post-count threshold accessor
    # ------------------------------------------------------------------

    @property
    def post_count_threshold(self) -> int:
        return self.tag_index.min_threshold

    def set_post_count_threshold(self, value: int) -> None:
        self.tag_index.set_min_threshold(value)
        self._system_prompt_cache.clear()

    # ------------------------------------------------------------------
    # System prompt builders  (vocabulary-grounded)
    # ------------------------------------------------------------------

    _VOCAB_CATEGORIES_BY_MODE: dict[str, list[str]] = {
        "safe":     ["character", "clothing", "setting", "action", "style"],
        "creative": ["character", "clothing", "setting", "action", "style", "supernatural"],
        "mature":   ["character", "clothing", "setting", "action", "style", "supernatural", "explicit"],
    }

    _VOCAB_SIZE_BY_MODE: dict[str, int] = {
        "safe": 8,
        "creative": 10,
        "mature": 10,
    }

    # Tags that are franchise/game proper-noun collisions — they match generic
    # keywords by accident. Most disambiguator tags (``tag_(context)``) are
    # now filtered automatically by _parse_tags and the vocabulary builder,
    # so this list only needs to hold non-parenthesised franchise collisions
    # and a few edge cases the relevance gate can't catch.
    _BLOCKED_VOCAB_TAGS: set[str] = {
        "hollow_knight", "knight_(hollow_knight)", "flower_knight_girl",
        "master_detective_archives:_rain_code",
        "shachiku_succubus_no_hanashi", "high_priest_(ragnarok_online)",
        "priest_(ragnarok_online)", "king", "queen", "chen",
        "standing_sex", "standing_split", "meta_knight", "standing_cunnilingus",
        "lord_knight_(ragnarok_online)", "little_witch_academia", "sue_storm",
        "storm_(x-men)", "flying_kick", "flying_sweatdrops", "priest_(dq3)",
        "male_priest_(dungeon_and_fighter)", "disgaea",
        "knight_(chess)", "standing_missionary", "standing_on_liquid",
        "ranni_the_witch", "sanoba_witch", "flying_fish",
        "cleaning_&_clearing_(blue_archive)", "kissing_penis",
        "kissing_cheek", "kissing_forehead", "kissing_hand",
        "rain_world", "rain_code",
        "flying_teardrops", "flying_paper",
    }

    # Words to skip when extracting keywords from user descriptions
    _STOP_WORDS: set[str] = {
        "a", "an", "the", "is", "at", "with", "and", "being", "by", "in", "on",
        "of", "to", "for", "it", "as", "or", "its", "from", "that", "this",
        "was", "are", "were", "been", "has", "had", "have", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall",
        "not", "no", "but", "if", "then", "else", "when", "where", "who",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "so", "than", "too", "very",
        "just", "about", "above", "after", "again", "against", "below",
        "between", "during", "into", "through", "under", "up", "down", "out",
        "off", "over", "while", "her", "his", "she", "he", "they", "them",
        "me", "my", "we", "our", "you", "your", "i", "him", "us", "their",
        "giving", "give", "gets", "get", "got", "make", "made", "making",
        "like", "look", "looking", "see", "seen", "want", "using", "use",
        "take", "taken", "taking", "come", "came", "go", "going", "went",
        "there", "here", "now", "then", "also", "still", "even", "much",
        "many", "any", "one", "two", "three", "well", "back", "way",
    }

    @staticmethod
    def _extract_keywords(description: str) -> list[str]:
        """Extract meaningful lowercase keywords from a user description."""
        import re
        # Lowercase and split on non-alphanumeric
        words = re.findall(r"[a-z0-9]+", description.lower())
        # Filter: at least 3 chars, no pure digits, no stop words
        keywords: list[str] = []
        seen: set[str] = set()
        for w in words:
            if len(w) < 3:
                continue
            if w.isdigit():
                continue
            if w in DescriptionTagger._STOP_WORDS:
                continue
            if w in seen:
                continue
            seen.add(w)
            keywords.append(w)
        return keywords

    def _build_vocabulary_section(
        self, creativity: str, description: str, exclude: set[str] | None = None,
    ) -> str:
        """Build the AVAILABLE VOCABULARY block, including prompt-specific tags."""
        categories = self._VOCAB_CATEGORIES_BY_MODE.get(
            creativity, self._VOCAB_CATEGORIES_BY_MODE["creative"]
        )
        n = self._VOCAB_SIZE_BY_MODE.get(creativity, 12)
        cat_tags = self.tag_index.top_by_category(
            n=n, categories=categories,
            exclude=(exclude or set()) | self._BLOCKED_VOCAB_TAGS,
        )

        label_map = {
            "character": "Character / Participants",
            "clothing":  "Clothing / Accessories",
            "setting":   "Setting / Environment",
            "action":    "Action / Pose / Expression",
            "style":     "Style / Quality",
            "supernatural": "Supernatural / Horror / Monster",
            "explicit":  "Explicit / Adult",
        }

        lines: list[str] = []
        for cat in categories:
            tags = cat_tags.get(cat, [])
            if tags:
                label = label_map.get(cat, cat.title())
                lines.append(f"- {label}: {', '.join(tags[:n])}")

        # --- Prompt-specific: inject tags matching user's keywords ---
        keywords = self._extract_keywords(description)
        # Allow disambiguator tags whose parenthetical matches any keyword in
        # the description (so e.g. "hollow knight" surfaces hollow_knight tags
        # but a plain "knight" does not).
        allowed_ctx: set[str] = set()
        desc_lower = description.lower()
        for kw in keywords:
            allowed_ctx.add(kw)
        # Also match multi-word parentheticals present in the description
        for token in re.findall(r"[a-z0-9_]{3,}", desc_lower.replace(" ", "_")):
            allowed_ctx.add(token)
        if keywords:
            kw_tags = self.tag_index.search_by_keywords(
                keywords,
                n=5,
                min_count=100,
                exclude=(exclude or set()) | self._BLOCKED_VOCAB_TAGS,
                allowed_parentheticals=allowed_ctx,
            )
            # Filter out tags already shown in category sections
            shown: set[str] = set()
            for line in lines:
                for tag in line.split(", "):
                    if tag:
                        shown.add(tag.rstrip("."))
            # Collect up to 4 tags per keyword to ensure niche words surface
            per_kw: dict[str, list[str]] = {}
            for tag in kw_tags:
                if tag in shown or tag in self._BLOCKED_VOCAB_TAGS:
                    continue
                for kw in keywords:
                    kw_l = kw.lower()
                    tag_parts = tag.split("_")
                    if kw_l in tag_parts or (len(tag) >= 4 and tag in kw_l):
                        per_kw.setdefault(kw_l, []).append(tag)
                        break
            # Interleave: take 1st from each keyword, then 2nd, etc.
            interleaved: list[str] = []
            seen_kw: set[str] = set()
            for i in range(3):  # up to 3 per keyword
                for kw in keywords:
                    kw_l = kw.lower()
                    bucket = per_kw.get(kw_l, [])
                    if i < len(bucket):
                        tag = bucket[i]
                        if tag not in seen_kw:
                            seen_kw.add(tag)
                            interleaved.append(tag)
                        if len(interleaved) >= 20:
                            break
                if len(interleaved) >= 20:
                    break
            if interleaved:
                lines.append(
                    f"- RELEVANT TO YOUR DESCRIPTION: {', '.join(interleaved)}"
                )

        return "\n".join(lines)

    _MODE_EXTRA_RULES: dict[str, str] = {
        "safe": (
            "SAFE MODE RULES:\n"
            "- NEVER use explicit/sexual tags (penis, fellatio, sex, nude, cum, erection, etc.).\n"
            "- Stay close to the description but be THOROUGH — tag every visual element.\n"
            "- Output 20-35 tags. Cover ALL categories that apply."
        ),
        "creative": (
            "CREATIVE MODE RULES:\n"
            "- Be EXHAUSTIVE. Tag every visual element you can imagine in this scene.\n"
            "- Invent specific clothing items, accessories, colors, materials.\n"
            "- Add specific pose details (which arm, hand position, leg position).\n"
            "- Add background objects, furniture, props that fit the scene.\n"
            "- Do NOT contradict the description. Stay SFW unless the user is explicit.\n"
            "- Do NOT add cum/ejaculation tags unless the description explicitly says so.\n"
            "- Output 35-50 tags. MORE IS BETTER. Tag like a real Danbooru annotator."
        ),
        "mature": (
            "MATURE MODE RULES:\n"
            "- Be EXHAUSTIVE like Creative mode — tag every visual element.\n"
            "- Add suggestive/explicit tags appropriate to the scene.\n"
            "- If the description IS explicit: include anatomy, positions, expressions,\n"
            "  fluids, and reactions in detail.\n"
            "- If the description is SFW: add tasteful suggestive flair (cleavage,\n"
            "  thighs, bare_shoulders, bedroom_eyes, etc.) but don't force nudity.\n"
            "- Do NOT add cum/ejaculation tags unless the description explicitly says so.\n"
            "- Output 35-50 tags. MORE IS BETTER. Tag like a real Danbooru annotator."
        ),
    }

    def _build_system_prompt(self, creativity: str, description: str = "") -> str:
        """Build a vocabulary-grounded system prompt for *creativity* mode.

        Uses structured category slots to force the LLM to cover all visual
        aspects of the scene, producing Danbooru-density tag output.
        """
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        vocab = self._build_vocabulary_section(creativity, description)
        extra_rules = self._MODE_EXTRA_RULES.get(creativity, "")

        # Structured category checklist — the key innovation for rich output
        if creativity == "safe":
            category_checklist = (
                "For EACH tag you output, it should belong to one of these categories.\n"
                "Cover AS MANY categories as apply to the scene:\n"
                "  [PARTICIPANTS] 1girl, 1boy, solo, multiple_girls, etc.\n"
                "  [HAIR] color, length, style (ponytail, twintails, braid, messy_hair, etc.)\n"
                "  [EYES] color, expression (half-closed_eyes, looking_at_viewer, etc.)\n"
                "  [CLOTHING] SPECIFIC items with colors (black_shirt, pleated_skirt, boots, etc.)\n"
                "  [ACCESSORIES] jewelry, hair_ornament, choker, belt, bag, phone, etc.\n"
                "  [BODY] body type, skin details, collarbone, bare_shoulders, etc.\n"
                "  [POSE] specific pose (standing, arm_up, hand_on_hip, crossed_arms, etc.)\n"
                "  [EXPRESSION] smile, open_mouth, blush, frown, etc.\n"
                "  [SETTING] location, background, indoors/outdoors\n"
                "  [FRAMING] full_body, upper_body, cowboy_shot, close-up, from_behind, etc.\n"
                "  [QUALITY] highres, absurdres, detailed, etc."
            )
            target_range = "20-35 tags"
        elif creativity == "creative":
            category_checklist = (
                "For EACH tag you output, it should belong to one of these categories.\n"
                "You MUST cover ALL categories that apply — be exhaustive like a real tagger:\n"
                "  [PARTICIPANTS] 1girl, 1boy, solo, hetero, multiple_girls, etc.\n"
                "  [HAIR] color + length + style (e.g. long_hair, black_hair, twintails, sidelocks)\n"
                "  [EYES] color + expression (e.g. red_eyes, looking_at_viewer, half-closed_eyes)\n"
                "  [CLOTHING] EVERY visible item with color/material (e.g. black_shirt, pleated_skirt,\n"
                "    thighhighs, boots, jacket — be SPECIFIC about each piece)\n"
                "  [ACCESSORIES] choker, belt, jewelry, earrings, hair_ornament, phone, bag, etc.\n"
                "  [BODY] collarbone, navel, bare_shoulders, thighs, midriff, etc.\n"
                "  [POSE] SPECIFIC actions (arm_up, hand_on_hip, holding_phone, crossed_arms,\n"
                "    leaning_forward, v, peace_sign, selfie, etc.)\n"
                "  [EXPRESSION] smile, open_mouth, blush, parted_lips, teeth, :d, etc.\n"
                "  [SETTING] specific location + objects (bench, desk, couch, window, etc.)\n"
                "  [LIGHTING] sunlight, shadow, backlighting, dim_lighting, etc.\n"
                "  [ATMOSPHERE] depth_of_field, bokeh, wind, rain, petals, etc.\n"
                "  [FRAMING] full_body, cowboy_shot, upper_body, from_behind, dutch_angle, etc.\n"
                "  [QUALITY] highres, absurdres, detailed, etc."
            )
            target_range = "35-50 tags"
        else:  # mature
            category_checklist = (
                "For EACH tag you output, it should belong to one of these categories.\n"
                "You MUST cover ALL categories that apply — be exhaustive like a real tagger:\n"
                "  [PARTICIPANTS] 1girl, 1boy, solo, hetero, multiple_girls, etc.\n"
                "  [HAIR] color + length + style\n"
                "  [EYES] color + expression\n"
                "  [CLOTHING] EVERY visible item OR state of undress (nude, topless, etc.)\n"
                "  [ACCESSORIES] choker, jewelry, collar, restraints, etc.\n"
                "  [BODY] breasts, ass, thighs, navel, collarbone, skin color, etc.\n"
                "  [NSFW] sexual acts, positions, anatomy, fluids, reactions AS APPROPRIATE\n"
                "  [POSE] specific body position and limb placement\n"
                "  [EXPRESSION] ahegao, blush, open_mouth, tears, moaning, etc.\n"
                "  [SETTING] specific location + furniture/objects\n"
                "  [LIGHTING] dim_lighting, backlighting, candlelight, etc.\n"
                "  [ATMOSPHERE] depth_of_field, sweat, steam, etc.\n"
                "  [FRAMING] full_body, cowboy_shot, from_behind, pov, etc.\n"
                "  [QUALITY] highres, absurdres, detailed, etc."
            )
            target_range = "35-50 tags"

        adults_only = ADULTS_ONLY_RULE
        prompt = f"""You are a Danbooru image tagger. Convert the user's description into a
COMPREHENSIVE comma-separated tag list, exactly like tags on a real Danbooru/Gelbooru post.

CRITICAL RULES:
- Output ONLY tags separated by commas. No prose, no markdown, no explanations.
- Do NOT write <think> or reasoning blocks. Output tags immediately.
- Use lowercase with underscores (Danbooru format): black_hair, looking_at_viewer
- Be SPECIFIC about clothing (not just "dress" but "black_dress, long_sleeves, frills")
- Be SPECIFIC about poses (not just "standing" but "standing, hand_on_hip, arm_up")
- Tag COLORS of items: black_boots, red_skirt, white_shirt
- Aim for {target_range}. Real Danbooru posts have 30-50+ tags. MORE IS BETTER.
- Do NOT contradict the description.
- {adults_only}
{extra_rules}

CATEGORY CHECKLIST — cover all that apply:
{category_checklist}

AVAILABLE VOCABULARY (prefer these common tags):
{vocab}

Now output ONLY the comma-separated tags for the user's description."""

        return prompt

    # ------------------------------------------------------------------
    # Generation prompt builders
    # ------------------------------------------------------------------

    def _build_generation_prompt(
        self, description: str, creativity: str, attempt_index: int = 0
    ) -> str:
        """Build the user message sent to the LLM containing description + few-shot."""
        examples = _FEWSHOT_EXAMPLES.get(creativity, _FEWSHOT_EXAMPLES["creative"])

        # Deterministically pick 2-3 examples based on description hash
        seed = hashlib.md5(description.strip().lower().encode("utf-8")).hexdigest()
        idx = int(seed[:8], 16) % max(1, len(examples) - 2)
        selected = examples[idx : idx + 3] if len(examples) > 3 else examples

        parts: list[str] = []
        for inp, out in selected:
            parts.append(f'Input: "{inp}"\nOutput: {out}')
        parts.append(f'Input: "{description.strip()}"\nOutput:')

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tag parsing & post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response_text(raw_response: str) -> str:
        """Remove common wrapper formats from model output before parsing."""
        text = raw_response.strip()
        text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        text = re.sub(r"\\(?:text|mathrm|mathbf|mathtt|operatorname)\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
        text = text.replace("{", " ").replace("}", " ")
        return text.strip()

    @staticmethod
    def _truncate_at_repetition_loop(text: str, max_reps: int = 4) -> str:
        """Truncate *text* the moment the model starts stuttering.

        Qwen3-abliterated occasionally emits a loop like
        ``saliva, saliva, saliva, saliva, ...`` that wastes the num_predict
        budget and clouds downstream parsing. Walk the comma-split tokens and
        cut the response the moment the *same token* (window=1) or the *same
        pair* (window=2) appears more than *max_reps* times.
        """
        if not text:
            return text
        parts = [p.strip() for p in re.split(r"[,\n]", text) if p.strip()]
        if len(parts) < max_reps + 2:
            return text

        # Try window sizes 1, 2, 3 — smaller wins first
        for window in (1, 2, 3):
            seen: dict[tuple[str, ...], list[int]] = {}
            for i in range(len(parts) - window + 1):
                gram = tuple(parts[i : i + window])
                seen.setdefault(gram, []).append(i)
                if len(seen[gram]) > max_reps:
                    # Non-overlapping window reps matter most; check positions
                    positions = seen[gram]
                    # Require reps to be roughly consecutive (gaps <= window)
                    consecutive = 1
                    for j in range(1, len(positions)):
                        if positions[j] - positions[j - 1] <= window:
                            consecutive += 1
                        else:
                            consecutive = 1
                        if consecutive > max_reps:
                            return ", ".join(parts[: positions[j - max_reps]])
        return text

    def _parse_tags(self, raw_response: str) -> list[str]:
        """Extract tags from LLM response, validating against the CSV index.

        Strips  think blocks then parses comma-separated tag output.
        If the result is sparse (LLM cut off mid-reasoning), the caller
        supplements with CSV keyword search results — a cleaner fallback
        than mining prose for individual words.
        """
        text = re.sub(r"<think>.*?</think>", " ", raw_response, flags=re.DOTALL)
        text = re.sub(r"<think>.*", " ", text, flags=re.DOTALL)
        text = self._clean_response_text(text)
        # Cut off any repetition loop before the budget was burned.
        text = self._truncate_at_repetition_loop(text)

        tags: list[str] = []
        seen: set[str] = set()

        for chunk in re.split(r"[\n,]+", text):
            line = chunk.strip()
            if not line:
                continue
            line = re.sub(r"^[0-9]+[.)]\s*|^[\-\*]\s+", "", line)
            line = line.strip().strip(".")
            if not line or any(c in line for c in [":", "=", ">"]):
                continue
            normalized = line.lower().replace(" ", "_")
            if normalized in seen:
                continue
            if not self.tag_index.is_valid(normalized):
                continue
            if normalized in self._BLOCKED_VOCAB_TAGS:
                continue
            # Drop disambiguator tags — these are franchise/character specific
            # and rarely what a generic description implies.
            if "_(" in normalized and normalized.endswith(")"):
                continue
            seen.add(normalized)
            tags.append(normalized)
            if len(tags) >= self.MAX_TAGS:
                break

        return tags

    # Tags that are ONLY acceptable when the description explicitly mentions them.
    # These are blocked in post-processing to prevent LLM hallucination.
    # Minor age-descriptors (loli, shota, child, ...) are NOT on this list:
    # they are unconditionally removed by backend.content_policy, regardless
    # of what the description says. See that module for why.
    _SENSITIVE_TAGS: set[str] = {
        "trap", "futanari",
        "rape", "tentacles", "bestiality", "guro", "vore",
        "necrophilia", "scat", "watersports", "urination",
    }

    def _score_tag_relevance(
        self,
        tag: str,
        description_keywords: set[str],
        expanded_concepts: set[str],
        injected_tags: set[str],
    ) -> int:
        """Score how well *tag* is supported by the description.

        Returns:
            3 — injected (actor/literal) or direct full-match keyword
            2 — expected for an archetype/setting in the description
            1 — universal scene tag (hair/eye/style/quality/pose)
            0 — no support found; probably hallucinated

        Franchise / multi-word tags (e.g. ``magic_knight_rayearth``,
        ``tales_of_symphonia_knight_of_ratatosk``) only match on partial
        components and are demoted unless their entire token set is backed
        by the description.
        """
        if tag in injected_tags:
            return 3
        tag_parts = set(tag.split("_"))
        overlap = tag_parts & description_keywords
        # Ignore generic common-word overlap (won't save a franchise tag)
        non_structural = {"of", "the", "and", "a", "an"}
        meaningful_tag_parts = tag_parts - non_structural

        if overlap:
            # For 1-part tags, any match is perfect.
            if len(meaningful_tag_parts) <= 1:
                return 3
            # For 2-part tags: distinguish descriptive compounds (shaded_face,
            # soft_lighting, forest_mist) from franchise collisions (master_sword,
            # library_of_ruina). Descriptive compounds have a modifier + a noun
            # that appears in the description — allow them at score 2.
            if len(meaningful_tag_parts) == 2:
                foreign = meaningful_tag_parts - description_keywords
                if not foreign:
                    return 3  # Both tokens in description — strong match
                # One token matches. Check if the foreign token looks like a
                # proper noun (starts uppercase in original, or is a known
                # franchise word). We work on lowercase so use a heuristic:
                # if the foreign token is short (≤4 chars) or is a common
                # modifier word, it's likely descriptive, not a franchise name.
                _COMMON_MODIFIERS = {
                    "soft", "hard", "dark", "light", "bright", "dim", "deep",
                    "high", "low", "long", "short", "wide", "narrow", "open",
                    "closed", "bare", "wet", "dry", "warm", "cool", "cold",
                    "hot", "gentle", "heavy", "light", "natural", "shaded",
                    "muted", "vivid", "pale", "rich", "thin", "thick", "small",
                    "large", "big", "tiny", "full", "half", "side", "back",
                    "front", "top", "bottom", "inner", "outer", "upper", "lower",
                    "mid", "far", "near", "distant", "close", "deep", "shallow",
                    "misty", "foggy", "sunny", "cloudy", "rainy", "snowy",
                    "windy", "stormy", "calm", "quiet", "loud", "silent",
                    "dynamic", "static", "flowing", "floating", "falling",
                    "rising", "glowing", "shining", "sparkling", "fading",
                }
                if all(tok in _COMMON_MODIFIERS or len(tok) <= 3
                       for tok in foreign):
                    return 2  # Descriptive compound — allow
                # Check archetype expansion
                if all(
                    tok in description_keywords or tok in expanded_concepts
                    for tok in meaningful_tag_parts
                ):
                    return 2
                return 0  # Likely franchise tag
            # For longer tags (franchise candidates), demand that the
            # fraction of description-supported tokens is high.
            foreign = meaningful_tag_parts - description_keywords
            if not foreign:
                return 3  # Every token in the tag is in the description
            # Need at least 75% of tokens supported for 3+ part tags
            support_ratio = 1 - (len(foreign) / len(meaningful_tag_parts))
            if support_ratio >= 0.75:
                return 2
            return 0  # Probably a franchise tag: too many foreign tokens

        if tag in expanded_concepts:
            return 2
        if tag in _UNIVERSAL_TAGS:
            return 1
        # Universal tag family matches (hair/eye/style/quality are common)
        if any(
            suffix in tag_parts
            for suffix in ("hair", "eyes", "eye", "skin", "ears", "tail", "wings")
        ):
            return 1
        return 0

    def _resolve_conflicts(self, tags: list[str]) -> list[str]:
        """Drop lower-count members of each mutual-exclusion group.

        Keeps the first-seen (which is already the highest post_count because
        post-processing sorts by count before this runs) tag per conflict
        group and drops the rest.
        """
        seen_groups: set[int] = set()
        keep: list[str] = []
        for tag in tags:
            group = _TAG_TO_CONFLICT_GROUP.get(tag)
            if group is None:
                keep.append(tag)
                continue
            if group in seen_groups:
                continue  # Conflict: drop this lower-count member
            seen_groups.add(group)
            keep.append(tag)
        return keep

    def _post_process_tags(self, tags: list[str], creativity: str, target_count: int,
                           description: str = "",
                           injected_tags: set[str] | None = None,
                           skip_relevance_gate: bool = False,
                           seed_tags: list[str] | None = None) -> list[str]:
        """Post-processing pipeline.

        gate modes (controlled via skip_relevance_gate + injected_tags):
        - skip_relevance_gate=False : strict gate (description-to-tags)
        - skip_relevance_gate=True  : enrichment gate — seeds drive the
          allowed set; thin-seed runs (only universal tags) pass everything

        The minimum gate score required to keep a tag depends on *creativity*:
        Safe demands strong support (>= 2), Creative/Mature accept any
        plausible (>= 1) tag, including universal atmosphere/style tags.
        """
        injected_tags = injected_tags or set()

        # Step 2: threshold filter
        tags = self.tag_index.filter_by_threshold(tags)

        # Step 3 (always, every mode): never emit minor age-descriptor tags.
        # This is the app's own rule, not the model's — see content_policy.
        tags = strip_minor_terms(tags, creativity)

        # Step 3a: block sensitive tags unless description explicitly mentions them
        if description:
            desc_lower = description.lower()
            tags = [t for t in tags if t not in self._SENSITIVE_TAGS
                    or any(kw in desc_lower for kw in (t, t.replace("_", " ")))]

        # Step 3b: block explicit/sex-act tags entirely in Safe mode regardless
        # of what the description says — Safe is a hard SFW guarantee.
        if creativity == "safe":
            # Block explicit sex-act tags
            tags = [t for t in tags if t not in _ACTOR_ACT_TAGS or t in {
                "1girl", "2girls", "3girls", "multiple_girls",
                "1boy", "2boys", "3boys", "multiple_boys",
                "kneeling", "bent_over", "on_all_fours", "on_back",
                "spread_legs", "lying", "open_mouth", "blush", "tears",
                "breasts", "large_breasts",
            }]
            # Also block suggestive body-focus tags in Safe mode unless the
            # description explicitly names them (e.g. user says "bikini" so
            # bare_shoulders is fine, but "cleavage" is suggestive flair).
            _SAFE_BLOCKED_SUGGESTIVE = {
                "cleavage", "navel", "thighs", "midriff", "bare_shoulders",
                "sideboob", "underboob", "ass", "bare_ass", "pantyshot",
                "skirt_lift", "showing_panties", "see-through", "wet_clothes",
                "seductive_smile", "naughty_face", "bedroom_eyes",
                "lipstick", "garter_belt",
            }
            desc_lower_safe = description.lower() if description else ""
            tags = [
                t for t in tags
                if t not in _SAFE_BLOCKED_SUGGESTIVE
                or t.replace("_", " ") in desc_lower_safe
                or t in desc_lower_safe
            ]

        # Step 4: Relevance gate — score each tag and drop tags below the
        # mode-specific threshold unless injected.
        if description:
            desc_keywords = set(self._extract_keywords(description))
            expanded: set[str] = set()
            for kw in desc_keywords:
                for expansion in _CONCEPT_EXPANSIONS.get(kw, []):
                    expanded.add(expansion)
            # Also expand common user-written phrases that differ from single keywords
            desc_lower = description.lower()
            for archetype, expansions in _CONCEPT_EXPANSIONS.items():
                if archetype in desc_lower:
                    expanded.update(expansions)

            # NSFW act expansion — only apply outside SAFE mode, since SAFE
            # explicitly forbids anatomy/sex-act tags regardless of what the
            # description says.
            if creativity != "safe":
                for kw in desc_keywords:
                    for expansion in _ACT_EXPANSIONS.get(kw, []):
                        expanded.add(expansion)
                for act_kw, expansions in _ACT_EXPANSIONS.items():
                    if act_kw in desc_lower:
                        expanded.update(expansions)

            # For enrichment, also treat every injected seed tag as a keyword
            # so the gate allows tags that are plausible neighbours of the seeds
            # even if they don't appear in the pseudo-description text.
            if skip_relevance_gate and injected_tags:
                for seed in injected_tags:
                    for part in seed.split("_"):
                        desc_keywords.add(part)
                    for part in seed.split("_"):
                        for tag in _CONCEPT_EXPANSIONS.get(part, []):
                            expanded.add(tag)
                        if creativity != "safe":
                            for tag in _ACT_EXPANSIONS.get(part, []):
                                expanded.add(tag)
                    for tag in _CONCEPT_EXPANSIONS.get(seed, []):
                        expanded.add(tag)
                    if creativity != "safe":
                        for tag in _ACT_EXPANSIONS.get(seed, []):
                            expanded.add(tag)

            scored = [
                (t, self._score_tag_relevance(t, desc_keywords, expanded, injected_tags))
                for t in tags
            ]

            # Mode-specific gate threshold:
            #   Safe     → require >= 1 (literal + expansion + universal atmosphere)
            #              Suggestive content is blocked separately in Step 3b.
            #   Creative → require >= 1 (allows universal atmosphere/style)
            #   Mature   → require >= 1 (same as Creative; wildcards added later)
            min_score = 1

            # For thin-seed enrichment, the LLM is inventing the scene from
            # scratch — drop the gate entirely.
            thin_seeds = (
                skip_relevance_gate
                and seed_tags is not None
                and not self._seeds_have_context(seed_tags)
            )
            if thin_seeds:
                tags = [t for t, _ in scored]  # gate fully open
            else:
                tags = [t for t, s in scored if s >= min_score]

        # Step 4b: Mature wildcard injection — when in Mature mode and the
        # description is not already explicit, sprinkle in 1-3 tasteful
        # suggestive tags that match the scene cues.
        if creativity == "mature" and description:
            tags = self._inject_mature_wildcards(tags, description)

        # Step 4c: Backfill — if the tag list is short, top up from concept
        # expansions and the universal atmosphere pool so the output feels
        # complete. Safe gets a lighter backfill; Creative/Mature get more.
        if description:
            tags = self._backfill_atmosphere(
                tags, description, target_count=target_count, creativity=creativity
            )

        # Step 5: semantic dedup — keep highest-count tag per synonym group
        seen_canonicals: set[str] = set()
        seen_synonyms: set[str] = set()
        deduped: list[str] = []
        for tag in tags:
            if tag in seen_synonyms:
                continue
            canon = _SYNONYM_TO_CANONICAL.get(tag)
            if canon is not None:
                # This tag is a synonym — skip if canonical already included
                if canon in seen_canonicals:
                    seen_synonyms.add(tag)
                    continue
                # Replace with canonical if canonical also in tags
                if canon in tags:
                    seen_synonyms.add(tag)
                    continue
            # Check if tag is a canonical with synonyms already present
            if tag in _SEMANTIC_DEDUP_MAP:
                if tag in seen_canonicals:
                    continue
                seen_canonicals.add(tag)
                for syn in _SEMANTIC_DEDUP_MAP[tag]:
                    seen_synonyms.add(syn)
            deduped.append(tag)

        # Step 7: sort by post_count descending (before conflict resolution so
        # higher-count tags win within a conflict group)
        deduped = self.tag_index.sort_by_count(deduped)

        # Step 6: mutual-exclusion conflict resolution
        deduped = self._resolve_conflicts(deduped)

        # Step 8: cap
        return deduped[:target_count]

    def _inject_mature_wildcards(
        self, tags: list[str], description: str
    ) -> list[str]:
        """Add tasteful suggestive tags for Mature mode on SFW descriptions.

        If the user's description (or the current tag list) already contains
        explicit content, return *tags* unchanged. Otherwise pick 1-3
        wildcards from the cue-matched pool, falling back to the generic pool.
        """
        explicit_markers = {
            "sex", "fellatio", "blowjob", "anal", "cum", "creampie", "fuck",
            "oral", "penetration", "rape", "deepthroat", "handjob", "footjob",
            "masturbation", "ahegao", "doggystyle", "missionary", "cowgirl",
        }
        desc_lower = description.lower()
        if any(marker in desc_lower for marker in explicit_markers):
            return tags
        # Already-explicit tag list → don't pile on
        explicit_tags = {
            "penis", "pussy", "fellatio", "sex", "vaginal", "anal", "cum",
            "cumshot", "creampie", "ahegao", "nude", "completely_nude",
            "doggystyle", "missionary", "cowgirl_position", "deepthroat",
        }
        if any(t in explicit_tags for t in tags):
            return tags

        # Pick wildcard pool based on cues in the description
        candidates: list[str] = []
        for cue, pool in _MATURE_WILDCARDS_BY_CUE.items():
            if cue in desc_lower:
                candidates.extend(pool)
        if not candidates:
            candidates = list(_MATURE_WILDCARDS_GENERIC)

        # Stable per-description pick — same description gets same wildcards
        seed = hashlib.md5(description.strip().lower().encode("utf-8")).hexdigest()
        offset = int(seed[:8], 16)
        existing = set(tags)
        result = list(tags)
        added = 0
        n_wildcards = 2 + (offset % 2)  # 2-3 wildcards
        for i in range(len(candidates)):
            cand = candidates[(offset + i) % len(candidates)]
            if cand in existing:
                continue
            if not self.tag_index.is_valid(cand):
                continue
            result.append(cand)
            existing.add(cand)
            added += 1
            if added >= n_wildcards:
                break
        return result

    def _backfill_atmosphere(
        self,
        tags: list[str],
        description: str,
        target_count: int,
        creativity: str,
    ) -> list[str]:
        """Top up *tags* with concept-expansion + universal atmosphere tags.

        When the LLM produces a sparse result, we still want the output to
        feel complete. This method pulls from :data:`_CONCEPT_EXPANSIONS`
        (matched against keywords/phrases in the description) and a curated
        atmosphere pool drawn from :data:`_UNIVERSAL_TAGS`.

        Safe mode gets a lighter backfill (60% of target) and skips
        suggestive tags. Creative/Mature aim for 80%.
        """
        # Safe aims lower — it should stay literal
        fill_ratio = 0.60 if creativity == "safe" else 0.80
        min_desired = max(15, int(target_count * fill_ratio))
        if len(tags) >= min_desired:
            return tags

        # Tags blocked from backfill in Safe mode
        _SAFE_BACKFILL_BLOCKED = {
            "cleavage", "navel", "thighs", "midriff", "bare_shoulders",
            "sideboob", "underboob", "wet", "wet_hair", "wet_clothes",
            "seductive_smile", "naughty_face", "bedroom_eyes", "lipstick",
            "see-through", "garter_belt", "pantyshot", "skirt_lift",
        }

        existing = set(tags)
        result = list(tags)
        desc_lower = description.lower()
        desc_keywords = set(self._extract_keywords(description))

        # 1) Concept expansions for archetypes/settings the user mentioned
        backfill_pool: list[str] = []
        for archetype, expansions in _CONCEPT_EXPANSIONS.items():
            if archetype in desc_lower or archetype in desc_keywords:
                for tag in expansions:
                    if tag not in existing and tag not in backfill_pool:
                        if creativity == "safe" and tag in _SAFE_BACKFILL_BLOCKED:
                            continue
                        backfill_pool.append(tag)

        # 2) Act expansions outside Safe mode
        if creativity != "safe":
            for act_kw, expansions in _ACT_EXPANSIONS.items():
                if act_kw in desc_lower or act_kw in desc_keywords:
                    for tag in expansions:
                        if tag not in existing and tag not in backfill_pool:
                            backfill_pool.append(tag)

        # 3) Clothing/accessory/detail pool — specific items that make
        #    output feel like real Danbooru tags (not just atmosphere)
        detail_pool: list[str] = []
        # Add clothing details based on archetype cues
        if any(kw in desc_lower for kw in ("emo", "goth", "punk")):
            detail_pool.extend([
                "black_shirt", "t-shirt", "long_sleeves", "miniskirt",
                "pleated_skirt", "black_skirt", "boots", "black_boots",
                "knee_boots", "arm_warmers", "black_arm_warmers",
                "nail_polish", "hair_ornament", "x_hair_ornament",
                "hair_over_one_eye", "twintails", "sidelocks",
                "simple_background", "red_background", "full_body",
                "standing", "solo",
            ])
        if any(kw in desc_lower for kw in ("school", "uniform", "student")):
            detail_pool.extend([
                "serafuku", "sailor_collar", "pleated_skirt", "blue_skirt",
                "white_shirt", "short_sleeves", "kneehighs", "loafers",
                "school_bag", "hair_ribbon",
            ])
        if any(kw in desc_lower for kw in ("bikini", "swimsuit", "beach", "lake")):
            detail_pool.extend([
                "barefoot", "sand", "towel", "sunscreen",
                "side-tie_bikini_bottom", "tan", "tanlines",
            ])
        if any(kw in desc_lower for kw in ("elf", "archer", "warrior", "knight")):
            detail_pool.extend([
                "bow_(weapon)", "quiver", "arrow", "cape", "leather",
                "belt", "pouch", "gauntlets", "armor",
            ])
        for tag in detail_pool:
            if tag not in existing and tag not in backfill_pool:
                if creativity == "safe" and tag in _SAFE_BACKFILL_BLOCKED:
                    continue
                if self.tag_index.is_valid(tag):
                    backfill_pool.append(tag)

        # 4) Curated atmosphere/quality pool — universal tags that
        #    enhance any image-gen prompt
        atmosphere_pool = [
            "depth_of_field", "soft_focus", "bokeh", "highres", "absurdres",
        ]
        # Add outdoor atmosphere only if scene is outdoors
        outdoor_cues = ("lake", "river", "ocean", "beach", "park", "forest",
                        "mountain", "field", "meadow", "garden", "city",
                        "street", "alley", "rooftop", "outdoor", "outside",
                        "cliff", "sunset")
        if any(c in desc_lower for c in outdoor_cues) or "outdoors" in tags:
            atmosphere_pool.extend([
                "sky", "cloud", "sunlight", "shadow", "wind",
                "blue_sky", "scenery", "outdoors",
            ])
        # Add indoor atmosphere only if scene is indoors
        indoor_cues = ("bedroom", "kitchen", "library", "office", "classroom",
                       "indoor", "inside", "bathroom", "living_room",
                       "cathedral", "church")
        if any(c in desc_lower for c in indoor_cues) or "indoors" in tags:
            atmosphere_pool.extend([
                "indoors", "window", "curtains", "warm_lighting",
                "soft_lighting", "lamp",
            ])

        for tag in atmosphere_pool:
            if tag not in existing and self.tag_index.is_valid(tag):
                if creativity == "safe" and tag in _SAFE_BACKFILL_BLOCKED:
                    continue
                backfill_pool.append(tag)
                existing.add(tag)

        # Add backfill candidates until we hit min_desired
        for cand in backfill_pool:
            if len(result) >= min_desired:
                break
            if cand in existing and cand not in result:
                result.append(cand)
            elif cand not in existing:
                if not self.tag_index.is_valid(cand):
                    continue
                if creativity == "safe" and cand in _SAFE_BACKFILL_BLOCKED:
                    continue
                result.append(cand)
                existing.add(cand)
        return result

    # ------------------------------------------------------------------
    # Literal / actor extraction  (kept from old code — still useful)
    # ------------------------------------------------------------------

    def _extract_literal_tags_from_description(self, description: str) -> list[str]:
        """Map explicit user words/phrases to known tags for safe mode."""
        desc = description.lower()
        phrase_map = [
            ("1girl", "1girl"),
            ("1boy", "1boy"),
            ("2boys", "2boys"),
            ("2girls", "2girls"),
            ("fellatio", "fellatio"),
            ("blowjob", "fellatio"),
            ("oral", "oral"),
            ("deepthroat", "deepthroat"),
            ("anal", "anal"),
            ("creampie", "creampie"),
            ("cunnilingus", "cunnilingus"),
            ("handjob", "handjob"),
            ("footjob", "footjob"),
            ("titjob", "paizuri"),
            ("titfuck", "paizuri"),
            ("paizuri", "paizuri"),
            ("gangbang", "gangbang"),
            ("spitroast", "spitroast"),
            ("threesome", "threesome"),
            ("missionary", "missionary"),
            ("doggystyle", "doggystyle"),
            ("doggy style", "doggystyle"),
            ("cowgirl", "cowgirl_position"),
            ("masturbate", "masturbation"),
            ("masturbation", "masturbation"),
            ("fingering", "fingering"),
            ("squirting", "squirting"),
            ("squirt", "squirting"),
            ("ahegao", "ahegao"),
            ("bondage", "bondage"),
            ("bdsm", "bdsm"),
            ("rape", "rape"),
            ("forced", "rape"),
            ("forcing", "rape"),
            ("sex", "sex"),
            ("fuck", "sex"),
            ("fucking", "sex"),
            ("kissing", "kiss"),
            ("kiss", "kiss"),
            ("groping", "groping"),
            ("seducing", "seductive_smile"),
            ("seduce", "seductive_smile"),
            # Participants / gender
            ("male", "1boy"),
            ("female", "1girl"),
            # Body type
            ("fat", "fat"),
            ("chubby", "fat"),
            ("busty", "large_breasts"),
            ("curvy", "curvy"),
            ("muscular", "muscular"),
            ("petite", "small_breasts"),
            ("dark skinned", "dark_skin"),
            ("dark-skinned", "dark_skin"),
            ("dark skin", "dark_skin"),
            # Subcultures / aesthetic
            ("emo", "goth"),
            ("goth", "goth"),
            ("punk", "punk"),
            # Hair colour (expand)
            ("blonde", "blonde_hair"),
            ("blond", "blonde_hair"),
            ("blonde hair", "blonde_hair"),
            ("blonde-haired", "blonde_hair"),
            ("brunette", "brown_hair"),
            ("redhead", "red_hair"),
            ("black haired", "black_hair"),
            ("black-haired", "black_hair"),
            ("silver haired", "silver_hair"),
            ("white haired", "white_hair"),
            # Archetypes / species
            ("nun", "nun"),
            ("priest", "priest"),
            ("elf", "elf"),
            ("dark elf", "dark_elf"),
            ("orc", "orc"),
            ("goblin", "goblin"),
            ("succubus", "succubus"),
            ("witch", "witch"),
            ("knight", "knight"),
            ("maid", "maid"),
            ("nurse", "nurse"),
            ("catgirl", "cat_girl"),
            ("cat girl", "cat_girl"),
            # State
            ("nude", "nude"),
            ("naked", "nude"),
            ("topless", "topless"),
            ("bottomless", "bottomless"),
            ("bikini", "bikini"),
            ("swimsuit", "swimsuit"),
            ("one-piece", "one-piece_swimsuit"),
            ("school uniform", "school_uniform"),
            # Settings (literal)
            ("lake", "lake"),
            ("river", "river"),
            ("ocean", "ocean"),
            ("beach", "beach"),
            ("forest", "forest"),
            ("park", "park"),
            ("bedroom", "bedroom"),
            ("kitchen", "kitchen"),
            ("library", "library"),
            ("classroom", "classroom"),
            ("rooftop", "rooftop"),
            ("alley", "alley"),
            ("cathedral", "cathedral"),
            ("church", "church"),
            # Anatomy (when user explicitly names them)
            ("penis", "penis"),
            ("cock", "penis"),
            ("dick", "penis"),
            ("erection", "erection"),
            ("pussy", "pussy"),
            ("vagina", "pussy"),
            ("breast", "breasts"),
            ("breasts", "breasts"),
            ("boobs", "breasts"),
            ("tits", "breasts"),
            ("nipples", "nipples"),
            ("ass", "ass"),
            ("butt", "ass"),
            # Fluids
            ("cum", "cum"),
            ("cumshot", "cumshot"),
            ("cumming", "cumshot"),
            # Expressions
            ("sloppy", "open_mouth"),
            ("messy", "messy_hair"),
            ("crying", "tears"),
            ("moaning", "moaning"),
        ]
        tags: list[str] = []
        seen: set[str] = set()
        for phrase, tag in phrase_map:
            if phrase not in desc:
                continue
            if not self.tag_index.is_valid(tag):
                continue
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _extract_actor_tags_from_description(self, description: str) -> list[str]:
        """Infer participant tags so visible actors are always included."""
        desc = description.lower()
        tags: list[str] = []

        def add(tag: str) -> None:
            if not self.tag_index.is_valid(tag):
                return
            if tag not in tags:
                tags.append(tag)

        male_markers = ["man", "male", "boy", "guy", "husband", "father",
                        "priest", "orc", "incubus", "demon", "dwarf"]
        female_markers = ["woman", "female", "girl", "nun", "lady", "wife",
                          "mother", "sister", "succubus", "witch", "maid",
                          "bride", "nurse", "schoolgirl", "mermaid", "catgirl",
                          "foxgirl"]
        plural_male_markers = ["2 men", "two men", "multiple men", "several men", "boys", "men"]
        plural_female_markers = ["2 women", "two women", "multiple women", "several women", "girls", "women"]

        has_male = any(marker in desc for marker in male_markers)
        has_female = any(marker in desc for marker in female_markers)

        if any(marker in desc for marker in plural_male_markers):
            add("2boys")
        elif has_male:
            add("1boy")

        if any(marker in desc for marker in plural_female_markers):
            add("2girls")
        elif has_female:
            add("1girl")

        # Implicit partner: acts that require a partner even if not mentioned
        partner_acts = [
            "blowjob", "fellatio", "sex", "doggystyle", "missionary",
            "cowgirl", "paizuri", "handjob", "footjob", "cunnilingus", "anal",
            "fuck", "penetration", "intercourse",
        ]
        multi_partner_acts = [
            "spitroast", "gangbang", "double_penetration", "orgy",
            "threesome", "group_sex", "multiple men",
        ]
        has_partner = any(act in desc for act in partner_acts)
        has_multi = any(act in desc for act in multi_partner_acts)
        if has_multi and not tags:
            add("1girl")
            add("multiple_boys")
        elif has_multi and len(tags) == 1:
            add("multiple_boys" if tags[0] == "1girl" else "1girl")
        elif has_partner and not tags:
            add("1girl")
            add("1boy")
        elif has_partner and len(tags) == 1:
            add("1girl" if tags[0] == "1boy" else "1boy")

        return tags

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    def generate_tags(
        self, description: str, creativity: str = DEFAULT_CREATIVITY
    ) -> DescriptionTagResult:
        """Generate Danbooru tags from a text description.

        Uses the TagFrequencyIndex for vocabulary-grounded prompting and
        post-processing quality filtering.  Retries up to 3 times when the
        LLM returns fewer than the mode's minimum tag count (3 for safe,
        5 for creative, 6 for mature) — a common failure mode with abliterated
        models whose <think> blocks consume the num_predict budget or that
        partially refuse NSFW content. Each retry bumps temperature by 0.15
        to explore a different sampling path. Best result across attempts
        is kept so a worse retry cannot overwrite a passable earlier result.
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        creativity = (creativity or self.DEFAULT_CREATIVITY).strip().lower()
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        mode_options = {
            "safe":     {"base_temp": 0.55, "num_predict": 500, "target_tags": 35,
                         "top_p": 0.92, "min_accept": 10, "max_attempts": 3},
            "creative": {"base_temp": 0.75, "num_predict": 700, "target_tags": 50,
                         "top_p": 0.95, "min_accept": 20, "max_attempts": 3},
            "mature":   {"base_temp": 0.90, "num_predict": 800, "target_tags": 50,
                         "top_p": 0.97, "min_accept": 20, "max_attempts": 3},
        }
        opts = mode_options[creativity]
        target_tags = opts["target_tags"]
        min_accept = opts["min_accept"]
        max_attempts = opts["max_attempts"]

        # Pre-compute literal/actor tags (stable across retries)
        literal_tags = self._extract_literal_tags_from_description(description)
        actor_tags = self._extract_actor_tags_from_description(description)

        last_error: Exception | None = None
        raw_text_final: str = ""
        # Track best-so-far so we don't lose a usable result to a worse retry
        best_tags: list[str] = []
        best_raw: str = ""

        # Detect model family for generation hints.
        # qwen3 supports the "/no_think" user-message directive which skips
        # the <think>...</think> reasoning block entirely. Apply it on all
        # qwen3 variants except the Instruct-2507 branch (which is already
        # non-thinking by design).
        model_lower = self.model.lower()
        no_think_prefix = ""
        is_qwen3_thinking = (
            "qwen3" in model_lower
            and "instruct-2507" not in model_lower
            and "instruct_2507" not in model_lower
        )
        if is_qwen3_thinking:
            no_think_prefix = self._MODEL_PREFILLS.get("qwen3_no_think") or ""

        for attempt in range(max_attempts):
            try:
                # Build prompts
                system_prompt = self._build_system_prompt(creativity, description)
                gen_prompt = self._build_generation_prompt(description, creativity)
                
                # Prepend /no_think directive on qwen3 so the model skips
                # its reasoning block. Placed at the very start of the user
                # message — Qwen's documented location for this hint.
                if no_think_prefix:
                    gen_prompt = f"{no_think_prefix}\n\n{gen_prompt}"

                # Bump temperature on retry to shake the model out of a rut.
                # Each retry nudges temp + top_p slightly higher to encourage
                # a different sampling path.
                temp = opts["base_temp"] + 0.15 * attempt
                temp = min(1.35, temp)

                response = self.client.generate(
                    model=self.model,
                    prompt=gen_prompt,
                    system=system_prompt,
                    options={
                        "temperature": temp,
                        "top_p": opts["top_p"],
                        "top_k": 50,
                        "repeat_penalty": 1.12,
                        "num_predict": opts["num_predict"],
                        "stop": ["</think>", "\n\n\n", "\n\nInput:", "Output:", "<think>"],  # Aggressive stop
                    },
                    stream=False,
                )
                raw_text = response.get("response", "").strip()
                raw_text_final = raw_text

                # Parse and post-process
                parsed_tags = self._parse_tags(raw_text)

                # Inject literal/actor tags at front
                injected: set[str] = set()
                merged: list[str] = []
                for tag in actor_tags + literal_tags + parsed_tags:
                    if tag in injected:
                        continue
                    injected.add(tag)
                    merged.append(tag)

                # Mark the tags that must bypass the relevance gate
                must_keep: set[str] = set(actor_tags) | set(literal_tags)

                final_tags = self._post_process_tags(
                    merged, creativity, target_tags, description,
                    injected_tags=must_keep,
                )

                # Track best attempt so a worse retry can't overwrite a
                # passable earlier result.
                if len(final_tags) > len(best_tags):
                    best_tags = list(final_tags)
                    best_raw = raw_text_final

                # Accept based on LLM-produced tags (before backfill) to
                # avoid accepting sparse LLM output that was padded by
                # backfill. parsed_tags is what the model actually generated.
                llm_produced = len(parsed_tags)
                if llm_produced >= min_accept:
                    return DescriptionTagResult(
                        tags=final_tags[:target_tags],
                        raw_response=raw_text_final,
                        model=self.model,
                    )
                # Otherwise loop to retry

            except Exception as e:
                last_error = e
                # Don't retry on connection/network errors
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Tag generation failed: {e}")

        # Exhausted retries — return whatever we have (even if sparse)
        # or raise if every attempt was an exception
        if last_error is not None and not best_tags:
            raise RuntimeError(f"Tag generation failed after retries: {last_error}")

        # Prefer the best attempt we tracked across retries
        if best_tags:
            return DescriptionTagResult(
                tags=best_tags[:target_tags],
                raw_response=best_raw,
                model=self.model,
            )

        # Rebuild best-effort result from the last attempt
        parsed_tags = self._parse_tags(raw_text_final)
        injected = set()
        merged = []
        for tag in actor_tags + literal_tags + parsed_tags:
            if tag in injected:
                continue
            injected.add(tag)
            merged.append(tag)
        must_keep: set[str] = set(actor_tags) | set(literal_tags)
        final_tags = self._post_process_tags(
            merged, creativity, target_tags, description,
            injected_tags=must_keep,
        )

        # Final rescue: if the pipeline still produced nothing, synthesise a
        # minimal tag set from concept expansions so the user at least sees a
        # sensible seed they can build on. Better than an empty result.
        if not final_tags:
            rescue: list[str] = list(actor_tags) + list(literal_tags)
            desc_lower = description.lower()
            desc_keywords = self._extract_keywords(description)
            for archetype, expansions in _CONCEPT_EXPANSIONS.items():
                if archetype in desc_lower or archetype in desc_keywords:
                    for tag in expansions:
                        if self.tag_index.is_valid(tag) and tag not in rescue:
                            rescue.append(tag)
            # Include act expansions outside of safe mode — a failed
            # NSFW run should still return fellatio/penis etc. when the
            # description names the act.
            if creativity != "safe":
                for act_kw, expansions in _ACT_EXPANSIONS.items():
                    if act_kw in desc_lower or act_kw in desc_keywords:
                        for tag in expansions:
                            if self.tag_index.is_valid(tag) and tag not in rescue:
                                rescue.append(tag)
            final_tags = self._post_process_tags(
                rescue, creativity, target_tags, description,
                injected_tags=set(rescue),  # bypass gate — every tag here is description-grounded
            )

        return DescriptionTagResult(
            tags=final_tags[:target_tags],
            raw_response=raw_text_final,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Tag enrichment (seed tags → expanded tag list)
    # ------------------------------------------------------------------

    def _parse_seed_tags(self, raw: str | list[str]) -> list[str]:
        """Normalize seed-tag input and validate against the CSV vocabulary.

        Accepts either a raw comma/newline-separated string or a pre-split
        list. Returns only tags that exist in the Danbooru index, preserving
        the user's order. Unknown tags are silently dropped.
        """
        if isinstance(raw, str):
            chunks = re.split(r"[\n,]+", raw)
        else:
            chunks = list(raw)

        seen: set[str] = set()
        valid: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, str):
                continue
            tag = chunk.strip().lower().replace(" ", "_").strip(".")
            if not tag or tag in seen:
                continue
            if not self.tag_index.is_valid(tag):
                continue
            seen.add(tag)
            valid.append(tag)
        return valid

    # Tags that are "universal" — they carry no scene/archetype/act context
    # on their own. Seeds consisting only of these need special handling.
    _CONTEXT_FREE_TAGS: set[str] = {
        "1girl", "2girls", "3girls", "multiple_girls",
        "1boy", "2boys", "3boys", "multiple_boys",
        "solo", "duo", "couple",
        "standing", "sitting", "kneeling", "lying", "squatting",
        "leaning_forward", "leaning_back", "leaning_to_the_side",
        "bent_over", "on_all_fours", "on_back", "on_stomach",
        "looking_at_viewer", "looking_away", "looking_back",
        "looking_up", "looking_down",
        "smile", "blush", "frown", "angry", "surprised",
        "open_mouth", "closed_mouth", "parted_lips",
        "long_hair", "short_hair", "medium_hair",
        "blonde_hair", "brown_hair", "black_hair", "white_hair",
        "blue_hair", "red_hair", "pink_hair", "silver_hair",
        "blue_eyes", "brown_eyes", "green_eyes", "red_eyes",
        "highres", "absurdres", "masterpiece", "best_quality",
    }

    def _seeds_have_context(self, seeds: list[str]) -> bool:
        """Return True if seeds contain at least one archetype, setting, or act tag."""
        for seed in seeds:
            if seed in self._CONTEXT_FREE_TAGS:
                continue
            # Any tag not in the context-free set is considered contextual
            return True
        return False

    def _build_enrichment_system_prompt(
        self, creativity: str, seed_tags: list[str]
    ) -> str:
        """Build a vocabulary-grounded system prompt for enrichment mode.

        When seeds are thin (only universal/pose tags), the prompt explicitly
        asks the LLM to invent a plausible scene rather than just expand.
        """
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        pseudo_desc = ", ".join(seed_tags)
        vocab = self._build_vocabulary_section(
            creativity, pseudo_desc, exclude=set(seed_tags)
        )
        mode_rules = self._MODE_EXTRA_RULES.get(creativity, "")

        thin_seeds = not self._seeds_have_context(seed_tags)

        if thin_seeds:
            task_instruction = (
                "The user has given you only basic pose/character tags with no scene context.\n"
                "Your job is to INVENT a plausible, visually rich scene around these tags.\n"
                "Add: clothing, hair details, expression, setting, lighting, atmosphere,\n"
                "     objects, style cues, and any other tags that make the image vivid.\n"
                "Be creative and specific — generic tags like 'outdoors' are fine but also\n"
                "add concrete details like 'cherry_blossoms', 'school_uniform', 'sunset', etc."
            )
        else:
            task_instruction = (
                "Your job is to ADD more tags that plausibly belong in the same scene —\n"
                "things like clothing, atmosphere, lighting, expressions, body language,\n"
                "objects, and style cues."
            )
        task_instruction += f"\n{ADULTS_ONLY_RULE}"

        return f"""You are a Danbooru-style tag enricher. The user gives you a list of seed tags.
{task_instruction}

CRITICAL RULES:
- Output ONLY the NEW tags (comma-separated). Do NOT repeat any seed tag.
- Do NOT write <think> or reasoning blocks. Output tags immediately.
- Use lowercase with underscores for spaces (Danbooru format).
- Do NOT invent new characters, franchises, or contradict the existing tags.
- Prefer common tags from the vocabulary below.
- Output 15-30 additional tags.
{mode_rules}

AVAILABLE VOCABULARY (most common Danbooru tags — prefer these):
{vocab}

OUTPUT FORMAT (example):
school_uniform, pleated_skirt, cherry_blossoms, smile, outdoors, day, wind

Now output ONLY the comma-separated ADDITIONAL tags that complement the user's seeds."""

    def _build_enrichment_generation_prompt(
        self, seed_tags: list[str], creativity: str
    ) -> str:
        """Build the user message for enrichment with few-shot examples."""
        examples = _ENRICH_FEWSHOT_EXAMPLES.get(
            creativity, _ENRICH_FEWSHOT_EXAMPLES["creative"]
        )
        seed_str = ", ".join(seed_tags)
        digest = hashlib.md5(seed_str.lower().encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % max(1, len(examples) - 2)
        selected = examples[idx : idx + 3] if len(examples) > 3 else examples

        parts: list[str] = []
        for seeds, out in selected:
            parts.append(f'Seeds: {seeds}\nAdd: {out}')
        parts.append(f'Seeds: {seed_str}\nAdd:')
        return "\n\n".join(parts)

    def enrich_tags(
        self,
        seed_tags: list[str] | str,
        creativity: str = DEFAULT_CREATIVITY,
    ) -> DescriptionTagResult:
        """Expand a list of seed tags with complementary tags from the LLM.

        The seeds are preserved in the output (always marked as injected/must-keep).
        The LLM generates additional tags which pass through the same
        post-processing pipeline as description-based generation.
        """
        creativity = (creativity or self.DEFAULT_CREATIVITY).strip().lower()
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        seeds = self._parse_seed_tags(seed_tags)
        if not seeds:
            raise ValueError(
                "No valid seed tags provided. Enter at least one Danbooru tag."
            )

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        mode_options = {
            "safe":     {"base_temp": 0.55, "num_predict": 500, "target_tags": 35,
                         "top_p": 0.92, "min_new_accept": 15, "max_attempts": 3},
            "creative": {"base_temp": 0.70, "num_predict": 700, "target_tags": 50,
                         "top_p": 0.95, "min_new_accept": 20, "max_attempts": 3},
            "mature":   {"base_temp": 0.90, "num_predict": 800, "target_tags": 50,
                         "top_p": 0.97, "min_new_accept": 20, "max_attempts": 3},
        }
        opts = mode_options[creativity]
        target_tags = opts["target_tags"]
        min_new_accept = opts["min_new_accept"]
        max_attempts = opts["max_attempts"]

        # Synthesize a pseudo-description so the relevance gate + concept/act
        # expansions fire on seed tokens.
        pseudo_desc = " ".join(t.replace("_", " ") for t in seeds)

        # qwen3 /no_think directive (same as generate_tags)
        model_lower = self.model.lower()
        no_think_prefix = ""
        is_qwen3_thinking = (
            "qwen3" in model_lower
            and "instruct-2507" not in model_lower
            and "instruct_2507" not in model_lower
        )
        if is_qwen3_thinking:
            no_think_prefix = self._MODEL_PREFILLS.get("qwen3_no_think") or ""

        best_new: list[str] = []
        best_raw = ""
        last_error: Exception | None = None
        raw_text_final = ""

        for attempt in range(max_attempts):
            try:
                system_prompt = self._build_enrichment_system_prompt(creativity, seeds)
                gen_prompt = self._build_enrichment_generation_prompt(seeds, creativity)
                if no_think_prefix:
                    gen_prompt = f"{no_think_prefix}\n\n{gen_prompt}"

                temp = min(1.35, opts["base_temp"] + 0.15 * attempt)

                response = self.client.generate(
                    model=self.model,
                    prompt=gen_prompt,
                    system=system_prompt,
                    options={
                        "temperature": temp,
                        "top_p": opts["top_p"],
                        "top_k": 50,
                        "repeat_penalty": 1.12,
                        "num_predict": opts["num_predict"],
                        "stop": ["</think>", "\n\n\n", "\nSeeds:", "Seeds:", "<think>"],
                    },
                    stream=False,
                )
                raw_text = response.get("response", "").strip()
                raw_text_final = raw_text

                parsed = self._parse_tags(raw_text)
                seeds_set = set(seeds)
                new_only = [t for t in parsed if t not in seeds_set]

                combined = list(seeds) + new_only
                final_tags = self._post_process_tags(
                    combined, creativity, target_tags, pseudo_desc,
                    injected_tags=seeds_set,
                    skip_relevance_gate=True,
                    seed_tags=seeds,
                )
                produced_new = [t for t in final_tags if t not in seeds_set]

                if len(produced_new) > len(best_new):
                    best_new = produced_new
                    best_raw = raw_text_final

                if len(produced_new) >= min_new_accept:
                    return DescriptionTagResult(
                        tags=final_tags[:target_tags],
                        raw_response=raw_text_final,
                        model=self.model,
                    )

            except Exception as e:
                last_error = e
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Tag enrichment failed: {e}")

        if last_error is not None and not best_new:
            raise RuntimeError(f"Tag enrichment failed after retries: {last_error}")

        combined = list(seeds) + best_new
        final_tags = self._post_process_tags(
            combined, creativity, target_tags, pseudo_desc,
            injected_tags=set(seeds),
            skip_relevance_gate=True,
            seed_tags=seeds,
        )

        # Rescue path — if nothing new survived, fall back to concept/act
        # expansions for any archetype tokens present in the seeds.
        if not [t for t in final_tags if t not in set(seeds)]:
            rescue = list(seeds)
            for seed in seeds:
                for token in seed.split("_"):
                    for tag in _CONCEPT_EXPANSIONS.get(token, []):
                        if self.tag_index.is_valid(tag) and tag not in rescue:
                            rescue.append(tag)
                    if creativity != "safe":
                        for tag in _ACT_EXPANSIONS.get(token, []):
                            if self.tag_index.is_valid(tag) and tag not in rescue:
                                rescue.append(tag)
            # For thin seeds, also pull top tags from relevant categories
            if not self._seeds_have_context(seeds):
                cats = self._VOCAB_CATEGORIES_BY_MODE.get(creativity, ["character", "clothing", "setting", "action"])
                cat_tags = self.tag_index.top_by_category(n=6, categories=cats, exclude=set(rescue))
                for cat_list in cat_tags.values():
                    for tag in cat_list:
                        if tag not in rescue:
                            rescue.append(tag)
            final_tags = self._post_process_tags(
                rescue, creativity, target_tags, pseudo_desc,
                injected_tags=set(rescue),
                skip_relevance_gate=True,
                seed_tags=seeds,
            )

        return DescriptionTagResult(
            tags=final_tags[:target_tags],
            raw_response=best_raw or raw_text_final,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Natural-language prompt generation  (stage 2: tags -> prose)
    # ------------------------------------------------------------------

    # Target prose length. Natural-language image models (Krea, Flux and
    # similar T5-encoder models) condition well on a descriptive paragraph;
    # past roughly this range the extra tokens mostly dilute the subject.
    _PROSE_TARGET_WORDS: tuple[int, int] = (70, 110)

    # Minimum words before a prose attempt is accepted. Well below the target
    # range — this only rejects a truncated or refused generation.
    _PROSE_MIN_WORDS = 35

    # Hard ceiling. Models routinely blow past the stated target range, and an
    # over-long prompt dilutes the subject on these encoders, so anything past
    # this is trimmed back to a sentence boundary. Set above the target range
    # so a slightly long-but-complete paragraph survives intact.
    _PROSE_MAX_WORDS = 130

    _PROSE_MODE_RULES: dict[str, str] = {
        "safe": (
            "CONTENT: Keep the description entirely non-sexual. Describe clothing, "
            "pose and setting plainly. No nudity, no suggestive framing."
        ),
        "creative": (
            "CONTENT: Stay non-explicit unless the tags clearly call for otherwise. "
            "Rich visual detail is the goal — materials, textures, light, atmosphere."
        ),
        "mature": (
            "CONTENT: Adult content is permitted. Describe bodies, state of dress, "
            "acts and expressions directly and in plain language as the tags "
            "indicate. Keep it descriptive prose, never a clinical tag list. "
            + ADULTS_ONLY_RULE
        ),
    }

    def _no_think_prefix(self) -> str:
        """Return the '/no_think' directive when the active model needs it.

        qwen3 variants (except the already non-thinking instruct-2507 branch)
        burn their generation budget on a <think> block unless told otherwise.
        """
        model_lower = self.model.lower()
        is_qwen3_thinking = (
            "qwen3" in model_lower
            and "instruct-2507" not in model_lower
            and "instruct_2507" not in model_lower
        )
        if is_qwen3_thinking:
            return self._MODEL_PREFILLS.get("qwen3_no_think") or ""
        return ""

    def _build_prose_system_prompt(self, creativity: str) -> str:
        """Build the system prompt that turns a tag set into a prose prompt."""
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        lo, hi = self._PROSE_TARGET_WORDS
        mode_rules = self._PROSE_MODE_RULES.get(creativity, "")

        return f"""You write prompts for natural-language image models (Krea, Flux and
similar). The user gives you a Danbooru tag set. You rewrite it as ONE flowing
paragraph of descriptive English that reads like a person describing a
photograph or illustration out loud.

CRITICAL RULES:
- Output ONE paragraph of plain prose. Nothing else.
- Do NOT write <think> or reasoning blocks. Write the paragraph immediately.
- NEVER output a comma-separated tag list. Write real sentences.
- NEVER use underscores. "black_hair" becomes "black hair".
- NEVER use weight syntax like (detailed:1.3) or ((emphasis)). It does nothing
  on these models.
- NEVER add quality spam like "masterpiece, best quality, 4k, highres,
  absurdres". Those are booru-model habits and are pure noise here.
- Do NOT write a negative prompt, a title, a preamble, or commentary.
- Do NOT wrap the output in quotes or markdown.

WHAT TO WRITE:
- Open with the main subject in the very first clause — these models weight
  early tokens most heavily. Start with the person or object, not the setting.
- Then work outward in this order: appearance (hair, eyes, build, skin),
  clothing with materials and colors, pose and action, the setting and its
  objects, lighting, then mood.
- Close with the shot: framing, camera angle, lens or depth-of-field feel, and
  the overall style or medium.
- Convert every tag faithfully. Do not silently drop the distinctive ones.
- NEVER contradict a tag. If boots are tagged she is not barefoot; if a shirt
  is tagged she is not topless; if it is tagged night the sky is not sunlit.
  Re-read the tags before you write each clause.
- You may add small neutral connective detail to make sentences read naturally,
  but never a new garment, body feature, object or time of day the tags do not
  mention.
- Merge related tags into natural phrases: "long_hair, black_hair, twintails"
  becomes "long black hair in twintails", not three separate clauses.
- LENGTH IS A HARD LIMIT: {lo}-{hi} words, and never more than {hi}. Write
  tight, information-dense sentences and stop. A long prompt is a worse prompt
  — it dilutes the subject. Do not pad.
{mode_rules}

Write only the paragraph."""

    def _build_prose_generation_prompt(
        self,
        tags: list[str],
        description: str,
        creativity: str,
        seed_tags_mode: bool = False,
    ) -> str:
        """Build the user message for the prose stage: few-shot + tags + intent."""
        examples = _PROSE_FEWSHOT.get(creativity, _PROSE_FEWSHOT["creative"])

        # Deterministically rotate the few-shot pair so re-runs on the same
        # input stay stable while different inputs see different examples.
        seed = hashlib.md5(",".join(tags).encode("utf-8")).hexdigest()
        start = int(seed[:8], 16) % max(1, len(examples))
        ordered = examples[start:] + examples[:start]

        parts: list[str] = []
        for ex_tags, ex_prose in ordered[:2]:
            parts.append(f"TAGS:\n{ex_tags}\n\nPROMPT:\n{ex_prose}")

        tag_line = ", ".join(tags)
        block = [f"TAGS:\n{tag_line}"]

        # Give the model the user's own words as intent context. It anchors the
        # subject and tone; the tags remain the source of visual truth.
        original = (description or "").strip()
        if original and not seed_tags_mode:
            block.append(
                f"\nThe user originally asked for: \"{original}\"\n"
                "Honour that intent, but describe everything the tags list."
            )

        parts.append("\n\n".join(block) + "\n\nPROMPT:")
        return "\n\n---\n\n".join(parts)

    # -- prose post-processing -----------------------------------------

    _PROSE_META_LINE = re.compile(
        r"^\s*(?:note|tip|explanation|word count|this (?:prompt|description|paragraph)|"
        r"i hope|hope this|let me know|feel free)\b",
        re.IGNORECASE,
    )

    # Reasoning that leaks as plain text rather than inside a <think> block —
    # qwen3 does this regularly, both before and after the actual paragraph.
    # Matched per sentence, so one leaked sentence does not discard the prose.
    _PROSE_META_SENTENCE = re.compile(
        r"\b(?:okay,?\s+(?:let'?s|so)|let'?s see|let me (?:see|think|start|know|write)|"
        r"i (?:need|should|will|have|'ll|'m going) to|i'?ll (?:write|start|convert)|"
        r"the user (?:wants|asked|said|originally|gave|provided)|"
        r"based on (?:the|these) (?:user|given|provided|instructions|tags provided)|"
        r"the prompt is built|according to the instructions|as an ai|"
        r"word count|per the instructions|first,? i)\b",
        re.IGNORECASE,
    )

    # A meta preamble that hands off into the real description mid-sentence.
    # Everything up to and including the handoff phrase is commentary.
    _PROSE_LEAD_HANDOFF = re.compile(
        r"^.{0,240}?\b(?:the prompt is built around|the (?:image|scene) "
        r"(?:shows|depicts|features)|here'?s the (?:prompt|paragraph)[:,]?)\s+",
        re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def _clean_prose_output(cls, text: str) -> str:
        """Scrub LLM prose down to a single clean paragraph.

        Removes reasoning blocks, markdown, tag-model syntax the prose stage
        must never emit (underscores, attention weights), and the trailing
        commentary chatty models like to append.
        """
        if not text:
            return ""

        # Reasoning blocks — closed, or dangling in either direction when the
        # stop sequence cut the generation mid-block.
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # Code fences and leading labels ("Prompt:", "Here is the prompt:").
        text = re.sub(r"```[a-zA-Z]*", " ", text)
        text = text.replace("```", " ")
        text = re.sub(
            r"^\s*(?:here(?:'s| is)[^:\n]{0,60}:|prompt\s*:|output\s*:|answer\s*:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Attention weights: (foo:1.3) -> foo, ((foo)) -> foo. Loop for nesting.
        for _ in range(3):
            new = re.sub(r"\(([^()]*?):\s*-?\d+(?:\.\d+)?\)", r"\1", text)
            new = re.sub(r"\(\(([^()]*?)\)\)", r"\1", new)
            if new == text:
                break
            text = new

        # Drop trailing commentary lines, bullets and blank lines; the result
        # must read as one paragraph.
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if cls._PROSE_META_LINE.match(stripped):
                continue
            stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", stripped)
            kept.append(stripped)
        text = " ".join(kept)

        # Underscores are a tag-format artefact — never valid in prose output.
        text = re.sub(r"(?<=\w)_(?=\w)", " ", text)
        text = text.replace("_", " ")

        # Remaining markdown noise, then whitespace normalisation.
        text = re.sub(r"[*`#>\[\]]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # A meta preamble that runs into the real description: keep only the
        # part after the handoff phrase.
        handoff = cls._PROSE_LEAD_HANDOFF.match(text)
        if handoff:
            text = text[handoff.end():].lstrip("\"'“‘( ")
            if text:
                text = text[0].upper() + text[1:]

        # Drop whole sentences that are leaked reasoning rather than description.
        sentences = re.findall(r"[^.!?]*[.!?]+(?:\s|$)|[^.!?]+$", text)
        if len(sentences) > 1:
            kept_sentences = [
                s for s in sentences if not cls._PROSE_META_SENTENCE.search(s)
            ]
            if kept_sentences:
                text = "".join(kept_sentences).strip()

        # Unbalanced quotes/parens left behind by a stripped preamble.
        text = text.strip("\"'“”‘’ ")
        if text.count("(") != text.count(")"):
            text = text.replace("(", "").replace(")", "")
        text = re.sub(r"\s+", " ", text).strip()

        # Strip symmetric wrapping quotes.
        if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
            text = text[1:-1].strip()

        # Close an unterminated final sentence.
        if text and text[-1] not in ".!?":
            text = text.rstrip(",;: ") + "."

        return text

    @classmethod
    def _contains_meta_leak(cls, text: str) -> bool:
        """True when reasoning/commentary survived cleanup.

        Used as a retry trigger: a leaked attempt is regenerated rather than
        handed to the user, since cleanup cannot catch every phrasing.
        """
        return bool(text) and bool(cls._PROSE_META_SENTENCE.search(text))

    @classmethod
    def _trim_to_word_budget(cls, text: str) -> str:
        """Cut prose back to ``_PROSE_MAX_WORDS`` at a sentence boundary.

        Models regularly overshoot the stated range, and an over-long prompt
        dilutes the subject on these encoders. Trimming at a sentence boundary
        keeps the result readable; a mid-sentence cut would not.
        """
        words = text.split()
        if len(words) <= cls._PROSE_MAX_WORDS:
            return text

        # Walk sentences, keeping the longest prefix that fits the budget.
        sentences = re.findall(r"[^.!?]*[.!?]+(?:\s|$)|[^.!?]+$", text)
        kept: list[str] = []
        used = 0
        for sentence in sentences:
            n = len(sentence.split())
            if used + n > cls._PROSE_MAX_WORDS and kept:
                break
            kept.append(sentence)
            used += n

        trimmed = "".join(kept).strip()
        # The loop always keeps the first sentence so it can never return
        # nothing — which means a single runaway sentence arrives here still
        # over budget. Hard-cut it and close it off.
        if not trimmed or len(trimmed.split()) > cls._PROSE_MAX_WORDS:
            trimmed = " ".join(words[: cls._PROSE_MAX_WORDS]).rstrip(",;: ") + "."
        return trimmed

    @staticmethod
    def _looks_like_tag_list(text: str) -> bool:
        """True when the model ignored the format and emitted tag soup."""
        if not text:
            return True
        segments = [s.strip() for s in text.split(",") if s.strip()]
        if len(segments) < 8:
            return False
        avg_words = sum(len(s.split()) for s in segments) / len(segments)
        # Real prose averages well above three words between commas.
        return avg_words < 3.0

    @staticmethod
    def _prose_fallback(tags: list[str]) -> str:
        """Deterministic last-resort prose when every LLM attempt fails.

        Not as good as a generated paragraph, but it keeps the feature from
        returning nothing when Ollama is having a bad day.
        """
        if not tags:
            return ""
        readable = [t.replace("_", " ") for t in tags]
        subject = readable[0]
        rest = readable[1:]
        if not rest:
            return f"A detailed image of {subject}."
        body = ", ".join(rest[:-1])
        tail = rest[-1]
        joined = f"{body} and {tail}" if body else tail
        return (
            f"A detailed image of {subject}, showing {joined}. "
            "Natural lighting, clear composition, high level of detail."
        )

    def generate_natural_prompt(
        self,
        description: str,
        creativity: str = DEFAULT_CREATIVITY,
        seed_tags_mode: bool = False,
        tag_result: Optional[DescriptionTagResult] = None,
    ) -> NaturalPromptResult:
        """Generate a natural-language prompt for Krea/Flux-style models.

        Runs in two stages. Stage 1 produces a Danbooru tag set through the
        existing pipeline (``generate_tags``, or ``enrich_tags`` when
        *seed_tags_mode* is set), so the prose inherits vocabulary grounding,
        concept expansion, dedup and conflict resolution. Stage 2 rewrites that
        cleaned tag set as one flowing paragraph.

        Pass *tag_result* to reuse tags already generated and skip stage 1.

        Retries the prose stage up to 3 times when the model returns something
        truncated or still tag-shaped, nudging temperature each time.
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        creativity = (creativity or self.DEFAULT_CREATIVITY).strip().lower()
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        # --- Stage 1: description/seeds -> cleaned Danbooru tags ---------
        if tag_result is None:
            if seed_tags_mode:
                tag_result = self.enrich_tags(description, creativity=creativity)
            else:
                tag_result = self.generate_tags(description, creativity=creativity)

        tags = list(tag_result.tags)
        if not tags:
            raise RuntimeError(
                "Stage 1 produced no tags, so there is nothing to write a prompt "
                "from. Try a more specific description."
            )

        # --- Stage 2: tags -> prose --------------------------------------
        system_prompt = self._build_prose_system_prompt(creativity)
        gen_prompt = self._build_prose_generation_prompt(
            tags, description, creativity, seed_tags_mode=seed_tags_mode
        )
        no_think = self._no_think_prefix()
        if no_think:
            gen_prompt = f"{no_think}\n\n{gen_prompt}"

        max_attempts = 3
        base_temp = 0.70
        best_prose = ""
        raw_final = ""
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                temp = min(1.05, base_temp + 0.10 * attempt)
                response = self.client.generate(
                    model=self.model,
                    prompt=gen_prompt,
                    system=system_prompt,
                    options={
                        "temperature": temp,
                        "top_p": 0.92,
                        "top_k": 50,
                        # Prose repeats connectives naturally; a tag-level
                        # repeat penalty makes it read strangely.
                        "repeat_penalty": 1.05,
                        # Sized for the word budget plus the overshoot these
                        # models reliably produce; _trim_to_word_budget handles
                        # the rest. Keeping it tight also keeps stage 2 fast.
                        "num_predict": 220,
                        "stop": ["<think>", "\n\nTAGS:", "\n\n---", "\n\nNote:"],
                    },
                    stream=False,
                )
                raw_final = response.get("response", "").strip()
                prose = self._trim_to_word_budget(self._clean_prose_output(raw_final))

                usable = (
                    len(prose.split()) >= self._PROSE_MIN_WORDS
                    and not self._looks_like_tag_list(prose)
                    and not self._contains_meta_leak(prose)
                )

                # Only track a clean attempt as "best" — a longer but leaked
                # attempt must never beat a shorter clean one.
                if usable and len(prose.split()) > len(best_prose.split()):
                    best_prose = prose

                if usable:
                    return NaturalPromptResult(
                        prompt=prose,
                        tags=tags,
                        raw_response=raw_final,
                        model=self.model,
                    )
                # Otherwise loop to retry

            except Exception as e:
                last_error = e
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Prompt generation failed: {e}")

        # Exhausted retries. best_prose only ever holds a clean attempt, so
        # fall back to the deterministic paragraph when it is empty.
        if best_prose:
            return NaturalPromptResult(
                prompt=best_prose,
                tags=tags,
                raw_response=raw_final,
                model=self.model,
            )

        if last_error is not None and not best_prose:
            raise RuntimeError(
                f"Prompt generation failed after retries: {last_error}"
            )

        return NaturalPromptResult(
            prompt=self._prose_fallback(tags),
            tags=tags,
            raw_response=raw_final,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Ollama connection & model management
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """Get list of available local models."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = []
            for item in payload.get("models", []):
                name = item.get("model") or item.get("name")
                if name:
                    models.append(name)
            return models
        except Exception:
            try:
                resp = self.client.list()
                return [m.model for m in resp.models if getattr(m, "model", None)]
            except Exception:
                return []

    def pull_model(self, model: str) -> None:
        """Download a model from Ollama registry."""
        try:
            self.client.pull(model)
        except Exception as e:
            raise RuntimeError(f"Failed to pull model {model}: {e}")

    def delete_model(self, model: str) -> None:
        """Delete a model from the local Ollama instance."""
        try:
            self.client.delete(model)
        except Exception as e:
            raise RuntimeError(f"Failed to delete model {model}: {e}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_description_tagger(
    host: str = DescriptionTagger.DEFAULT_HOST,
    model: str = DescriptionTagger.DEFAULT_MODEL,
    post_count_threshold: int = 500,
) -> DescriptionTagger:
    """Factory function to get a description tagger instance."""
    return DescriptionTagger(
        host=host,
        model=model,
        post_count_threshold=post_count_threshold,
    )
