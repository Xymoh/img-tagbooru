"""Tag frequency index backed by danbooru_tags_post_count.csv.

Provides fast O(1) tag validation, post_count lookups, and top-N queries
grouped by semantic category for prompt-building.
"""

from __future__ import annotations

import csv
import logging
import os
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category keyword sets used by top_by_category()
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "character": [
        "1girl", "1boy",
        "solo", "solo_focus", "no_humans",
        "male_focus", "female_focus",
        "couple", "trap", "futanari",
        "mature_female", "mature_male",
    ],
    "clothing": [
        "dress", "shirt", "skirt", "coat", "jacket", "hat",
        "gloves", "shoes", "boots", "pants", "shorts",
        "sweater", "hoodie", "uniform", "swimsuit", "bikini",
        "thighhighs", "stockings", "pantyhose", "socks",
        "underwear", "panties", "bra", "lingerie",
        "ribbon", "bow", "necktie", "bowtie", "choker",
        "collar", "belt", "scarf", "cape", "cloak",
        "kimono", "robe", "apron", "leotard", "bodysuit",
        "sleeveless", "long_sleeves", "short_sleeves",
        "sailor", "serafuku", "school_uniform",
        "jewelry", "earrings", "necklace", "bracelet",
    ],
    "setting": [
        "indoors", "outdoors", "bedroom", "classroom", "office",
        "beach", "forest", "city", "street", "alley",
        "sky", "night", "day", "sunset", "moonlight",
        "water", "ocean", "pool", "river",
        "room", "window", "door", "wall", "floor",
        "chair", "bed", "table", "desk", "couch",
        "building", "church", "temple", "school",
        "park", "garden", "field", "mountain",
        "rain", "snow", "fog", "cloud",
        "simple_background", "white_background", "grey_background",
        "gradient_background",
    ],
    "action": [
        "standing", "sitting", "lying", "walking", "running",
        "dancing", "jumping", "flying", "floating",
        "holding", "carrying", "reaching", "pointing",
        "looking_at_viewer", "looking_away", "looking_back",
        "looking_down", "looking_up",
        "smile", "blush", "crying", "angry", "surprised",
        "open_mouth", "closed_mouth", "tongue_out",
        "kneeling", "bent_over", "squatting",
        "hands_up", "arms_up", "arms_behind_back",
        "crossed_arms", "hands_on_hips",
        "eating", "drinking", "reading", "sleeping",
        "hug", "kiss", "holding_hands",
    ],
    "style": [
        "highres", "absurdres", "detailed", "masterpiece",
        "best_quality", "high_quality", "ultra_detailed",
        "realistic", "photorealistic", "semi-realistic",
        "sketch", "watercolor", "oil_painting",
        "monochrome", "greyscale", "sepia",
        "flat_color", "cel_shading", "anime_coloring",
        "depth_of_field", "motion_blur", "lens_flare",
        "backlighting", "dutch_angle", "close-up",
        "cowboy_shot", "full_body", "upper_body", "portrait",
        "from_behind", "from_side", "from_above", "from_below",
        "pov", "selfie",
    ],
    "supernatural": [
        "zombie", "monster", "undead", "monster_girl", "demon", "ghost",
        "spirit", "vampire", "werewolf", "skeleton", "alien", "creature",
        "blood", "gore", "horror", "scary", "spooky", "nightmare",
        "angel", "devil", "reaper", "shinigami", "yokai", "oni",
        "tentacles", "eldritch", "abomination", "mutant", "infected",
        "chainsaw", "weapon", "guitar", "fighting", "battle",
    ],
    "explicit": [
        "sex", "fellatio", "cunnilingus", "anal",
        "penis", "erection", "pussy", "clitoris",
        "breasts", "nipples", "ass", "nude", "completely_nude",
        "spread_legs", "spread_pussy", "vaginal",
        "doggystyle", "missionary", "cowgirl_position",
        "masturbation", "fingering", "handjob", "footjob",
        "ahegao", "moaning", "panting", "tears",
        "saliva", "drooling", "gagging",
        "cum", "cumshot", "creampie", "facial",
        "rape", "gangbang", "group_sex", "bondage",
        "tentacles", "bdsm",
    ],
}

# ---------------------------------------------------------------------------
# TagFrequencyIndex
# ---------------------------------------------------------------------------


class TagFrequencyIndex:
    """Efficient lookup and ranking over Danbooru tag frequencies."""

    __slots__ = ("_counts", "_sorted_tags", "_min_threshold")

    def __init__(self, csv_path: str, min_threshold: int = 500) -> None:
        self._counts: dict[str, int] = {}
        self._sorted_tags: list[tuple[str, int]] = []
        self._min_threshold = min_threshold
        self._load(csv_path)

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> None:
        """Parse the CSV and build the internal lookup structures."""
        if not os.path.exists(csv_path):
            logger.warning("Tag frequency CSV not found at %s — index will be empty", csv_path)
            return

        counts: dict[str, int] = {}
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("name") or "").strip().lower()
                    if not name:
                        continue
                    try:
                        pc = int((row.get("post_count") or "0").strip())
                    except ValueError:
                        pc = 0
                    counts[name] = pc
        except Exception as exc:
            logger.exception("Failed to load tag frequency CSV: %s", exc)
            return

        self._counts = counts
        self._sorted_tags = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        logger.info(
            "TagFrequencyIndex loaded %d tags (top: %s = %s)",
            len(counts),
            self._sorted_tags[0][0] if self._sorted_tags else "none",
            f"{self._sorted_tags[0][1]:,}" if self._sorted_tags else "",
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_count(self, tag: str) -> int | None:
        """Return the post_count for *tag*, or ``None`` if unknown."""
        return self._counts.get(tag.lower().strip())

    def is_valid(self, tag: str) -> bool:
        """Return ``True`` if *tag* exists in the Danbooru vocabulary."""
        return tag.lower().strip() in self._counts

    def __contains__(self, tag: str) -> bool:
        return tag.lower().strip() in self._counts

    # ------------------------------------------------------------------
    # Ranking / threshold
    # ------------------------------------------------------------------

    @property
    def min_threshold(self) -> int:
        return self._min_threshold

    def set_min_threshold(self, value: int) -> None:
        self._min_threshold = max(0, value)

    def above_threshold(self, tag: str) -> bool:
        """``True`` if the tag's post_count >= min_threshold."""
        count = self.get_count(tag)
        return count is not None and count >= self._min_threshold

    def top_n(self, n: int = 50, exclude: set[str] | None = None) -> list[str]:
        """Return the top-*n* tags by post_count, skipping *exclude*."""
        skip = exclude or set()
        result: list[str] = []
        for tag, _count in self._sorted_tags:
            if tag in skip:
                continue
            result.append(tag)
            if len(result) >= n:
                break
        return result

    def top_by_category(
        self,
        n: int = 25,
        categories: Sequence[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, list[str]]:
        """Return ``{category_name: [top_n_tags]}`` for each requested category.

        Categories are defined by keyword matching against ``_CATEGORY_KEYWORDS``.
        If *categories* is ``None``, all categories are returned.
        """
        cats = list(categories) if categories else list(_CATEGORY_KEYWORDS.keys())
        skip = exclude or set()
        result: dict[str, list[str]] = {}

        for cat in cats:
            keywords = _CATEGORY_KEYWORDS.get(cat, [])
            if not keywords:
                result[cat] = []
                continue
            matched: list[str] = []
            for tag, _count in self._sorted_tags:
                if tag in skip:
                    continue
                if any(kw in tag.split("_") for kw in keywords):
                    matched.append(tag)
                    if len(matched) >= n:
                        break
            result[cat] = matched
        return result

    def search_by_keywords(
        self,
        keywords: Sequence[str],
        n: int = 5,
        min_count: int = 100,
        exclude: set[str] | None = None,
        allowed_parentheticals: set[str] | None = None,
    ) -> list[str]:
        """Return up to *n* tags per keyword whose name contains that keyword.

        Tags are ranked by post_count descending. Only tags with post_count
        >= *min_count* are considered.  Useful for building prompt-specific
        vocabulary sections.

        Tags with a Danbooru disambiguator (``tag_(context)``) are excluded
        unless the parenthetical content is in *allowed_parentheticals*.
        This prevents franchise-name collisions like ``knight_(hollow_knight)``
        surfacing when the user simply writes "knight".

        Returns a deduplicated list of all matching tags (not grouped by keyword).
        """
        skip = exclude or set()
        allowed_ctx = {p.lower() for p in (allowed_parentheticals or set())}
        seen: set[str] = set()
        result: list[str] = []

        # Track how many matches per keyword we've collected
        keyword_counts: dict[str, int] = {kw.lower(): 0 for kw in keywords}

        # Walk sorted tags once — O(N) single pass
        for tag, count in self._sorted_tags:
            if count < min_count:
                break  # rest are below threshold
            if tag in skip or tag in seen:
                continue
            # Skip disambiguator tags unless the context is allowed.
            if "_(" in tag and tag.endswith(")"):
                ctx = tag[tag.index("_(") + 2 : -1]
                if ctx not in allowed_ctx:
                    continue
            for kw in keywords:
                kw_lower = kw.lower()
                if keyword_counts.get(kw_lower, n) >= n:
                    continue
                # Word-boundary match: keyword must appear as a complete
                # underscore-delimited component of the tag (or vice versa).
                tag_parts = tag.split("_")
                if kw_lower in tag_parts or (len(tag) >= 4 and tag in kw_lower):
                    seen.add(tag)
                    result.append(tag)
                    keyword_counts[kw_lower] = keyword_counts.get(kw_lower, 0) + 1
                    break  # tag can only match one keyword-bucket
            if all(v >= n for v in keyword_counts.values()):
                break
        return result

    # ------------------------------------------------------------------
    # Tag filtering helper
    # ------------------------------------------------------------------

    def filter_by_threshold(self, tags: list[str]) -> list[str]:
        """Return *tags* with entries below ``min_threshold`` removed."""
        return [t for t in tags if self.above_threshold(t)]

    def sort_by_count(self, tags: list[str]) -> list[str]:
        """Sort *tags* by descending post_count, then alphabetically."""
        return sorted(
            tags,
            key=lambda t: (self.get_count(t) or 0),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._counts)

    def stats(self) -> dict:
        """Return summary statistics for debugging."""
        if not self._sorted_tags:
            return {"total": 0}
        total = len(self._counts)
        above = sum(1 for _, c in self._sorted_tags if c >= self._min_threshold)
        return {
            "total_tags": total,
            "above_threshold": above,
            "threshold": self._min_threshold,
            "top_10": [t for t, _ in self._sorted_tags[:10]],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_tag_index(csv_path: str | None = None, min_threshold: int = 500) -> TagFrequencyIndex:
    """Return a cached :class:`TagFrequencyIndex` instance.

    First call initializes the singleton with the given parameters.
    Subsequent calls with different arguments will return a different cached instance
    (lru_cache with maxsize=1 means only the most recent call is cached).
    """
    path = csv_path or os.path.join(
        os.path.dirname(__file__), "..", "danbooru_tags_post_count.csv",
    )
    return TagFrequencyIndex(os.path.abspath(path), min_threshold=min_threshold)
