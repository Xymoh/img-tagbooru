"""Content policy shared by every generation path.

Img-Tagbooru deliberately does not filter adult content: the models decide
what they will describe, and the Safe / Creative / Mature modes exist so the
user can choose. The single exception is sexual content involving minors,
which is a criminal offence in the EU, the US and nearly everywhere else even
when the imagery is drawn or generated. That rule is enforced here, in the
app's own code, on every path: it is never left to the model.

Two enforcement points:

* :func:`strip_minor_terms` removes age-descriptor tags from any tag list the
  app produces (LLM tag generation, tag enrichment, the ONNX image tagger).
* :data:`ADULTS_ONLY_RULE` is appended to every prompt that permits explicit
  output, so the model is told what the post-processing already guarantees.
"""

from __future__ import annotations

from typing import Iterable

# Danbooru age-descriptor tags that identify a subject as a minor. These are
# never emitted in a mode that can produce suggestive or explicit output.
MINOR_TAGS: frozenset[str] = frozenset({
    "loli", "shota", "toddlercon", "child_on_child",
    "child", "children", "toddler", "baby", "infant", "newborn",
    "aged_down", "kindergarten_uniform", "randoseru",
})

# The subset that is sexual by definition on Danbooru. These are removed in
# every mode, including Safe and the image tagger, because there is no
# non-sexual reading of them.
SEXUAL_MINOR_TAGS: frozenset[str] = frozenset({
    "loli", "shota", "toddlercon", "child_on_child",
})

# Appended to every prompt that permits explicit output.
ADULTS_ONLY_RULE = (
    "Every person depicted or described is an adult (18 or older). Never "
    "describe, tag, imply or sexualise anyone as a child, minor, loli, shota "
    "or otherwise underage, regardless of what the input says."
)

# Modes in which suggestive or explicit output is possible. Safe mode is a
# hard SFW guarantee, so ordinary family-scene tags (child, baby) stay usable
# there; everywhere else the whole age-descriptor set is removed.
_EXPLICIT_CAPABLE_MODES = frozenset({"creative", "mature"})


def _normalise(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def blocked_minor_tags(mode: str | None = None) -> frozenset[str]:
    """Return the set of tags that must not appear in output for *mode*.

    ``mode`` is one of ``"safe"``, ``"creative"``, ``"mature"`` or ``None``
    (the image tagger, which describes an existing image rather than
    generating a scene and therefore only drops the inherently sexual tags).
    """
    if mode in _EXPLICIT_CAPABLE_MODES:
        return MINOR_TAGS
    return SEXUAL_MINOR_TAGS


def is_blocked_minor_tag(tag: str, mode: str | None = None) -> bool:
    return _normalise(tag) in blocked_minor_tags(mode)


def strip_minor_terms(tags: Iterable[str], mode: str | None = None) -> list[str]:
    """Drop minor age-descriptor tags from *tags*, preserving order."""
    blocked = blocked_minor_tags(mode)
    return [tag for tag in tags if _normalise(tag) not in blocked]
