from __future__ import annotations

import io
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image

from backend.tagger import TagPrediction, category_label


IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Standard DataFrame columns for tag tables
TAG_COLUMNS = ["include", "rank", "tag", "confidence", "category"]


@dataclass
class TaggingResult:
    """Unified result item used by the desktop frontend."""

    name: str  # display filename
    path: Path | None  # filesystem path (None for in-memory uploads)
    image: Image.Image
    frame: pd.DataFrame
    caption: str

    @property
    def caption_filename(self) -> str:
        """Derived .txt filename for this result."""
        stem = Path(self.name).stem
        return f"{stem}.txt"


# ---------------------------------------------------------------------------
# Tag text / frame helpers
# ---------------------------------------------------------------------------


def split_tags(text: str) -> list[str]:
    """Split comma/newline-separated tag text into a clean list."""
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def frame_from_predictions(predictions: Sequence[TagPrediction]) -> pd.DataFrame:
    """Build a DataFrame from raw :class:`TagPrediction` objects."""
    rows: list[dict] = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "include": True,
                "rank": index + 1,
                "tag": prediction.tag,
                "confidence": round(prediction.confidence, 4),
                "category": category_label(prediction.category),
            }
        )
    if not rows:
        return pd.DataFrame(columns=TAG_COLUMNS)
    return pd.DataFrame(rows)


def apply_filters(frame: pd.DataFrame, blacklist: Iterable[str], whitelist: Iterable[str]) -> pd.DataFrame:
    """Mark tags for exclusion based on blacklist / whitelist.

    Returns a copy of *frame* with the ``include`` column adjusted.

    Each blacklist/whitelist entry may be a literal tag name or a regex
    pattern wrapped in ``/slashes/`` (e.g. ``/^bad_/`` or ``/.*_old$/``).
    """
    blacklist_literal: set[str] = set()
    blacklist_regex: list[re.Pattern] = []
    for raw in blacklist:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("/") and raw.endswith("/") and len(raw) > 2:
            try:
                blacklist_regex.append(re.compile(raw[1:-1], re.IGNORECASE))
            except re.error:
                blacklist_literal.add(raw.lower())
        else:
            blacklist_literal.add(raw.lower())

    whitelist_literal: set[str] = set()
    whitelist_regex: list[re.Pattern] = []
    for raw in whitelist:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("/") and raw.endswith("/") and len(raw) > 2:
            try:
                whitelist_regex.append(re.compile(raw[1:-1], re.IGNORECASE))
            except re.error:
                whitelist_literal.add(raw.lower())
        else:
            whitelist_literal.add(raw.lower())

    def keep_row(tag: str) -> bool:
        normalized = str(tag).strip().lower()
        if normalized in blacklist_literal:
            return False
        for pattern in blacklist_regex:
            if pattern.search(normalized):
                return False
        if whitelist_literal:
            if normalized not in whitelist_literal:
                return False
        elif whitelist_regex:
            if not any(p.search(normalized) for p in whitelist_regex):
                return False
        return True

    result = frame.copy()
    result["include"] = result["include"] & result["tag"].map(keep_row)
    return result


def sort_frame(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    """Sort a tag DataFrame *in place copy* by the given mode.

    *sort_mode* may be ``"alphabetical"``, ``"confidence"``, or anything else
    (defaults to manual rank ordering).
    """
    if frame.empty:
        return frame.copy()
    if sort_mode == "alphabetical":
        return frame.sort_values(by=["tag", "confidence"], ascending=[True, False]).reset_index(drop=True)
    if sort_mode == "confidence":
        return frame.sort_values(by=["confidence", "tag"], ascending=[False, True]).reset_index(drop=True)
    return frame.sort_values(by=["rank", "confidence"], ascending=[True, False]).reset_index(drop=True)


def frame_to_caption(frame: pd.DataFrame, include_scores: bool = False) -> str:
    """Convert a tag DataFrame to a comma-separated caption string."""
    included = frame[frame["include"]].copy()
    if included.empty:
        return ""
    included = included.sort_values(by=["rank", "confidence"], ascending=[True, False])
    if include_scores:
        return ", ".join(f"{row.tag}:{row.confidence:.3f}" for row in included.itertuples())
    return ", ".join(included["tag"].tolist())


def caption_from_frame(frame: pd.DataFrame, include_scores: bool = False) -> str:
    """Alias for :func:`frame_to_caption`."""
    return frame_to_caption(frame, include_scores=include_scores)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_zip_from_results(results: Iterable[TaggingResult]) -> bytes:
    """Create a ZIP of ``caption_filename → caption`` for every result."""
    return export_zip((r.caption_filename, r.caption) for r in results)


def export_zip(caption_pairs: Iterable[tuple[str, str]]) -> bytes:
    """Create a ZIP archive of ``(filename, content)`` caption pairs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in caption_pairs:
            archive.writestr(filename, content.encode("utf-8"))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# AI Generation Metadata extraction
# ---------------------------------------------------------------------------

_SD_PARAM_RE = re.compile(
    r'(?:^|\n)([A-Za-z_][A-Za-z0-9_]*(?:\s*:\s*.+))',
    re.MULTILINE,
)


def extract_ai_metadata(image: Image.Image) -> dict[str, str]:
    """Extract AI generation parameters embedded in *image* metadata.

    Reads PNG ``tEXt``/``iTXt`` chunks and JPEG EXIF ``UserComment``,
    looking for Stable-Diffusion-style key-value parameters or raw JSON
    (ComfyUI / A1111 / NovelAI).

    Returns a dict mapping parameter names to their string values.
    An empty dict means no AI metadata was found.
    """
    raw: dict[str, str] = {}
    info = getattr(image, "info", {}) or {}

    # 1) PNG textual chunks
    for key, value in info.items():
        if isinstance(value, str):
            raw[key] = value
        elif isinstance(value, bytes):
            try:
                raw[key] = value.decode("utf-8", errors="replace")
            except Exception:
                pass

    # 2) JPEG EXIF UserComment (EXIF tag 0x9286: UserComment)
    exif_data = image.getexif() if hasattr(image, "getexif") else None
    if exif_data:
        try:
            # Py 3.8+ or Pillow: get_exif() returns a dictionary-like object.
            # UserComment may be bytes; decode it.
            user_comment = exif_data.get(0x9286, None)  # 0x9286 = UserComment
            if user_comment:
                if isinstance(user_comment, bytes):
                    # First 8 bytes in EXIF UserComment are the character code
                    if len(user_comment) > 8:
                        user_comment = user_comment[8:]
                    raw["UserComment"] = user_comment.decode("utf-8", errors="replace")
                elif isinstance(user_comment, str):
                    raw["UserComment"] = user_comment
        except Exception:
            pass

    if not raw:
        return {}

    # ---- Parse out generation parameters ----
    params: dict[str, str] = {}

    for raw_value in raw.values():
        # Try JSON first (ComfyUI workflow blocks)
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                # Flatten top-level string values
                for k, v in parsed.items():
                    if isinstance(v, (str, int, float, bool)):
                        params[str(k)] = str(v)
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Try SD parameter syntax:
        # A1111 / AUTOMATIC1111 format:
        #   <positive prompt (no label, can span lines)>
        #   Negative prompt: <negative prompt text>
        #   Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, ...
        # Some tools use "Positive prompt:" prefix; handle both.
        lines = raw_value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        has_structured = False

        # --- Detect the Negative prompt boundary ---
        neg_idx: int | None = None
        for i, line in enumerate(lines):
            if re.match(r'^Negative prompt\s*:', line, re.IGNORECASE):
                neg_idx = i
                break

        if neg_idx is not None:
            # Everything before Negative prompt = positive prompt
            positive_text = "\n".join(lines[:neg_idx]).strip()
            if positive_text:
                params["Positive prompt"] = positive_text
                has_structured = True

            # Negative prompt content
            neg_content = lines[neg_idx].partition(":")[2].strip()
            if neg_content:
                params["Negative prompt"] = neg_content

            # Lines after the negative prompt may contain key:value pairs
            for line in lines[neg_idx + 1:]:
                line = line.strip()
                if not line or line.startswith("{"):
                    continue
                if "," in line and ":" in line:
                    for chunk in re.split(r",\s*", line):
                        chunk = chunk.strip()
                        if ":" in chunk:
                            key, _, val = chunk.partition(":")
                            key = key.strip()
                            val = val.strip()
                            if key and val:
                                params[key] = val
                                has_structured = True
                elif ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key and val:
                        params[key] = val
                        has_structured = True
        else:
            # No Negative prompt boundary - try generic line-by-line
            for line in lines:
                line = line.strip()
                if not line or line.startswith("{"):
                    continue
                if "," in line and ":" in line:
                    for chunk in re.split(r",\s*", line):
                        chunk = chunk.strip()
                        if ":" in chunk:
                            key, _, val = chunk.partition(":")
                            key = key.strip()
                            val = val.strip()
                            if key and val:
                                params[key] = val
                                has_structured = True
                elif ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key and val:
                        params[key] = val
                        has_structured = True

        # If no structured params found, capture the raw block as is
        if not has_structured and raw_value.strip():
            params["raw_metadata"] = raw_value.strip()

    return params


def metadata_to_tags(params: dict[str, str]) -> list[str]:
    """Convert AI generation parameters into Danbooru-style tags.

    Only the **positive prompt** text is tokenized into tags (split on
    commas, each token stripped and lowercased with underscores).
    All other generation parameters are ignored.
    """
    # Look for the positive prompt under various possible keys
    prompt_keys = (
        "Positive prompt", "positive prompt", "Prompt", "prompt",
        "Positive", "positive",
    )
    prompt_text = ""
    for key in prompt_keys:
        if key in params:
            prompt_text = params[key]
            break

    # Fallback: if no explicit positive prompt is found, check if
    # the raw text contains "Positive prompt:" and extract it.
    if not prompt_text:
        raw = params.get("raw_metadata", "")
        for raw_val in [raw] + [v for k, v in params.items() if k != "raw_metadata"]:
            match = re.search(
                r"(?:^|\n)(?:Positive prompt|positive prompt|Prompt)\s*:\s*(.+?)(?:\n(?:Negative prompt|Steps|$)|$)",
                raw_val,
                re.DOTALL,
            )
            if match:
                prompt_text = match.group(1).strip()
                break

    if not prompt_text:
        return []

    # Normalise: collapse newlines → commas (some prompts span lines)
    prompt_text = prompt_text.replace("\n", ", ")

    # Tokenize: split by comma, clean each token
    tags: list[str] = []
    for token in prompt_text.split(","):
        token = token.strip()
        if not token:
            continue
        # Replace spaces and dashes with underscores, lowercase
        tag = token.replace(" ", "_").replace("-", "_").lower()
        # Remove leading/trailing non-alphanumeric chars (except underscores)
        tag = tag.strip(".,;:!?()[]{}\"'")
        # Collapse multiple underscores
        tag = re.sub(r"_+", "_", tag).strip("_")
        if tag and len(tag) >= 2:
            tags.append(tag)

    return tags
