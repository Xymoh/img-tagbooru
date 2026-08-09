from __future__ import annotations

import csv
import inspect
import io
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image


# Windowed builds (``pythonw`` / PyInstaller ``--windowed``) have no console, so
# ``sys.stderr`` is None. huggingface_hub draws a tqdm download progress bar to
# stderr, and writing to None crashes the download with
# "'NoneType' object has no attribute 'write'". We show our own progress UI, so
# disable the library's progress bars entirely.
try:  # pragma: no cover - depends on huggingface_hub version
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass


# The ONNX file name and tags file name are consistent across every WD tagger
# repo published by SmilingWolf, so they stay constant while the repo varies.
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"

# Default recognition model — kept as the historical name for backward compat
# (older code / configs may still reference ``MODEL_REPO``).
DEFAULT_MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_REPO = DEFAULT_MODEL_REPO


@dataclass(frozen=True)
class TaggerModelInfo:
    """Metadata for a selectable ONNX image-recognition model."""

    repo_id: str
    name: str
    description: str
    approx_size_mb: int
    recommended: bool = False
    builtin: bool = True  # False for user-added custom Hugging Face repos


# Curated registry of WD-style ONNX taggers (all from SmilingWolf on Hugging
# Face). Every entry exposes ``model.onnx`` + ``selected_tags.csv`` and shares
# the same BGR / square-pad preprocessing, so they are drop-in interchangeable.
MODEL_REGISTRY: list[TaggerModelInfo] = [
    TaggerModelInfo(
        repo_id="SmilingWolf/wd-swinv2-tagger-v3",
        name="WD SwinV2 v3",
        description=(
            "Balanced accuracy and speed. The proven default for anime/"
            "illustration tagging and LoRA dataset captioning."
        ),
        approx_size_mb=380,
        recommended=True,
    ),
    TaggerModelInfo(
        repo_id="SmilingWolf/wd-vit-tagger-v3",
        name="WD ViT v3",
        description=(
            "Vision Transformer variant. Slightly faster, comparable quality — "
            "a good lightweight alternative to SwinV2."
        ),
        approx_size_mb=380,
    ),
    TaggerModelInfo(
        repo_id="SmilingWolf/wd-convnext-tagger-v3",
        name="WD ConvNeXt v3",
        description=(
            "ConvNeXt architecture. Different inductive biases than the ViT/Swin "
            "models — useful for a second opinion on tricky images."
        ),
        approx_size_mb=400,
    ),
    TaggerModelInfo(
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        name="WD ViT-Large v3",
        description=(
            "Larger ViT with higher tag accuracy at the cost of speed and disk. "
            "Recommended when quality matters more than throughput."
        ),
        approx_size_mb=1200,
    ),
    TaggerModelInfo(
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        name="WD EVA02-Large v3",
        description=(
            "The most accurate model in the family. Largest download and slowest "
            "inference — best for careful, high-quality single-image tagging."
        ),
        approx_size_mb=1300,
    ),
]


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    confidence: float
    category: int


@dataclass(frozen=True)
class TagRecord:
    name: str
    category: int


def _cache_dir() -> Path:
    return Path.home() / ".img_tagger"


def _repo_cache_root(repo_id: str) -> Path:
    return _cache_dir() / repo_id.replace("/", "__")


def _download_model_file(
    filename: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    *,
    local_files_only: bool = False,
) -> Path:
    cache_root = _repo_cache_root(repo_id)
    cache_root.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_root),
            local_files_only=local_files_only,
        )
    )


# ---------------------------------------------------------------------------
# Config persistence (selected model + user-added custom repos)
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return _cache_dir() / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(config: dict) -> None:
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        _config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort persistence
        warnings.warn(f"Could not persist tagger config: {exc}", RuntimeWarning, stacklevel=2)


def get_selected_model() -> str:
    """Return the repo id of the recognition model the user last selected."""
    repo = _load_config().get("tagger_model")
    if isinstance(repo, str) and repo.strip():
        return repo
    return DEFAULT_MODEL_REPO


def set_selected_model(repo_id: str) -> None:
    config = _load_config()
    config["tagger_model"] = repo_id
    _save_config(config)


def _custom_models() -> list[TaggerModelInfo]:
    infos: list[TaggerModelInfo] = []
    for repo in _load_config().get("custom_tagger_models", []) or []:
        if not isinstance(repo, str) or not repo.strip():
            continue
        if any(m.repo_id == repo for m in MODEL_REGISTRY):
            continue
        infos.append(
            TaggerModelInfo(
                repo_id=repo,
                name=repo.split("/")[-1],
                description="Custom Hugging Face model added by you.",
                approx_size_mb=0,
                builtin=False,
            )
        )
    return infos


def _remember_custom_model(repo_id: str) -> None:
    if any(m.repo_id == repo_id for m in MODEL_REGISTRY):
        return
    config = _load_config()
    existing = list(config.get("custom_tagger_models", []) or [])
    if repo_id not in existing:
        existing.append(repo_id)
        config["custom_tagger_models"] = existing
        _save_config(config)


def _forget_custom_model(repo_id: str) -> None:
    config = _load_config()
    existing = list(config.get("custom_tagger_models", []) or [])
    if repo_id in existing:
        existing.remove(repo_id)
        config["custom_tagger_models"] = existing
        _save_config(config)


# ---------------------------------------------------------------------------
# Model registry / download / delete
# ---------------------------------------------------------------------------


def list_models() -> list[TaggerModelInfo]:
    """All selectable recognition models: built-in registry + custom repos."""
    return list(MODEL_REGISTRY) + _custom_models()


def find_model(repo_id: str) -> TaggerModelInfo | None:
    for info in list_models():
        if info.repo_id == repo_id:
            return info
    return None


def is_model_downloaded(repo_id: str) -> bool:
    """True if both the ONNX model and tags file are already in the local cache."""
    try:
        _download_model_file(MODEL_FILE, repo_id, local_files_only=True)
        _download_model_file(TAGS_FILE, repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def download_model(repo_id: str) -> None:
    """Fetch the ONNX model + tags for *repo_id* into the local cache.

    Blocking and potentially slow (hundreds of MB) — call from a worker thread.
    """
    _remember_custom_model(repo_id)
    _download_model_file(MODEL_FILE, repo_id)
    _download_model_file(TAGS_FILE, repo_id)


def delete_model_files(repo_id: str) -> None:
    """Remove a downloaded model's cached files to reclaim disk space."""
    cache_root = _repo_cache_root(repo_id)
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    _tagger_cache.pop(repo_id, None)
    _forget_custom_model(repo_id)


def _load_tags(csv_path: Path) -> list[TagRecord]:
    records: list[TagRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        peek = handle.readline()
        handle.seek(0)
        if "name" in peek.lower() and "category" in peek.lower():
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    category = int((row.get("category") or "0").strip())
                except ValueError:
                    category = 0
                records.append(TagRecord(name=name, category=category))
            return records

        reader = csv.reader(handle)
        rows = list(reader)
        for row in rows:
            if not row:
                continue
            if len(row) >= 3 and row[0].lower() == "name":
                continue
            name = row[0].strip()
            if not name:
                continue
            try:
                category = int(row[1]) if len(row) > 1 else 0
            except ValueError:
                category = 0
            records.append(TagRecord(name=name, category=category))
    return records


def mcut_threshold(probs: np.ndarray) -> float:
    if probs.size < 2:
        return float(probs.max()) if probs.size else 0.0
    sorted_probs = np.sort(probs)[::-1]
    diffs = sorted_probs[:-1] - sorted_probs[1:]
    cut_index = int(np.argmax(diffs))
    return float((sorted_probs[cut_index] + sorted_probs[cut_index + 1]) / 2.0)


def _make_square(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - resized.size[0]) // 2, (size - resized.size[1]) // 2)
    square.paste(resized, offset)
    return square


class AnimeTagger:
    def __init__(self, repo_id: str = DEFAULT_MODEL_REPO) -> None:
        self.repo_id = repo_id
        model_path = _download_model_file(MODEL_FILE, repo_id)
        tags_path = _download_model_file(TAGS_FILE, repo_id)
        self.tags = _load_tags(tags_path)
        self.session = self._create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _create_session(self, model_path: Path) -> ort.InferenceSession:
        providers = self._providers()
        try:
            return ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            if providers == ["CPUExecutionProvider"]:
                raise
            warnings.warn(
                f"Falling back to CPUExecutionProvider because CUDA initialization failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @staticmethod
    def _providers() -> list[str]:
        available = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _target_size(self) -> int:
        shape = self.input_shape
        candidates = [shape[-2], shape[-1]] if len(shape) >= 4 else []
        for candidate in candidates:
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return 448

    def _prepare_image(self, image: Image.Image, normalize_pixels: bool) -> np.ndarray:
        size = self._target_size()
        square = _make_square(image.convert("RGB"), size)
        array = np.asarray(square, dtype=np.float32)
        array = array[:, :, ::-1]
        if normalize_pixels:
            array = array / 255.0
        array = np.expand_dims(array, axis=0)
        return array

    def predict(
        self,
        image: Image.Image,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        normalize_pixels: bool = False,
        use_mcut: bool = False,
        limit: int | None = None,
    ) -> list[TagPrediction]:
        inputs = self._prepare_image(image, normalize_pixels=normalize_pixels)
        raw = self.session.run([self.output_name], {self.input_name: inputs})[0]
        scores = 1.0 / (1.0 + np.exp(-raw[0]))

        predictions: list[TagPrediction] = []
        general_candidates: list[TagPrediction] = []
        character_candidates: list[TagPrediction] = []

        for index, score in enumerate(scores[: len(self.tags)]):
            record = self.tags[index]
            tag_name = record.name.replace("_", " ")
            prediction = TagPrediction(tag=tag_name, confidence=float(score), category=record.category)
            if record.category == 9:
                continue
            if record.category == 4:
                character_candidates.append(prediction)
                continue
            general_candidates.append(prediction)

        if use_mcut:
            general_values = np.array([prediction.confidence for prediction in general_candidates], dtype=np.float32)
            if general_values.size:
                general_threshold = max(general_threshold, mcut_threshold(general_values))
            character_values = np.array([prediction.confidence for prediction in character_candidates], dtype=np.float32)
            if character_values.size:
                character_threshold = max(character_threshold, mcut_threshold(character_values))

        for prediction in general_candidates:
            if prediction.confidence >= general_threshold:
                predictions.append(prediction)

        for prediction in character_candidates:
            if prediction.confidence >= character_threshold:
                predictions.append(prediction)

        predictions.sort(key=lambda item: item.confidence, reverse=True)
        if limit is not None:
            predictions = predictions[:limit]
        return predictions


def predict_tags(
    tagger: AnimeTagger,
    image: Image.Image,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    normalize_pixels: bool = False,
    use_mcut: bool = False,
    limit: int | None = None,
) -> list[TagPrediction]:
    parameters = inspect.signature(tagger.predict).parameters
    kwargs: dict[str, object] = {}

    if "general_threshold" in parameters:
        kwargs["general_threshold"] = general_threshold
    elif "threshold" in parameters:
        kwargs["threshold"] = general_threshold

    if "character_threshold" in parameters:
        kwargs["character_threshold"] = character_threshold

    if "normalize_pixels" in parameters:
        kwargs["normalize_pixels"] = normalize_pixels

    if "use_mcut" in parameters:
        kwargs["use_mcut"] = use_mcut

    if "limit" in parameters:
        kwargs["limit"] = limit

    return tagger.predict(image, **kwargs)


# Only the active model is kept resident — each ONNX session is large (RAM/VRAM),
# so switching models evicts the previous one rather than accumulating them.
_tagger_cache: dict[str, AnimeTagger] = {}


def get_tagger(repo_id: str | None = None) -> AnimeTagger:
    """Return the tagger for *repo_id* (defaults to the user's selected model)."""
    if repo_id is None:
        repo_id = get_selected_model()
    tagger = _tagger_cache.get(repo_id)
    if tagger is None:
        _tagger_cache.clear()
        tagger = AnimeTagger(repo_id)
        _tagger_cache[repo_id] = tagger
    return tagger


def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def category_label(category: int) -> str:
    return {
        0: "general",
        4: "character",
        9: "rating",
    }.get(category, f"category_{category}")


def caption_from_predictions(
    predictions: Iterable[TagPrediction],
    blacklist: Sequence[str] | None = None,
    whitelist: Sequence[str] | None = None,
    include_scores: bool = False,
) -> str:
    blacklist_set = {tag.strip().lower() for tag in (blacklist or []) if tag.strip()}
    whitelist_set = {tag.strip().lower() for tag in (whitelist or []) if tag.strip()}
    parts: list[str] = []
    for prediction in predictions:
        normalized = prediction.tag.lower().strip()
        if blacklist_set and normalized in blacklist_set:
            continue
        if whitelist_set and normalized not in whitelist_set:
            continue
        if include_scores:
            parts.append(f"{prediction.tag}:{prediction.confidence:.3f}")
        else:
            parts.append(prediction.tag)
    return ", ".join(parts)
