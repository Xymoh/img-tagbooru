"""Vision-language captioning: image straight to a natural-language prompt.

The ONNX taggers in :mod:`backend.tagger` are Danbooru-trained, so on
photographs they emit a thin, often contradictory tag set (a real photo picks
up ``painting (medium)``, mutually exclusive garments, and the occasional
anime character). Booru tags also have no vocabulary for lighting, camera
framing or scene character, so nothing downstream can recover it.

This module takes a different route for images: a vision-language model
describes the picture directly in prose. One call, no tag bottleneck - the
model that sees the image is the same one that writes the sentence.

Runs through the same local Ollama instance as the description tagger, so
there is no new dependency and no data leaves the machine.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:  # pragma: no cover - mirrors description_tagger
    ollama = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# Known vision models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VLMModelInfo:
    """A vision model the app knows how to offer and size."""

    model_id: str
    name: str
    description: str
    approx_size_gb: float
    builtin: bool = True


# JoyCaption is purpose-built for captioning images to prompt and train
# diffusion models, and is uncensored by design (deliberate SFW/NSFW parity) -
# stock VLMs refuse or sanitise a large share of this app's subject matter.
# Quants below are the tags actually published for the GGUF conversion; note
# the repo has no "latest" tag, so the quant must always be named explicitly.
_JOYCAPTION = "aha2025/llama-joycaption-beta-one-hf-llava"

BUILTIN_VLM_MODELS: list[VLMModelInfo] = [
    VLMModelInfo(
        model_id=f"{_JOYCAPTION}:Q8_0",
        name="JoyCaption beta-one (Q8_0) ⭐",
        description=(
            "Best quality. Purpose-built for diffusion prompt captioning, "
            "uncensored. Fits a 16 GB card on its own."
        ),
        approx_size_gb=9.42,
    ),
    VLMModelInfo(
        model_id=f"{_JOYCAPTION}:Q6_K",
        name="JoyCaption beta-one (Q6_K)",
        description="Near-Q8 quality, a little smaller.",
        approx_size_gb=7.47,
    ),
    VLMModelInfo(
        model_id=f"{_JOYCAPTION}:Q5_K_M",
        name="JoyCaption beta-one (Q5_K_M)",
        description="Middle ground between size and quality.",
        approx_size_gb=6.61,
    ),
    VLMModelInfo(
        model_id=f"{_JOYCAPTION}:Q4_K_M",
        name="JoyCaption beta-one (Q4_K_M)",
        description=(
            "Smallest. Leaves room to keep a text LLM resident alongside it "
            "on a 16 GB card."
        ),
        approx_size_gb=5.80,
    ),
]


def list_vlm_models() -> list[VLMModelInfo]:
    """Return the built-in vision model registry."""
    return list(BUILTIN_VLM_MODELS)


# ---------------------------------------------------------------------------
# Caption styles
# ---------------------------------------------------------------------------

# JoyCaption is instruction-tuned around explicit caption-type prompts; these
# are the published phrasings. `{length}` is filled from CAPTION_LENGTHS.
CAPTION_STYLES: dict[str, tuple[str, str]] = {
    "descriptive": (
        "Descriptive (formal)",
        "Write a {length}descriptive caption for this image in a formal tone.",
    ),
    "descriptive_casual": (
        "Descriptive (casual)",
        "Write a {length}descriptive caption for this image in a casual tone.",
    ),
    "straightforward": (
        "Straightforward",
        "Write a {length}straightforward caption for this image. Begin with the "
        "main subject and medium. Describe what is present, including colors, "
        "composition, lighting and framing. Avoid speculation and flowery "
        "language.",
    ),
    "art_critic": (
        "Art critic",
        "Analyze this image like an art critic would, in {length}prose covering "
        "composition, lighting, color, mood and technique.",
    ),
    "sd_prompt": (
        "Stable Diffusion prompt",
        "Write a stable diffusion prompt for this image.",
    ),
}

# Default is the formal descriptive caption: flowing prose, which is what
# natural-language image models (Krea, Flux) actually condition on.
DEFAULT_STYLE = "descriptive"

CAPTION_LENGTHS: dict[str, str] = {
    "any": "",
    "short": "short ",
    "medium": "medium-length ",
    "long": "long ",
    "very_long": "very long ",
}
DEFAULT_LENGTH = "medium"


# ---------------------------------------------------------------------------
# Extra options
# ---------------------------------------------------------------------------

# JoyCaption steers register and coverage through instruction lines appended
# to the base prompt. The strings below are the project's own verbatim
# wording - paraphrasing them measurably weakens the effect, since the model
# was tuned on these exact sentences.
EXTRA_OPTIONS: dict[str, tuple[str, str]] = {
    "lighting": (
        "Include lighting",
        "Include information about lighting.",
    ),
    "camera_angle": (
        "Include camera angle",
        "Include information about camera angle.",
    ),
    "camera_gear": (
        "Include likely camera",
        "If it is a photo you MUST include information about what camera was "
        "likely used",
    ),
    "content_rating": (
        "State SFW / suggestive / NSFW",
        "Include whether the image is sfw, suggestive, or nsfw.",
    ),
    "no_unchangeable": (
        "Skip fixed personal traits",
        "Do NOT include information about people/characters that cannot be "
        "changed",
    ),
    "no_euphemisms": (
        "Blunt phrasing, no euphemisms",
        "Do NOT use polite euphemisms-lean into blunt, casual phrasing.",
    ),
    "vulgar": (
        "Vulgar slang and profanity",
        'Use vulgar slang and profanity, such as (but not limited to) '
        '"fucking," "slut," "cock," etc.',
    ),
    "keep_pg": (
        "Keep it PG (no sexual content)",
        "Do NOT include anything sexual; keep it PG.",
    ),
}

# What the Explicit toggle turns on. A formal register is what produces
# "modest cleavage" and similar euphemism on suggestive images - the model is
# not refusing, it is being polite - so explicit mode also drops the formal
# tone in favour of the casual variant.
#
# Deliberately NOT bundling "vulgar" here. It is tuned for training-caption
# realism rather than prompt quality: in testing it traded descriptive
# precision for slang, degrading "natural light from a window on the right"
# to "lighting's bright". Profanity is also an opinion, not a visual
# descriptor, so it does nothing for a diffusion model. It stays available as
# a separate opt-in for anyone captioning a training set.
EXPLICIT_EXTRAS: tuple[str, ...] = ("no_euphemisms",)

# Mutually exclusive with explicit mode.
_PG_OPTION = "keep_pg"

# Options that set the *register* of the whole caption, as opposed to asking
# for an extra detail. These must be emitted first: appended after the
# "include lighting / camera angle" lines the model has already settled on a
# tone, and measurably softens - the same image captioned with the blunt
# instructions last gave "SFW but fucking suggestive", and with them first gave
# "NSFW photograph ... medium-sized tits".
_REGISTER_OPTIONS: tuple[str, ...] = ("no_euphemisms", "vulgar", "keep_pg")

# JoyCaption's own inference sets this; the GGUF ships with SYSTEM=None, so
# without it the chat template injects an empty system message.
JOYCAPTION_SYSTEM = "You are a helpful image captioner."

# Explicit captions compete for a fixed word budget. Measured on a real NSFW
# image: with "include lighting" and "include camera angle" also requested, the
# model spent its budget on those and never mentioned the exposed chest, while
# the same image without them did. Rather than drop those options, tell the
# model what to cover first.
_EXPLICIT_PRIORITY = (
    "Describe the subject's body, pose and exact state of dress FIRST and in "
    "full detail, including anything exposed. Cover lighting, camera and "
    "background only afterwards, and only if words remain. This applies to "
    "adults only: if anyone in the image appears to be a minor, describe them "
    "neutrally and never in sexual terms."
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VLMCaptionResult:
    """A natural-language caption produced directly from an image.

    ``video_prompt`` is filled in only when the caller also asked for an
    image-to-video prompt; the caption is then the first-frame description it
    was written from, and remains useful on its own.
    """

    caption: str
    model: str
    style: str
    raw_response: str = ""
    word_count: int = 0
    video_prompt: str = ""
    video_style: str = ""
    video_duration: int = 0
    # MMAudio conditioning for the same clip, when requested.
    audio_prompt: str = ""
    audio_negative: str = ""

    def __post_init__(self) -> None:
        if not self.word_count:
            object.__setattr__(self, "word_count", len(self.caption.split()))

    @property
    def primary_text(self) -> str:
        """What the user actually asked for: video prompt if any, else caption."""
        return self.video_prompt or self.caption


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------


ImageInput = Union[str, Path, bytes, "Image.Image"]


class VLMCaptioner:
    """Caption images with a local vision-language model via Ollama."""

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = BUILTIN_VLM_MODELS[0].model_id

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    # A user's own instruction is folded in right after the base request and
    # before the option lines, because position drives register here: the same
    # steering text appended last is measurably weaker than the same text first.
    _MAX_CUSTOM_INSTRUCTION = 400

    @staticmethod
    def clean_custom_instruction(value: str) -> str:
        """Normalise a user-supplied steering instruction.

        Collapsed to one line and length-capped so a stray paste cannot
        overwhelm the actual caption request.
        """
        if not value:
            return ""
        text = " ".join(str(value).split())
        if len(text) > VLMCaptioner._MAX_CUSTOM_INSTRUCTION:
            text = text[: VLMCaptioner._MAX_CUSTOM_INSTRUCTION].rstrip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    @staticmethod
    def build_prompt(
        style: str = DEFAULT_STYLE,
        length: str = DEFAULT_LENGTH,
        extras: Optional[list[str]] = None,
        explicit: bool = False,
        custom_instruction: str = "",
    ) -> str:
        """Build the instruction for *style* at *length*.

        *extras* are keys from :data:`EXTRA_OPTIONS`, appended verbatim.
        *explicit* additionally forces blunt, non-euphemistic phrasing and
        swaps a formal descriptive tone for the casual one - formality, not
        refusal, is what turns explicit imagery into "modest cleavage".
        """
        if style not in CAPTION_STYLES:
            style = DEFAULT_STYLE

        selected = list(extras or [])
        if explicit:
            # A formal register euphemises; casual + blunt does not.
            if style == "descriptive":
                style = "descriptive_casual"
            for key in EXPLICIT_EXTRAS:
                if key not in selected:
                    selected.append(key)
            # "Keep it PG" directly contradicts explicit mode.
            selected = [k for k in selected if k != _PG_OPTION]

        _, template = CAPTION_STYLES[style]
        length_token = CAPTION_LENGTHS.get(length, CAPTION_LENGTHS[DEFAULT_LENGTH])
        # Styles without a {length} slot (sd_prompt) ignore it harmlessly.
        prompt = template.format(length=length_token)

        custom = VLMCaptioner.clean_custom_instruction(custom_instruction)
        if custom:
            prompt = f"{prompt} {custom}"

        # Register-setting instructions first, detail requests after - see
        # _REGISTER_OPTIONS for why the order is load-bearing.
        register = [k for k in selected if k in _REGISTER_OPTIONS]
        details = [
            k for k in selected if k in EXTRA_OPTIONS and k not in _REGISTER_OPTIONS
        ]
        lines = [EXTRA_OPTIONS[k][1] for k in register + details]
        if explicit:
            # Placed after the register rules but before the detail requests,
            # so "describe the body first" outranks "include lighting".
            lines.insert(len(register), _EXPLICIT_PRIORITY)
        if lines:
            prompt = prompt + " " + " ".join(lines)
        return prompt

    # ------------------------------------------------------------------
    # Image normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_image_payload(image: ImageInput) -> Union[str, bytes]:
        """Normalise supported image inputs to what the ollama client accepts.

        Paths are passed through as strings; in-memory PIL images (pasted or
        dropped, where ``TaggingResult.path`` is None) are encoded to PNG.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return str(path)
        if isinstance(image, bytes):
            return image
        if Image is not None and isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        raise TypeError(f"Unsupported image input: {type(image).__name__}")

    # ------------------------------------------------------------------
    # Output cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_caption(text: str) -> str:
        """Scrub a VLM caption to plain prose.

        Reuses the description tagger's prose scrubber, which already strips
        reasoning blocks, markdown, "Here is the caption:" labels, attention
        weights and trailing commentary - every failure mode is shared.
        """
        from backend.description_tagger import DescriptionTagger

        return DescriptionTagger._clean_prose_output(text)

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def caption_image(
        self,
        image: ImageInput,
        style: str = DEFAULT_STYLE,
        length: str = DEFAULT_LENGTH,
        extras: Optional[list[str]] = None,
        explicit: bool = False,
        custom_instruction: str = "",
        temperature: float = 0.6,
        max_tokens: int = 400,
    ) -> VLMCaptionResult:
        """Describe *image* in natural language.

        Unlike the description tagger this is a single call - the model that
        sees the image writes the sentence, so there is no tag vocabulary in
        between to lose lighting, framing or scene detail.
        """
        if style not in CAPTION_STYLES:
            style = DEFAULT_STYLE
        if length not in CAPTION_LENGTHS:
            length = DEFAULT_LENGTH

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        payload = self._to_image_payload(image)
        prompt = self.build_prompt(
            style,
            length,
            extras=extras,
            explicit=explicit,
            custom_instruction=custom_instruction,
        )

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                # The GGUF ships without a system prompt, so the chat template
                # injects an empty one. Supply JoyCaption's own.
                system=JOYCAPTION_SYSTEM,
                images=[payload],
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                },
                stream=False,
            )
        except Exception as e:
            message = str(e)
            if "not found" in message.lower():
                raise RuntimeError(
                    f"Model '{self.model}' is not installed. Pull it first:\n"
                    f"  ollama pull {self.model}\n\n"
                    "Note: this repo has no 'latest' tag - the quant must be "
                    "named explicitly."
                ) from e
            raise RuntimeError(f"Captioning failed: {e}") from e

        raw = response.get("response", "").strip()
        caption = self._clean_caption(raw)

        if not caption:
            raise RuntimeError(
                "The model returned an empty caption. Try a different caption "
                "style, or check that the selected model supports vision."
            )

        return VLMCaptionResult(
            caption=caption,
            model=self.model,
            style=style,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Ollama connection & model management
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        import urllib.request

        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def list_installed_models(self) -> list[str]:
        """Names of every model installed in the local Ollama instance."""
        try:
            resp = self.client.list()
            return [m.model for m in resp.models if getattr(m, "model", None)]
        except Exception:
            return []

    def is_installed(self, model: Optional[str] = None) -> bool:
        """Whether *model* (default: the active one) is pulled locally."""
        target = model or self.model
        return target in self.list_installed_models()

    # Ollama reports per-model capabilities; "vision" is the one that decides
    # whether a model can take an image at all. Cached because /api/show is a
    # round trip per model and the dialog asks about every installed one.
    _vision_cache: dict[str, bool] = {}

    def has_vision(self, model: Optional[str] = None) -> bool:
        """Whether Ollama reports *model* as vision-capable.

        A text-only model has no image input path, so captioning with one fails
        in a confusing way - this lets the UI filter them out up front.
        """
        import json
        import urllib.request

        target = model or self.model
        if target in VLMCaptioner._vision_cache:
            return VLMCaptioner._vision_cache[target]

        capable = False
        try:
            request = urllib.request.Request(
                f"{self.host}/api/show",
                data=json.dumps({"model": target}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=20) as resp:
                info = json.load(resp)
            capable = "vision" in (info.get("capabilities") or [])
        except Exception:
            # Unknown rather than false: never hide a model because the probe
            # failed. Curated entries are trusted separately.
            logger.debug("Could not read capabilities for %s", target)
            return False

        VLMCaptioner._vision_cache[target] = capable
        return capable

    def list_vision_models(self) -> list[str]:
        """Installed models Ollama reports as vision-capable.

        Lets the picker offer anything the user has pulled - a newer or more
        accurate captioner shows up without a code change.
        """
        return [m for m in self.list_installed_models() if self.has_vision(m)]

    def pull_model(self, model: str) -> None:
        """Download a vision model from the Ollama registry."""
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


def get_vlm_captioner(
    host: str = VLMCaptioner.DEFAULT_HOST,
    model: str = VLMCaptioner.DEFAULT_MODEL,
) -> VLMCaptioner:
    """Factory function to get a VLM captioner instance."""
    return VLMCaptioner(host=host, model=model)
