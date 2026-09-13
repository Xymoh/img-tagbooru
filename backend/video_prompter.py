"""Turn a still-image description into an image-to-video prompt (Wan 2.2).

This is deliberately a *second* stage rather than something asked of the vision
model directly. JoyCaption is a captioning model — it describes what is, and
when asked to plan motion it mostly restates the still ("hair remains mostly in
place", "no significant changes"). Inventing plausible motion and holding an
exact output format is an instruction-following job, so it goes to the text LLM
that already powers the description tagger.

Stage 1 (vision) supplies an accurate description of the first frame; stage 2
(text) decides what happens over the next few seconds.
"""

from __future__ import annotations

import logging
import re

from backend.content_policy import ADULTS_ONLY_RULE
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

VIDEO_STYLES: dict[str, str] = {
    "timeline": "Wan 2.2 — second-by-second timeline",
    "flowing": "Wan 2.2 — flowing paragraph",
}
DEFAULT_VIDEO_STYLE = "timeline"

DEFAULT_DURATION = 5
MIN_DURATION = 2
# Not a Wan limit — a deliberate ceiling on how long a per-second timeline can
# get before it stops being reliable. Models start dropping beats before this,
# so the generator accepts a near-complete timeline after the first retry
# rather than burning attempts chasing the last one or two.
MAX_DURATION = 20

# Beyond this, warn that the model may under-deliver beats.
RELIABLE_DURATION = 15


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoPromptResult:
    """An image-to-video prompt built from a first-frame description."""

    prompt: str
    style: str
    duration: int
    source_caption: str = ""
    model: str = ""
    raw_response: str = ""
    # MMAudio conditioning for the same clip: what to hear, and what to avoid.
    audio_prompt: str = ""
    audio_negative: str = ""

    @property
    def beat_count(self) -> int:
        """Number of parsed timeline beats (0 for the flowing style)."""
        return len(VideoPrompter.parse_beats(self.prompt))


# ---------------------------------------------------------------------------
# Prompter
# ---------------------------------------------------------------------------


class VideoPrompter:
    """Write Wan 2.2 image-to-video prompts from a first-frame description."""

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "richardyoung/qwen3-14b-abliterated:latest"

    # Matches one "(At N seconds: ...)" beat, tolerating "second"/"seconds"
    # and a missing trailing paren on the final beat.
    _BEAT_RE = re.compile(
        r"\(\s*At\s+(\d+)\s+seconds?\s*:\s*(.*?)\s*(?:\)|$)",
        re.IGNORECASE | re.DOTALL,
    )

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
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _format_example(duration: int) -> str:
        """The literal output shape, spelled out for the model to copy."""
        return " ".join(
            f"(At {s} seconds: ...)" for s in range(min(duration, 5))
        ) + (" ..." if duration > 5 else "")

    @classmethod
    def build_system_prompt(
        cls,
        style: str,
        duration: int,
        explicit: bool = False,
        custom_instruction: str = "",
    ) -> str:
        """System prompt for *style* over *duration* seconds.

        With *explicit* set, the motion is described bluntly. Without it the
        second stage writes in its own polite register regardless of how the
        first-frame description was worded, which quietly sanitises the result.
        """
        duration = max(MIN_DURATION, min(MAX_DURATION, duration))

        explicit_rules = (
            f"""
EXPLICIT CONTENT:
- This is adult material. Describe sexual motion directly and specifically:
  what moves, on whom, how fast, and how the bodies are positioned.
- Use blunt anatomical and vulgar terms. Do NOT soften, fade out, imply, or
  substitute euphemism for what is happening.
- Never refuse, never abbreviate, never add a disclaimer.
- {ADULTS_ONLY_RULE}"""
            if explicit
            else ""
        )

        # The user's own steering, kept last so it reads as the final word on
        # style without displacing the format rules above it.
        from backend.vlm_captioner import VLMCaptioner

        custom = VLMCaptioner.clean_custom_instruction(custom_instruction)
        custom_rules = (
            f"""

NARRATION STYLE — the user asked for this specifically, follow it:
- {custom}
- Apply it to how you word the motion. It never overrides the output format
  or the one-second plausibility limit above."""
            if custom
            else ""
        )

        # Shared constraints. The "never restate the still" rule is the one
        # that matters most: without it the model burns every beat describing
        # clothing and hair that the first frame already fixes.
        common = f"""You write prompts for Wan 2.2, an image-to-video model.

The user gives you a description of a still image. That image is the FIRST
FRAME of a {duration}-second video. Your job is to invent what happens next.

RULES:
- Describe MOTION and CAMERA only. Clothing, hair colour, the room and the
  lighting are already fixed by the first frame — never restate them.
- Motion must be small and physically plausible. In one second a person can
  blink, breathe, tilt their head, let hair settle, shift their weight or lower
  a hand. They cannot cross a room, change clothes, or age.
- Never contradict the first frame.
- The camera may hold still, push in, pull back, tilt, or drift slowly. Say
  which.
- Never write "remains", "continues", "unchanged", "stays the same" or "no
  significant change". Every beat must contain real movement.
- No preamble, no explanation, no markdown, no reasoning.{explicit_rules}{custom_rules}"""

        if style == "flowing":
            return (
                common
                + f"""

OUTPUT: ONE flowing paragraph of 60-90 words describing how the shot unfolds
over the {duration} seconds. Name the camera behaviour explicitly. Output only
the paragraph."""
            )

        return (
            common
            + f"""

OUTPUT FORMAT — copy this shape exactly, on ONE line, and nothing else:
{cls._format_example(duration)}

- Write exactly {duration} beats, one per second, numbered 0 to {duration - 1}.
- Always write "seconds" (plural), even for 1.
- Keep each beat to one short sentence of movement plus the camera.
- Output only the parenthesised timeline."""
        )

    # ------------------------------------------------------------------
    # Output parsing / cleanup
    # ------------------------------------------------------------------

    @classmethod
    def parse_beats(cls, text: str) -> list[tuple[int, str]]:
        """Extract ``(second, description)`` pairs from a timeline string."""
        beats: list[tuple[int, str]] = []
        for match in cls._BEAT_RE.finditer(text or ""):
            try:
                second = int(match.group(1))
            except ValueError:
                continue
            body = " ".join(match.group(2).split())
            if body:
                beats.append((second, body))
        return beats

    @classmethod
    def clean_timeline(cls, text: str, duration: int) -> str:
        """Normalise timeline output to one line of well-formed beats.

        Rebuilding from parsed beats rather than scrubbing in place also
        discards any reasoning the model leaked around them — qwen3 emits a
        stray ``</think>`` or ``\\boxed{}`` often enough to matter.
        """
        beats = cls.parse_beats(text)
        if not beats:
            return ""

        # Keep the first occurrence of each second, in ascending order, so a
        # model that restarts its list does not produce duplicates.
        seen: dict[int, str] = {}
        for second, body in beats:
            if second not in seen and second < duration:
                seen[second] = body

        ordered = [
            f"(At {s} seconds: {cls._tidy_beat(seen[s])})"
            for s in sorted(seen)
        ]
        return " ".join(ordered)

    @staticmethod
    def _tidy_beat(body: str) -> str:
        """Tidy one beat's text: no stray quotes, single trailing period."""
        body = body.strip().strip('"“”')
        body = re.sub(r"\s+", " ", body)
        body = body.rstrip(" ,;:")
        if body and not body.endswith((".", "!", "?")):
            body += "."
        return body

    @staticmethod
    def clean_flowing(text: str) -> str:
        """Scrub a flowing video prompt using the shared prose scrubber."""
        from backend.description_tagger import DescriptionTagger

        cleaned = DescriptionTagger._clean_prose_output(text)
        # The prose scrubber leaves LaTeX-style artefacts qwen3 sometimes emits.
        cleaned = re.sub(r"\\boxed\{\s*\}", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        import urllib.request

        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def _no_think_prefix(self) -> str:
        model_lower = self.model.lower()
        if (
            "qwen3" in model_lower
            and "instruct-2507" not in model_lower
            and "instruct_2507" not in model_lower
        ):
            return "/no_think"
        return ""

    def build_from_caption(
        self,
        caption: str,
        style: str = DEFAULT_VIDEO_STYLE,
        duration: int = DEFAULT_DURATION,
        explicit: bool = False,
        custom_instruction: str = "",
        max_attempts: int = 3,
    ) -> VideoPromptResult:
        """Write a video prompt from a first-frame *caption*.

        Retries when the timeline comes back with too few beats — a model that
        stops early is the common failure, and a short timeline is worse than
        a re-roll.
        """
        if not caption or not caption.strip():
            raise ValueError("Caption cannot be empty")

        if style not in VIDEO_STYLES:
            style = DEFAULT_VIDEO_STYLE
        duration = max(MIN_DURATION, min(MAX_DURATION, duration))

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        system = self.build_system_prompt(
            style, duration, explicit=explicit,
            custom_instruction=custom_instruction,
        )
        prefix = self._no_think_prefix()
        user = f"FIRST FRAME:\n{caption.strip()}\n\nWrite the prompt."
        if prefix:
            user = f"{prefix}\n\n{user}"

        best = ""
        raw_final = ""
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=user,
                    system=system,
                    options={
                        "temperature": min(1.0, 0.75 + 0.1 * attempt),
                        "top_p": 0.92,
                        # A beat runs ~25-35 tokens; 70 each leaves headroom
                        # without letting a long clip run away.
                        "num_predict": (
                            min(6000, 200 + 70 * duration)
                            if style == "timeline"
                            else 400
                        ),
                        "stop": ["<think>"],
                    },
                    stream=False,
                )
                raw_final = response.get("response", "").strip()

                if style == "timeline":
                    cleaned = self.clean_timeline(raw_final, duration)
                    beats = len(self.parse_beats(cleaned))
                    if beats > len(self.parse_beats(best)):
                        best = cleaned
                    # A complete timeline is always accepted. After the first
                    # attempt a near-complete one is too: on long clips the
                    # model reliably drops the last beat or two, and three
                    # full retries cost far more than the missing second.
                    good_enough = beats >= duration or (
                        attempt > 0 and beats >= max(2, int(duration * 0.9))
                    )
                    if good_enough:
                        return VideoPromptResult(
                            prompt=cleaned,
                            style=style,
                            duration=duration,
                            source_caption=caption,
                            model=self.model,
                            raw_response=raw_final,
                        )
                else:
                    cleaned = self.clean_flowing(raw_final)
                    if len(cleaned.split()) > len(best.split()):
                        best = cleaned
                    if len(cleaned.split()) >= 35:
                        return VideoPromptResult(
                            prompt=cleaned,
                            style=style,
                            duration=duration,
                            source_caption=caption,
                            model=self.model,
                            raw_response=raw_final,
                        )
            except Exception as e:
                last_error = e
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Video prompt generation failed: {e}")

        if not best:
            if last_error is not None:
                raise RuntimeError(
                    f"Video prompt generation failed after retries: {last_error}"
                )
            raise RuntimeError(
                "The model did not return a usable video prompt. Try a different "
                "text model, or the flowing style."
            )

        # Best effort: a partial timeline beats nothing, and the beat count is
        # surfaced so the UI can say it came up short.
        return VideoPromptResult(
            prompt=best,
            style=style,
            duration=duration,
            source_caption=caption,
            model=self.model,
            raw_response=raw_final,
        )


    # ------------------------------------------------------------------
    # MMAudio conditioning
    # ------------------------------------------------------------------

    # MMAudio takes a positive soundscape description plus a negative prompt
    # for exclusions. Concrete sound sources work far better than abstract
    # nouns — "gentle waves lapping against shore" over "water".
    _AUDIO_SYSTEM_BASE = """You write audio prompts for MMAudio, a video-to-audio model.

The user gives you a video description. You write the soundtrack for it.

OUTPUT FORMAT — exactly two lines, nothing else:
POSITIVE: <the sounds that should be heard>
NEGATIVE: <sounds that must not appear>

RULES FOR POSITIVE:
- Name concrete sound SOURCES, not abstract nouns. "Gentle waves lapping
  against shore", never "water". "Fabric rustling against skin", never "noise".
- Layer the scene: the room's ambience, then the specific sounds the motion in
  the video would actually make.
- Only sounds that this scene would physically produce. If nothing in frame
  makes a sound, use the room tone and the subject's breathing.
- One line, roughly 10-25 words, comma-separated phrases.
- No camera or visual language — microphones do not see. Never mention shots,
  angles, lighting or framing.

RULES FOR NEGATIVE:
- List what would ruin the take. Default to "music, speech, narration" unless
  the scene clearly contains them.
- Add "distortion, clipping" when the scene is quiet or intimate.
- NEVER exclude room tone, ambience or the subject's own sounds — those are
  what you asked for in the positive prompt.
- At most 6 items. One line, comma-separated.

No preamble, no explanation, no markdown."""

    _AUDIO_EXPLICIT_RULES = """

EXPLICIT CONTENT:
- This is adult material. Include the sounds it would actually make: moaning,
  panting, laboured breathing, skin against skin, wet sounds, bedding creaking.
- Be direct. Do not sanitise, soften or omit them.
- Keep "music, narration" in the negative, but do NOT exclude moaning or
  breathing — they are the point.
- Everyone in the scene is an adult (18 or older)."""

    _POSITIVE_RE = re.compile(r"POSITIVE\s*:\s*(.+?)(?=\n|NEGATIVE\s*:|$)", re.IGNORECASE | re.DOTALL)
    _NEGATIVE_RE = re.compile(r"NEGATIVE\s*:\s*(.+?)(?=\n\n|$)", re.IGNORECASE | re.DOTALL)

    DEFAULT_AUDIO_NEGATIVE = "music, speech, narration, distortion"

    # Visual vocabulary that has no business in an audio prompt. The system
    # prompt forbids it, but models still slip in "soft focus shift" — a
    # camera move described as a sound.
    _VISUAL_TERMS = re.compile(
        r"\b(?:focus|camera|angle|lighting|light|framing|frame|shot|zoom|blur|"
        r"bokeh|colou?r|silhouette|visible|view)\b",
        re.IGNORECASE,
    )

    # Things the negative must never suppress — they are the point of the mix.
    _NEGATIVE_KEEP = re.compile(
        r"\b(?:ambien\w*|room tone|breath\w*|moan\w*|skin|rustl\w*)\b",
        re.IGNORECASE,
    )

    # MMAudio encodes text with CLIP, which accepts 77 tokens and silently
    # discards the rest — a long soundscape would lose its tail with no error.
    # Trimmed to whole comma-separated items so the prompt never ends mid-phrase.
    # (This is also why the per-second timeline format cannot be reused for
    # audio: five beats alone would exceed the budget.)
    _CLIP_TOKEN_LIMIT = 77

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        """Conservative CLIP token estimate for *value*."""
        # ~1.3 tokens per whitespace word, plus the start/end markers.
        return int(len(value.split()) * 1.3) + 2

    @classmethod
    def _fit_token_budget(cls, value: str) -> str:
        """Drop trailing items until the prompt fits CLIP's 77-token window."""
        if not value or cls._estimate_tokens(value) <= cls._CLIP_TOKEN_LIMIT:
            return value
        parts = [p.strip() for p in value.split(",") if p.strip()]
        kept: list[str] = []
        for part in parts:
            candidate = ", ".join(kept + [part])
            if kept and cls._estimate_tokens(candidate) > cls._CLIP_TOKEN_LIMIT:
                break
            kept.append(part)
        return ", ".join(kept)

    @classmethod
    def _filter_audio_terms(cls, value: str, negative: bool = False) -> str:
        """Drop comma-separated items that contradict the prompt's purpose.

        Visual language is stripped from the positive; anything that would mute
        the wanted ambience or performer sounds is stripped from the negative.
        """
        if not value:
            return value
        pattern = cls._NEGATIVE_KEEP if negative else cls._VISUAL_TERMS
        kept = [
            part.strip()
            for part in value.split(",")
            if part.strip() and not pattern.search(part)
        ]
        if not kept:
            return "" if not negative else cls.DEFAULT_AUDIO_NEGATIVE
        if negative:
            kept = kept[:6]
        return ", ".join(kept)

    @classmethod
    def build_audio_system_prompt(cls, explicit: bool = False) -> str:
        """System prompt for the MMAudio stage."""
        base = cls._AUDIO_SYSTEM_BASE
        return base + cls._AUDIO_EXPLICIT_RULES if explicit else base

    @classmethod
    def _parse_audio(cls, text: str) -> tuple[str, str]:
        """Pull the POSITIVE / NEGATIVE lines out of the model's reply."""
        if not text:
            return "", ""
        # Drop any leaked reasoning before parsing.
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\\boxed\{\s*\}", " ", text)

        def tidy(value: str) -> str:
            value = " ".join(value.split())
            value = value.strip().strip('"“”').strip()
            value = re.sub(r"[*`#\[\]]+", "", value).strip()
            return value.rstrip(".")

        pos_match = cls._POSITIVE_RE.search(text)
        neg_match = cls._NEGATIVE_RE.search(text)
        positive = tidy(pos_match.group(1)) if pos_match else ""
        negative = tidy(neg_match.group(1)) if neg_match else ""

        # Unlabelled single-line reply: treat it as the positive prompt.
        if not positive and not negative:
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            positive = tidy(first)

        positive = cls._fit_token_budget(cls._filter_audio_terms(positive))
        negative = cls._fit_token_budget(
            cls._filter_audio_terms(negative, negative=True)
        )
        return positive, negative

    def build_audio_prompt(
        self,
        video_prompt: str,
        caption: str = "",
        explicit: bool = False,
        max_attempts: int = 2,
    ) -> tuple[str, str]:
        """Write MMAudio ``(positive, negative)`` conditioning for a clip.

        Takes both the motion and the first-frame description: the motion
        decides the event sounds, the still decides the room tone.
        """
        if not video_prompt or not video_prompt.strip():
            raise ValueError("Video prompt cannot be empty")

        system = self.build_audio_system_prompt(explicit)
        prefix = self._no_think_prefix()

        parts = []
        if caption:
            parts.append(f"SCENE:\n{caption.strip()}")
        parts.append(f"WHAT HAPPENS:\n{video_prompt.strip()}")
        user = "\n\n".join(parts) + "\n\nWrite the audio prompt."
        if prefix:
            user = f"{prefix}\n\n{user}"

        best: tuple[str, str] = ("", "")
        for attempt in range(max_attempts):
            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=user,
                    system=system,
                    options={
                        "temperature": min(0.9, 0.6 + 0.15 * attempt),
                        "top_p": 0.9,
                        "num_predict": 200,
                        "stop": ["<think>"],
                    },
                    stream=False,
                )
                positive, negative = self._parse_audio(
                    response.get("response", "").strip()
                )
                if positive and len(positive.split()) > len(best[0].split()):
                    best = (positive, negative or self.DEFAULT_AUDIO_NEGATIVE)
                if positive:
                    return best
            except Exception as e:
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Audio prompt generation failed: {e}")
                logger.debug("Audio prompt attempt %d failed: %s", attempt + 1, e)

        return best


def get_video_prompter(
    host: str = VideoPrompter.DEFAULT_HOST,
    model: str = VideoPrompter.DEFAULT_MODEL,
) -> VideoPrompter:
    """Factory function to get a video prompter instance."""
    return VideoPrompter(host=host, model=model)
