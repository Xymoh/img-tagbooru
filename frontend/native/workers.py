from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore

from backend.description_tagger import (
    DescriptionTagResult,
    NaturalPromptResult,
    get_description_tagger,
)
from backend.tag_utils import IMAGE_EXTENSIONS
from backend.vlm_captioner import VLMCaptionResult, get_vlm_captioner


class ModelOperationWorker(QtCore.QThread):
    """Pull or delete an Ollama model in a background thread to keep the UI responsive."""

    finished = QtCore.Signal(bool, str)
    """Emitted with (success, message) when the operation completes."""

    def __init__(self, operation: str, model_name: str) -> None:
        super().__init__()
        self._operation = operation  # "pull" or "delete"
        self._model_name = model_name

    def run(self) -> None:
        try:
            tagger = get_description_tagger()
            if self._operation == "pull":
                tagger.pull_model(self._model_name)
                self.finished.emit(True, f"Model '{self._model_name}' pulled successfully.")
            elif self._operation == "delete":
                tagger.delete_model(self._model_name)
                self.finished.emit(True, f"Model '{self._model_name}' deleted.")
            else:
                self.finished.emit(False, f"Unknown operation: {self._operation}")
        except Exception as e:
            self.finished.emit(False, str(e))


class TaggerModelWorker(QtCore.QThread):
    """Download (and warm) or delete an ONNX recognition model off the UI thread.

    Downloads are hundreds of megabytes, so they must never run on the Qt event
    loop. On a successful download the model session is also built and cached so
    the first image tag after switching is instant instead of stalling the UI.
    """

    finished = QtCore.Signal(bool, str, str)
    """Emitted with (success, message, repo_id) when the operation completes."""

    def __init__(self, operation: str, repo_id: str) -> None:
        super().__init__()
        self._operation = operation  # "download" or "delete"
        self._repo_id = repo_id

    def run(self) -> None:
        from backend import tagger as tagger_backend

        try:
            if self._operation == "download":
                tagger_backend.download_model(self._repo_id)
                # Build the ONNX session now so switching to it is instant and
                # any load error surfaces here instead of at first tag.
                tagger_backend.get_tagger(self._repo_id)
                self.finished.emit(True, "Model downloaded and ready to use.", self._repo_id)
            elif self._operation == "delete":
                tagger_backend.delete_model_files(self._repo_id)
                self.finished.emit(True, "Model deleted.", self._repo_id)
            else:
                self.finished.emit(False, f"Unknown operation: {self._operation}", self._repo_id)
        except Exception as e:
            self.finished.emit(False, str(e), self._repo_id)


class DescriptionTagWorker(QtCore.QThread):
    finished = QtCore.Signal(DescriptionTagResult)
    prompt_finished = QtCore.Signal(NaturalPromptResult)
    """Emitted instead of ``finished`` when output_format is "natural"."""
    stage = QtCore.Signal(str)
    """Progress label for multi-stage runs, e.g. "Stage 2/2: writing prompt"."""
    error = QtCore.Signal(str)

    def __init__(
        self,
        description: str,
        model: str,
        creativity: str,
        post_count_threshold: int = 500,
        enrich_mode: bool = False,
        output_format: str = "tags",
    ) -> None:
        super().__init__()
        self.description = description
        self.model = model
        self.creativity = creativity
        self.post_count_threshold = post_count_threshold
        self.enrich_mode = enrich_mode
        self.output_format = output_format  # "tags" or "natural"

    def run(self) -> None:
        try:
            tagger = get_description_tagger(model=self.model)
            tagger.set_post_count_threshold(self.post_count_threshold)

            # Stage 1 is shared by both output formats: description (or seed
            # tags) -> cleaned Danbooru tag set.
            if self.output_format == "natural":
                self.stage.emit("Stage 1/2: building tag set")
            if self.enrich_mode:
                result = tagger.enrich_tags(self.description, creativity=self.creativity)
            else:
                result = tagger.generate_tags(self.description, creativity=self.creativity)

            if self.output_format == "natural":
                # Stage 2 rewrites the finished tag set as prose. Passing the
                # stage-1 result avoids regenerating the tags.
                self.stage.emit("Stage 2/2: writing natural-language prompt")
                prompt_result = tagger.generate_natural_prompt(
                    self.description,
                    creativity=self.creativity,
                    seed_tags_mode=self.enrich_mode,
                    tag_result=result,
                )
                self.prompt_finished.emit(prompt_result)
                return

            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class VLMCaptionWorker(QtCore.QThread):
    """Caption one or more images with a vision model, off the UI thread.

    Vision inference is slow enough (seconds per image, plus a one-off model
    load of several GB) that it must never touch the Qt event loop.
    """

    finished = QtCore.Signal(list, list)
    """Emitted with ([(index, VLMCaptionResult)], [failure messages]).

    Failures are per-stage: an image can caption fine and still fail its
    video or audio stage. Those must reach the UI — a caption arriving with no
    video prompt and no explanation looks identical to "video mode is broken".
    """
    item_done = QtCore.Signal(int, VLMCaptionResult)
    """Emitted per image as it completes, so the UI can fill in progressively."""
    progress = QtCore.Signal(int, str)
    """(percent, label) for the progress bar."""
    error = QtCore.Signal(str)

    def __init__(
        self,
        images: list,
        model: str,
        style: str,
        length: str,
        extras: list[str] | None = None,
        explicit: bool = False,
        video_style: str = "",
        video_duration: int = 5,
        text_model: str = "",
        want_audio: bool = False,
        custom_instruction: str = "",
    ) -> None:
        super().__init__()
        # images: list of (index, path-or-PIL-image)
        self._images = images
        self._model = model
        self._style = style
        self._length = length
        self._extras = list(extras or [])
        self._explicit = explicit
        # Empty video_style means caption only (no second stage).
        self._video_style = video_style
        self._video_duration = video_duration
        self._text_model = text_model
        self._want_audio = want_audio
        self._custom_instruction = custom_instruction
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current image."""
        self._cancelled = True

    def run(self) -> None:
        try:
            captioner = get_vlm_captioner(model=self._model)
            if not captioner.check_connection():
                self.error.emit(
                    "Cannot connect to Ollama. Make sure it is running:\n"
                    "  ollama serve"
                )
                return
            if captioner.is_installed() and not captioner.has_vision():
                self.error.emit(
                    f"'{self._model}' is a text-only model — it cannot read "
                    "images.\n\nOllama reports no 'vision' capability for it, "
                    "so there is no image input path. Pick a vision model "
                    "(JoyCaption, or anything else you have pulled that "
                    "reports vision support)."
                )
                return
            if not captioner.is_installed():
                self.error.emit(
                    f"Vision model '{self._model}' is not installed.\n\n"
                    f"Pull it first:\n  ollama pull {self._model}\n\n"
                    "Note: this repo has no 'latest' tag, so the quant must be "
                    "named explicitly."
                )
                return
        except Exception as e:
            self.error.emit(str(e))
            return

        prompter = None
        if self._video_style:
            # A curated vision captioner cannot do stage 2: asked for a
            # timeline it restates the still or fails outright. Refuse up
            # front with a clear reason rather than burning retries per image
            # and then hiding the result.
            from backend.vlm_captioner import list_vlm_models

            captioner_ids = {info.model_id for info in list_vlm_models()}
            if self._text_model in captioner_ids:
                self.error.emit(
                    f"'{self._text_model}' is an image captioner, not a text "
                    "model — it cannot write the motion prompt.\n\n"
                    "Stage 2 turns the first-frame caption into a timeline, "
                    "which is a language task. Pick an instruction-following "
                    "text model (qwen3, Mistral) in the stage-2 dropdown."
                )
                return

            # Stage 2 is a text model, so this swaps VRAM on a 16 GB card.
            # Built once up front rather than per image.
            try:
                from backend.video_prompter import get_video_prompter

                prompter = get_video_prompter(
                    model=self._text_model or None  # type: ignore[arg-type]
                ) if self._text_model else get_video_prompter()
            except Exception as e:
                self.error.emit(f"Could not start the video prompt stage: {e}")
                return

        results: list = []
        total = len(self._images)
        failures: list[str] = []

        for i, (index, image) in enumerate(self._images):
            if self._cancelled:
                break
            stage = "Describing" if prompter else "Captioning"
            self.progress.emit(
                int(i / total * 100), f"{stage} image {i + 1} of {total}..."
            )
            try:
                result = captioner.caption_image(
                    image,
                    style=self._style,
                    length=self._length,
                    extras=self._extras,
                    explicit=self._explicit,
                    custom_instruction=self._custom_instruction,
                )
            except Exception as e:
                # One bad image must not sink the whole batch.
                failures.append(f"#{index + 1}: {e}")
                continue

            if prompter is not None and not self._cancelled:
                self.progress.emit(
                    int((i + 0.5) / total * 100),
                    f"Writing video prompt {i + 1} of {total} "
                    "(switching models, first one is slow)...",
                )
                try:
                    video = prompter.build_from_caption(
                        result.caption,
                        style=self._video_style,
                        duration=self._video_duration,
                        # Without this the motion stage writes in its own
                        # polite register no matter how blunt stage 1 was.
                        explicit=self._explicit,
                        custom_instruction=self._custom_instruction,
                    )
                    result = replace(
                        result,
                        video_prompt=video.prompt,
                        video_style=video.style,
                        video_duration=video.duration,
                    )

                    if self._want_audio and video.prompt:
                        # Same model is already resident, so this is cheap —
                        # no second VRAM swap.
                        self.progress.emit(
                            int((i + 0.75) / total * 100),
                            f"Writing MMAudio prompt {i + 1} of {total}...",
                        )
                        try:
                            positive, negative = prompter.build_audio_prompt(
                                video.prompt,
                                caption=result.caption,
                                explicit=self._explicit,
                            )
                            result = replace(
                                result,
                                audio_prompt=positive,
                                audio_negative=negative,
                            )
                        except Exception as e:
                            # The video prompt is still good on its own.
                            failures.append(f"#{index + 1} (audio stage): {e}")
                except Exception as e:
                    # Keep the caption — it is still useful on its own — but
                    # record which style was wanted, so the view can say the
                    # video prompt is missing instead of silently showing a
                    # caption where a timeline should be.
                    result = replace(result, video_style=self._video_style)
                    failures.append(f"#{index + 1} (video stage): {e}")

            results.append((index, result))
            self.item_done.emit(index, result)

        self.progress.emit(100, "Done")

        if not results and failures:
            self.error.emit("Captioning failed:\n\n" + "\n".join(failures[:5]))
            return
        # Partial failures ride along with the results rather than being
        # dropped: a caption that arrives without its video prompt must say so.
        self.finished.emit(results, failures)


class ImageLoadWorker(QtCore.QThread):
    finished = QtCore.Signal(list, list)
    progress = QtCore.Signal(int)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths

    def run(self) -> None:
        valid: list[Path] = []
        skipped: list[str] = []
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append(path.name)
                self.progress.emit(int((i + 1) / total * 100))
                continue
            try:
                Image.open(path).convert("RGB")
                valid.append(path)
            except (UnidentifiedImageError, OSError):
                skipped.append(path.name)
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(valid, skipped)
