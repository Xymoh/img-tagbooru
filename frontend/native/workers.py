from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore

from backend.description_tagger import DescriptionTagResult, get_description_tagger
from backend.tag_utils import IMAGE_EXTENSIONS


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
    error = QtCore.Signal(str)

    def __init__(
        self,
        description: str,
        model: str,
        creativity: str,
        post_count_threshold: int = 500,
        enrich_mode: bool = False,
    ) -> None:
        super().__init__()
        self.description = description
        self.model = model
        self.creativity = creativity
        self.post_count_threshold = post_count_threshold
        self.enrich_mode = enrich_mode

    def run(self) -> None:
        try:
            tagger = get_description_tagger(model=self.model)
            tagger.set_post_count_threshold(self.post_count_threshold)
            if self.enrich_mode:
                result = tagger.enrich_tags(self.description, creativity=self.creativity)
            else:
                result = tagger.generate_tags(self.description, creativity=self.creativity)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


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
