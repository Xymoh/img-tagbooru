# Contributing to Img-Tagbooru

## Prerequisites

- Python 3.10+
- Git
- (Optional) Ollama for Description-to-Tags feature
- (Optional) NVIDIA GPU + CUDA for accelerated inference

## Development Setup

```powershell
git clone https://github.com/<your-org>/img-tagger.git
cd img-tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU support:
```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

## Project Structure

| Path | Purpose |
|------|---------|
| `backend/tagger.py` | ONNX tagger inference (WD-SwinV2), model download, prediction |
| `backend/tag_utils.py` | Shared data structures (`TaggingResult`), DataFrame helpers, AI metadata extraction |
| `backend/description_tagger.py` | Description→tags via local LLM (Ollama), mature-mode enrichment |
| `backend/api.py` | FastAPI endpoints (`/tag`, `/health`) for Vite integration |
| `frontend/native/main_window.py` | PySide6 main window — all UI widgets, event handlers, and workflow orchestration |
| `frontend/native/completer.py` | `TagTrie` prefix tree + `CaptionCompleterMixin` for Danbooru autocomplete |
| `frontend/native/workers.py` | QThread workers for description tagging and image loading |
| `frontend/native/styles.py` | Dark-theme Qt stylesheet, SVG icon generation |
| `frontend/native/widgets.py` | `CheckboxDelegate` and `HelpDialog` |
| `build_exe.ps1` | PyInstaller packaging script |
| `.github/workflows/windows-build.yml` | CI/CD: builds `.exe` and publishes GitHub Release asset |
| `danbooru_tags_post_count.csv` | 1M+ Danbooru tag vocabulary for autocomplete |

## Running the App

Desktop (PySide6):
```powershell
python frontend/native/main_window.py
```

Desktop (PySide6):
```powershell
python frontend/native/main_window.py
```

API only:
```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

## Code Conventions

- **Python 3.10+** with `from __future__ import annotations` in every file.
- Use `logging` module — never `print()` for application output. Module-level logger:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```
- Type hints on all function signatures.
- GUI code uses PySide6 (Qt for Python).
- Workers for long-running tasks inherit `QtCore.QThread` and communicate via signals.
- Shared logic lives in `backend/` so it can be reused by any future frontend.

## Branching & Release Workflow

```
feature/bugfix → develop → main → tag → back-merge
```

1. Create a feature/bugfix branch from `develop`.
2. Open a PR targeting `develop`.
3. After review, squash-merge into `develop`.
4. When ready for release, open a PR from `develop` → `main`.
5. Merge to `main`, tag with `vX.Y.Z`, push the tag.
6. CI builds and attaches `Img-Tagbooru.exe` to the GitHub Release.
7. Back-merge `main` → `develop`.

## Building the EXE

```powershell
.\build_exe.ps1
```

Output: `dist\Img-Tagbooru.exe`

Requirements for the build:
- PyInstaller installed in venv
- ONNX Runtime (CPU or GPU) installed
- All `requirements.txt` packages present

## Testing

No formal test suite exists yet. Manual verification checklist:
- [ ] Image-to-tags: upload/drag/paste various image formats
- [ ] Batch folder processing
- [ ] Tag table editing (include/exclude, reorder, filter)
- [ ] Export: `.txt`, zip, beside-source
- [ ] Undo/Redo (Ctrl+Z / Ctrl+Y)
- [ ] Description-to-Tags with Ollama
- [ ] Watch-folder auto-tagging
- [ ] Autocomplete with partial tag input
- [ ] AI metadata extraction from PNG
- [ ] Prompt Tools (copy as prompt, negative prompt, tag frequency)
