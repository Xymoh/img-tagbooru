# Img-Tagbooru

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/saekimon)

A local, offline Danbooru-style image tagging tool for anime/illustration workflows. Tag images automatically with an ONNX vision model, or generate tags from text descriptions using a local LLM — no cloud APIs, no content restrictions, fully private.

Built for LoRA trainers, dataset curators, and anyone who needs clean Danbooru-format captions.

This tool serves as a vast expansion of my initial tool: [Tagbooru](https://xymoh.github.io/Tagbooru/) Give it a look, it's a tool to format and categorize your copy pasted tags, for example from Danbooru and output them in a clean way.

---

## Features at a Glance

| Feature | What it does |
|---------|-------------|
| **Image → Tags** | Drop images in, get Danbooru tags with confidence scores |
| **Switchable Recognition Models** | Download, update, and choose between WD tagger models in-app |
| **Description → Tags** | Describe a scene in English, get 30-50 Danbooru tags |
| **Tag Enrichment** | Provide seed tags, get them expanded into a full tag set |
| **Batch Processing** | Tag entire folders at once |
| **Caption Editor** | Edit, reorder, filter, blacklist/whitelist tags |
| **Export** | Save `.txt` captions per image, or export as ZIP |

---

## Image-to-Tags

Uses an ONNX-based WD tagger model to classify images into Danbooru tags with confidence scores. The recognition model is switchable from inside the app (see below).

**Loading images:**
- Drag & drop files or folders onto the app
- Copy an image and paste with Ctrl+V
- Copy an image URL and paste with Ctrl+V
- Use the "Open Image" or "Open Folder" buttons

**Tagging controls:**
- **General Threshold** (0.25–0.40): Lower = more tags, may include false positives
- **Character Threshold** (0.80–0.95): Higher = only confident character matches
- **Max Tags**: Limit per image (40–80 typical for training)
- **MCut**: Automatic threshold detection (overrides manual settings)

**Working with results:**
- Uncheck tags in the Include column to exclude them
- Reorder tags by rank (lower = appears first in caption)
- Blacklist tags to always exclude (e.g. `blurry, lowres`)
- Whitelist to only include specific tags
- Supports regex patterns in blacklist/whitelist: `/^bad_/`

**Exporting:**
- Save Current TXT — caption for selected image
- Save All TXT — all captions to a folder
- Export ZIP — all captions in a ZIP archive
- Format: `tag1, tag2, tag3` ready for training

### Recognition Model Management

The vision model that recognizes images is now selectable in-app — no config editing or reinstalling required. In the **Batch Tagger** tab, the **🧠 Recognition Model** dropdown (top of *Tagging Settings*) lets you choose the active model, and **⚙️ Manage** opens a manager to download, update, or delete models.

- **Choose:** pick any installed model from the dropdown; each entry shows a description and download state (`✓ installed` / `~size MB`).
- **Download:** selecting a model that isn't downloaded yet prompts to fetch it (cached under `~/.img_tagger`); the app stays responsive while it downloads in the background.
- **Update:** re-download a model in the manager to pull the latest weights.
- **Delete:** remove a downloaded model to reclaim disk space (re-downloadable anytime).
- **Custom models (advanced):** add any Hugging Face repo that ships `model.onnx` + `selected_tags.csv`.

Your selection is remembered between sessions. All models share the same BGR/square-pad preprocessing, so they are drop-in interchangeable.

| Model | Repo | Notes |
|-------|------|-------|
| **WD SwinV2 v3** ⭐ | `SmilingWolf/wd-swinv2-tagger-v3` | Balanced default — proven for LoRA captioning |
| **WD ViT v3** | `SmilingWolf/wd-vit-tagger-v3` | Slightly faster, comparable quality |
| **WD ConvNeXt v3** | `SmilingWolf/wd-convnext-tagger-v3` | Alternative architecture for a second opinion |
| **WD ViT-Large v3** | `SmilingWolf/wd-vit-large-tagger-v3` | Higher accuracy, larger/slower |
| **WD EVA02-Large v3** | `SmilingWolf/wd-eva02-large-tagger-v3` | Most accurate, largest download |

---

## Description-to-Tags

Describe what you want to see in plain English. The AI generates a comprehensive Danbooru-style tag set — the kind you'd find on a real Danbooru/Gelbooru post with 30-50+ tags covering every visual element.

**Runs 100% locally** via Ollama. No API keys, no content filtering, no data leaves your machine.

### How It Works

1. You write a description: `"a girl, emo, black hair"`
2. The system builds a structured prompt that tells the LLM to cover specific visual categories
3. The LLM generates tags validated against a 1M-entry Danbooru vocabulary
4. A post-processing pipeline applies: relevance gating, concept expansion, semantic dedup, conflict resolution, and backfill

### Example Output

**Input:** `a girl, emo, black hair`

**Creative mode output (39 tags):**
```
1girl, solo, long_hair, looking_at_viewer, open_mouth, simple_background, long_sleeves,
black_hair, hair_ornament, standing, full_body, twintails, sidelocks, pleated_skirt, boots,
choker, black_skirt, miniskirt, hair_over_one_eye, black_shirt, nail_polish, black_choker,
black_boots, depth_of_field, x_hair_ornament, ear_piercing, t-shirt, red_background,
messy_hair, knee_boots, pale_skin, eyeliner, arm_warmers, studded_belt, black_arm_warmers, ...
```

### Creativity Modes

| Mode | Best For | Behavior |
|------|----------|----------|
| 🟢 **Safe** | SFW portraits, scenery, characters | Never outputs explicit tags. Strips sexual content even if described. Literal to the description with basic atmosphere. 20-35 tags. |
| 🟡 **Creative** | Rich scene generation, atmosphere | Exhaustively tags every visual element — clothing items with colors, specific poses, accessories, lighting, composition. Stays SFW unless description is explicit. 35-50 tags. |
| 🔴 **Mature** | NSFW, explicit content | Everything Creative does + suggestive/explicit tags. On SFW descriptions adds tasteful flair (cleavage, bare_shoulders, bedroom_eyes). On explicit descriptions includes full sexual vocabulary. 35-50 tags. |

### Tag Enrichment Mode

Already have some tags? Switch to enrichment mode and provide seed tags like `1girl, witch_hat, forest, broom, night`. The AI expands them into a full 30-50 tag set with complementary clothing, atmosphere, lighting, and detail tags.

### Writing Better Descriptions

The AI maps your words to tags across these dimensions:

| Dimension | Good | Poor |
|-----------|------|------|
| **Subject** | "a knight", "two elves", "a catgirl" | "someone", "a character" |
| **Action** | "baking cookies", "standing on a cliff" | "existing" |
| **Setting** | "in a forest clearing", "dark cathedral" | (nothing) |
| **Clothing** | "maid outfit", "bikini", "armor and cape" | (nothing) |
| **Atmosphere** | "stormy night", "sunset", "candlelight" | (nothing) |

**Tips:**
- More detail = better output. `"a witch"` → generic. `"a witch flying through a dark storm"` → rich.
- Name specific clothing/accessories if you want them tagged
- Re-run for variety — temperature sampling means different runs produce different results
- Creative mode is usually the best default for rich output

### Post-Processing Pipeline

The raw LLM output goes through a deterministic pipeline that ensures quality:

1. **Vocabulary validation** — only tags that exist in the 1M-entry Danbooru CSV pass
2. **Relevance gate** — scores each tag against the description; drops off-topic hallucinations
3. **Safe mode filter** — hard-blocks explicit and suggestive tags in Safe mode
4. **Concept expansion** — `"emo"` expands to allow `pale_skin, eyeliner, choker, studded_belt`
5. **Act expansion** — `"blowjob"` permits `fellatio, penis, saliva, kneeling` (non-Safe only)
6. **Mature wildcards** — injects 2-3 tasteful suggestive tags on SFW descriptions in Mature mode
7. **Backfill** — if the LLM under-delivers, fills from archetype-specific clothing/detail pools
8. **Semantic dedup** — collapses synonyms (`blowjob` → `fellatio`, `drooling` → `saliva`)
9. **Conflict resolution** — mutual exclusion groups (can't be `standing` AND `sitting`)
10. **Sort by popularity** — higher post_count tags appear first

---

## Setup

### Requirements

- **OS:** Windows 10/11 (primary), Linux/macOS should work but untested
- **Python:** 3.10+
- **RAM:** 8 GB minimum (16 GB+ for Description Tagger)
- **GPU:** Optional for image tagging (NVIDIA/AMD). Recommended for Description Tagger (16 GB VRAM)
- **Disk:** ~5 GB for image tagger models, ~10 GB per LLM model

### Installation

```powershell
# Clone the repo
git clone https://github.com/Xymoh/img-tagbooru.git
cd img-tagbooru

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support for image tagging
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### Running the App

```powershell
python run.py
```

Or directly:
```powershell
python frontend/native/main_window.py
```

### Description Tagger Setup (Optional)

The description-to-tags feature requires Ollama running locally:

1. **Install Ollama** from [ollama.ai](https://ollama.ai)
2. **Start Ollama:** `ollama serve`
3. **Pull the model:** `ollama pull richardyoung/qwen3-14b-abliterated`
4. **Verify GPU:** `ollama ps` (should show "GPU loaded")

First inference takes 30s–2min (model loading). Subsequent runs average ~4s.

> **Note:** The Description Tagger is optional. Image-to-tags works without Ollama.

---

## Recommended LLM Models

### Primary (Tested & Verified)

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| **richardyoung/qwen3-14b-abliterated** | 14B (~9 GB) | 16 GB | ~4s/run | ⭐⭐⭐⭐⭐ |

```
ollama pull richardyoung/qwen3-14b-abliterated
```

This is the default and recommended model. Abliterated (uncensored) Qwen3-14B with `/no_think` support that skips reasoning tokens for fast, clean tag output.

### Alternative (Lighter)

| Model | Size | VRAM | Speed | Best For |
|-------|------|------|-------|----------|
| **goonsai/qwen2.5-3B-goonsai-nsfw-100k** | 3B (~2 GB) | 4 GB | ~7s/run | Quick iterations, low VRAM |

```
ollama pull goonsai/qwen2.5-3B-goonsai-nsfw-100k
```

Purpose-built for image prompts. Thinner atmospheric coverage but good for rapid re-runs.

### Not Recommended

- **huihui_ai/qwen3-abliterated:30b-a3b-q4_K_M** — Thinking-mode variant. Reasoning tokens consume the generation budget, produces empty output.
- Any Qwen3 variant without `instruct-2507` in the name (thinking mode wastes tokens)

### Untested (May Work)

- `huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M` — Non-thinking MoE, ~18 GB
- `huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M` — Qwen2.5, no thinking mode
- `Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M` — Dense 24B, uncensored creative

---

## Building a Windows Executable

```powershell
.\build_exe.ps1
```

Output: `dist\Img-Tagbooru.exe`

GitHub Releases: push a tag like `v1.0.0` and the CI workflow publishes the exe as a release asset.

Notes:
- First run still downloads model files to the user's cache
- Windows Defender/SmartScreen warnings are normal for unsigned builds

---

## Project Structure

```
img-tagger/
├── backend/
│   ├── api.py                  # Optional FastAPI server
│   ├── tagger.py               # ONNX image tagger
│   ├── description_tagger.py   # LLM description-to-tags engine
│   ├── tag_index.py            # Danbooru vocabulary index (1M tags)
│   └── tag_utils.py            # Tag manipulation utilities
├── frontend/
│   └── native/
│       ├── main_window.py      # PySide6 desktop app (entry point)
│       ├── widgets.py          # UI components, Help dialog
│       ├── workers.py          # Background threads for LLM/image ops
│       ├── completer.py        # Tag autocomplete
│       └── styles.py           # UI stylesheet
├── danbooru_tags_post_count.csv  # 1M Danbooru tags with post counts
├── requirements.txt
├── run.py                      # App launcher
└── build_exe.ps1               # Windows exe build script
```

---

## FastAPI Server (Optional)

For custom UI integration, the backend is also available as a REST API:

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

### Endpoints

**`GET /health`** — Health check
```json
{"status": "ok"}
```

**`POST /tag`** — Tag a single image

Request: `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image (PNG, JPG, WebP, BMP) |
| `general_threshold` | float | 0.35 | Min confidence for general tags |
| `character_threshold` | float | 0.85 | Min confidence for character tags |
| `normalize_pixels` | bool | false | Normalize to 0–1 range |
| `use_mcut` | bool | false | MCut auto-threshold |
| `limit` | int | 80 | Max tags (0 = unlimited) |

Response:
```json
{
  "caption": "1girl, smile, blue_eyes, solo, ...",
  "tags": [
    {"tag": "1girl", "confidence": 0.9876, "category": 0, "category_label": "general"},
    {"tag": "hatsune_miku", "confidence": 0.9521, "category": 4, "category_label": "character"}
  ]
}
```

CORS enabled for `localhost:5173` (Vite dev server).

---

## Technical Details

### Image Tagger
- Default model: `SmilingWolf/wd-swinv2-tagger-v3` (Hugging Face), switchable in-app
- Bundled choices: WD SwinV2 / ViT / ConvNeXt / ViT-Large / EVA02-Large (v3), plus custom HF repos
- Runtime: ONNX (CPU or GPU)
- Categories: general (0), artist (1), copyright (3), character (4), meta (5)
- Models are cached under `~/.img_tagger`; the active model is remembered in `~/.img_tagger/config.json`

### Description Tagger
- LLM: Ollama + abliterated Qwen3-14B (local, offline)
- Vocabulary: `danbooru_tags_post_count.csv` — 1,000,000 tags with post counts
- Post-count threshold: 500 (configurable) — tags below this are filtered out
- Structured category prompting forces coverage of: participants, hair, eyes, clothing, accessories, body, pose, expression, setting, lighting, atmosphere, framing, quality
- Concept expansion maps: 60+ archetypes/settings with associated tag pools
- Act expansion maps: 40+ NSFW keywords with anatomy/position/reaction tags
- Semantic dedup: 15 synonym groups collapsed to canonical forms
- Conflict groups: 6 mutual-exclusion sets (pose, viewpoint, framing, mouth, time, weather)

---

## Tips for Best Results

- **For training data:** Use image tagger with threshold 0.30–0.35, max 60–80 tags
- **For prompt generation:** Use description tagger in Creative mode for rich, specific tags
- **Blacklist common noise:** `blurry, lowres, bad_id, bad_pixiv_id, commentary_request`
- **Re-run descriptions:** Temperature sampling means each run is different — try 2-3 times
- **Combine both:** Tag an image first, then use those tags as seeds in enrichment mode

---

## Roadmap

These are not confirmed — just things being explored or considered for future versions.

- **Tag translation** — translate Japanese Danbooru tags to English equivalents
- **Preset tag profiles** — saved configurations for common workflows (LoRA training, prompt generation, minimal clean output)
- **Optional online LLM support** — opt-in API key support for GPT-4o / Claude as an alternative to local Ollama for the Description Tagger
- **App localization** — UI translation support for non-English users (long-term)

Have a feature request? Open an issue on [GitHub](https://github.com/Xymoh/img-tagboru-ai/issues).

---

## Support the Project

If you find Img-Tagbooru useful, consider supporting development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/saekimon)

Your support helps keep this project free, open-source, and actively maintained.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
