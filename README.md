# Img-Tagbooru

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/saekimon)

A local, offline Danbooru-style image tagging tool for anime/illustration workflows. Tag images automatically with an ONNX vision model, or generate tags from text descriptions using a local LLM - no cloud APIs, no telemetry, fully private. Adult content is left to you and the models you install; the one hard rule is in [Legal, privacy and attribution](#legal-privacy-and-attribution).

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
| **Natural-Language Prompts** | Output a prose prompt for Krea / Flux instead of booru tags |
| **Image → Prompt (photos)** | Caption a photo directly with a local vision model |
| **Image → Video Prompt** | Turn a still into a Wan 2.2 image-to-video prompt |
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
- Save Current TXT - caption for selected image
- Save All TXT - all captions to a folder
- Export ZIP - all captions in a ZIP archive
- Format: `tag1, tag2, tag3` ready for training

### Recognition Model Management

The vision model that recognizes images is now selectable in-app - no config editing or reinstalling required. In the **Batch Tagger** tab, the **🧠 Recognition Model** dropdown (top of *Tagging Settings*) lets you choose the active model, and **⚙️ Manage** opens a manager to download, update, or delete models.

- **Choose:** pick any installed model from the dropdown; each entry shows a description and download state (`✓ installed` / `~size MB`).
- **Download:** selecting a model that isn't downloaded yet prompts to fetch it (cached under `~/.img_tagger`); the app stays responsive while it downloads in the background.
- **Update:** re-download a model in the manager to pull the latest weights.
- **Delete:** remove a downloaded model to reclaim disk space (re-downloadable anytime).
- **Custom models (advanced):** add any Hugging Face repo that ships `model.onnx` + `selected_tags.csv`.

Your selection is remembered between sessions. All models share the same BGR/square-pad preprocessing, so they are drop-in interchangeable.

| Model | Repo | Notes |
|-------|------|-------|
| **WD SwinV2 v3** ⭐ | `SmilingWolf/wd-swinv2-tagger-v3` | Balanced default - proven for LoRA captioning |
| **WD ViT v3** | `SmilingWolf/wd-vit-tagger-v3` | Slightly faster, comparable quality |
| **WD ConvNeXt v3** | `SmilingWolf/wd-convnext-tagger-v3` | Alternative architecture for a second opinion |
| **WD ViT-Large v3** | `SmilingWolf/wd-vit-large-tagger-v3` | Higher accuracy, larger/slower |
| **WD EVA02-Large v3** | `SmilingWolf/wd-eva02-large-tagger-v3` | Most accurate, largest download |

---

## Image-to-Prompt (Photographs)

**The ONNX taggers are for anime and illustration.** They're trained on Danbooru, so on a photograph they don't just get thin - they fail structurally. On a real test photo the tagger returned 80 tags, roughly a third wrong or self-contradictory:

- `painting (medium)`, `acrylic paint (medium)`, `traditional media` - on a photograph
- `dress` + `shirt` + `camisole` + `tank top` + `white dress` + `white shirt` - six mutually exclusive garments
- `black hair` + `brown hair`, `brown eyes` + `black eyes`, `large breasts` + `medium breasts`
- `hatsune miku (cosplay)` - an anime character projected onto a real person

No threshold setting fixes this; photos are out of domain. And booru tags have **no vocabulary at all** for lighting, camera framing, depth of field, or the character of a room - so nothing downstream can recover them.

For photos, use **📝 Write Prompt from Image** in the Batch Tagger caption toolbar. A local vision model describes the picture directly: one call, no tags in between. On the same photo it reported window light, the makeup brush on the desk, "bedroom or personal space", selfie framing, medium close-up, slight upward angle and soft focus.

### Setup

Requires a vision model pulled through Ollama. The repo publishes **no `latest` tag**, so name the quant explicitly - a bare `ollama pull` of it will 404:

```powershell
ollama pull aha2025/llama-joycaption-beta-one-hf-llava:Q8_0
```

| Quant | Size | Notes |
|-------|------|-------|
| **Q8_0** ⭐ | 9.42 GB | Best quality; fits a 16 GB card on its own |
| Q6_K | 7.47 GB | Near-Q8, a little smaller |
| Q5_K_M | 6.61 GB | Middle ground |
| Q4_K_M | 5.80 GB | Leaves room to keep a text LLM resident alongside it |

[JoyCaption](https://github.com/fpgaminer/joycaption) is purpose-built for captioning images to prompt and train diffusion models, and is uncensored by design - stock VLMs refuse or sanitise a lot of this tool's subject matter.

> **VRAM note:** on 16 GB, JoyCaption Q8 (9.4 GB) and qwen3-14b (9 GB) won't both stay resident - Ollama unloads one to load the other, costing a reload when you alternate. Q4_K_M keeps both in memory at ~14.8 GB.

### Using it

- **No tagging pass needed** - load images and caption them straight away
- **Caption styles:** the descriptive styles give flowing prose (what Krea/Flux want); "Stable Diffusion prompt" returns comma-separated fragments instead
- **Speed:** first run loads the model into VRAM and takes ~a minute; after that roughly 1-6s per image
- **Batch:** caption a whole folder; one failure doesn't sink the run
- **Saving:** files are written as `<name>_prompt.txt`, so they never overwrite the `<name>.txt` tag captions used for LoRA training

### Narration hint

A free-text **Narration hint** field steers how the description is written - *"cinematic film-noir tone"*, *"clinical and factual"*, *"emphasise texture and fabric"*. It's folded into the caption request immediately after the base instruction rather than appended, because position drives register here (the same finding as the explicit options below).

**It works far better on video prompts than on image captions**, and it's worth knowing why before you rely on it:

| Stage | Model | Effect |
|-------|-------|--------|
| Motion / video prompt | qwen3 (general instruction-follower) | **Strong.** "film-noir" produced *"a quiet hesitation… a silent question to the silence"*; "nature documentary" produced *"as if the room itself is exhaling"*. |
| Image caption | JoyCaption (template fine-tuned) | **Weak.** "film-noir" only shifted it toward framing language - *"a close-up, slightly low-angle shot"* - with none of the mood. |

JoyCaption is fine-tuned on a fixed set of instruction templates, which is exactly why its own caption styles and extra options work so reliably, and why free-text style instructions sit outside what it was trained to follow. For strong stylistic control over an image caption, the built-in styles and options are the effective lever; the hint is a nudge.

> **Trade-off worth knowing:** a heavy stylistic hint buys atmosphere at the cost of concrete motion. The film-noir video prompt above is better prose but arguably a *worse* Wan prompt - "a silent question to the silence" gives the model nothing to animate. Keep hints light when the output is going straight into generation.

The hint is deliberately **not** applied to the MMAudio prompt, which has a strict two-line format and a 77-token budget.

### Explicit content

JoyCaption is uncensored, but that isn't the whole story: a **formal register still euphemises**. Ask for a caption "in a formal tone" and suggestive imagery comes back as "modest cleavage" and "enhancing her natural beauty" - the model isn't refusing, it's being polite.

Tick **🔞 Explicit** to switch to a casual tone and instruct the model to describe anatomy and state of dress directly. Side by side on the same photo:

| | Formal | Explicit |
|---|---|---|
| Wording | "reveals moderate cleavage" | "low neckline, showing some cleavage" |
| Lighting | "bright, natural light from a window" | "Bright lighting from the left" |
| Framing | "slightly above eye level" | "slightly above eye level, from the chest up" |

### Wan 2.2 video prompts

The same dialog's **Output** dropdown can turn a still into an image-to-video prompt, in two shapes:

**Second-by-second timeline** - one beat per second, on one line:

```
(At 0 seconds: The woman blinks slowly, eyes flickering with subtle focus as the camera gently tilts up slightly.) (At 1 seconds: Her head tilts to the left, a faint smile forming, while the camera drifts slightly backward.) (At 2 seconds: Her fingers trace the silver pendant, as the camera slowly pulls back.) (At 3 seconds: She shifts her weight to the left, shoulder slightly lowered, as the camera tilts down toward her chest.) (At 4 seconds: Her gaze softens, eyes half-lidded, as the camera drifts leftward, following her line of sight.)
```

**Flowing paragraph** - the same motion as continuous prose. Clip length is adjustable from 2 to 20 seconds (timeline only).

The 20s ceiling is a reliability bound on the timeline format, not a Wan limit. Past roughly 15 seconds the model starts dropping beats, so a near-complete timeline is accepted after the first retry rather than burning attempts chasing the last one - and if it still comes up short, the output says so rather than quietly handing the video model fewer seconds than you asked for.

**Why this runs two models.** Asking the vision model to plan motion directly does not work - it's a captioner, so it describes what *is*. Tested head-on, it filled every beat with "hair remains mostly in place", "camera angle is the same", "no significant changes", contradicted the image about which hand was raised, and broke the output format. So the vision model describes the first frame, and the **text** model (the same one the Description Tagger uses) invents what happens next. It's told never to restate anything the first frame already fixes, and the words "remains", "continues" and "unchanged" are banned outright.

> **Cost:** two models means a VRAM swap on a 16 GB card - roughly 20s for the first image, faster after. The stage-2 text model is selectable in the dialog.

Video prompts save as `<name>_video.txt`, colliding with neither the image prompts nor the tag captions.

### MMAudio prompts

**🔊 Also write an MMAudio prompt** (on by default in video modes) writes the soundtrack for the same clip, and appears below the video prompt. MMAudio takes both a positive and a negative prompt, so both are generated:

```
POSITIVE: gravel crunching underfoot, distant rustle of wind through leaves, soft breaths
NEGATIVE: music, speech, distortion, clipping, car engine, birdsong
```

It's built from the motion *and* the first frame - the motion decides the event sounds, the still decides the room tone. The text model is already resident from the video stage, so it costs a few seconds and no extra VRAM swap. `🔊 Copy Audio` copies the positive; Shift-click copies the negative.

**Why there's no per-second audio timeline.** MMAudio doesn't work that way, for two independent reasons. Its text and video features are *pooled* and injected as global conditioning, so word order and timestamps collapse into a single vector - `(At 3 seconds: …)` means nothing more to it than the same words shuffled. And temporal placement doesn't come from the text at all: it comes from Synchformer features sampled at 24fps from the video, which is what actually decides *when* each sound lands. The video already carries the timing; the prompt only has to say *what*. On top of that, text goes through CLIP's 77-token window - a five-beat timeline would exceed it on its own.

Three things are enforced in code rather than left to the prompt:

- **Visual terms are stripped from the positive.** Asked for sounds, it produced "soft focus shift" - a camera move described as audio. Items containing focus/camera/angle/lighting/framing/shot/zoom/blur are dropped.
- **The negative can't mute what the positive asked for.** One run excluded "ambient room sounds" while the positive requested room tone. Items naming ambience, breathing, moaning, skin or rustling are removed from the negative, which is capped at six.
- **Both prompts are trimmed to CLIP's 77-token window**, dropping whole comma-separated items so they never end mid-phrase. Past that limit MMAudio silently discards the rest, which would quietly lose the tail of a soundscape. Typical output sits at 21–39 tokens, so this rarely fires - it exists so a verbose run can't fail invisibly.

Batch saves put audio in its own `<name>_audio.txt`, so a pipeline can consume video and audio prompts separately.

### Vulgar slang

Explicit mode also reorders what the model covers: body, pose and state of dress first, lighting and camera afterwards. Without that, requesting lighting *and* camera angle *and* a content rating eats the word budget and the caption never says what is actually exposed.

The toggle applies to **video prompts too** - with it off, the motion stage writes tame beats ("she blinks slowly, lashes fluttering") no matter how explicit the still is.

There's also a separate **Vulgar slang** checkbox. It's worth knowing what it does before reaching for it: it's tuned for training-caption realism, not prompt quality. In testing it traded descriptive precision for slang - "natural light from a window on the right" collapsed to "lighting's bright" - and profanity isn't a visual descriptor a diffusion model can act on. Useful for captioning a training set; usually counterproductive for prompting.

---

## Description-to-Tags

Describe what you want to see in plain English. The AI generates a comprehensive Danbooru-style tag set - the kind you'd find on a real Danbooru/Gelbooru post with 30-50+ tags covering every visual element.

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
| 🟡 **Creative** | Rich scene generation, atmosphere | Exhaustively tags every visual element - clothing items with colors, specific poses, accessories, lighting, composition. Stays SFW unless description is explicit. 35-50 tags. |
| 🔴 **Mature** | NSFW, explicit content | Everything Creative does + suggestive/explicit tags. On SFW descriptions adds tasteful flair (cleavage, bare_shoulders, bedroom_eyes). On explicit descriptions includes full sexual vocabulary. 35-50 tags. |

### Tag Enrichment Mode

Already have some tags? Switch to enrichment mode and provide seed tags like `1girl, witch_hat, forest, broom, night`. The AI expands them into a full 30-50 tag set with complementary clothing, atmosphere, lighting, and detail tags.

### Output Format: Danbooru Tags or Natural Language

Danbooru tags are the right format for SDXL-anime models (Illustrious, NoobAI, Pony). They are the *wrong* format for models that read plain English - Krea, Flux and similar T5-encoder models are trained on descriptive captions, and a comma-separated tag dump conditions them poorly.

The **Output format** dropdown picks which you get. It's independent of the input mode, so all four combinations work:

| | → Danbooru Tags | → Natural Language |
|---|---|---|
| **From Description** | comma-separated tags | prose prompt |
| **From Seed Tags** | enriched tag set | prose prompt from your tags |

That last cell is the quick path for converting an existing booru caption into a Krea prompt.

**Two-stage generation.** Natural-language output doesn't just paraphrase your description - it runs the full tag pipeline first, then rewrites the finished tag set as prose. The prompt inherits everything the pipeline adds, so the paragraph describes detail you never typed.

**Input:** `a girl, emo, black hair` → **stage 1:** 35 tags → **stage 2 (109 words):**

> A pale girl with long black hair in messy twintails and sidelocks stands in a full-body shot, her fringe falling over one eye. She wears a black t-shirt, long-sleeved black arm warmers, a pleated black skirt, and black knee-high boots, paired with a black choker and a studded belt at her waist. Her nails are painted black, and she has subtle ear piercings and a black hair ornament tucked into her hair. The background is simple, soft-focused with bokeh, against a deep red backdrop. She stands slightly off-center, caught in a moment of quiet intensity, lit with a slight depth of field and soft focus, capturing an emo aesthetic.

**What the prose stage does differently:**
- Front-loads the subject in the first clause - these models weight early tokens most heavily
- Works outward: appearance → clothing → pose → setting → lighting → mood, closing on framing, camera angle and style
- Never emits underscores, `(weight:1.3)` syntax, or `masterpiece, best quality, highres` spam - booru-model habits that do nothing here
- Won't contradict a tag (no "barefoot" against a `boots` tag, no sunlight in a `night` scene)
- Targets 70-110 words, hard-capped at 130 - an over-long prompt dilutes the subject

**Cost:** roughly twice the wall time of tag output (two LLM calls; ~5s total with a warm model on the recommended 14B). The status bar shows which stage is running, and a `🏷️ Copy Source Tags` button appears afterwards so one run gives you both the prompt and the tag set.

> **Note:** Creativity mode still applies - it shapes stage 1, and the prose stage carries the same content rules through. Safe stays clean, Mature stays explicit.

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
- Re-run for variety - temperature sampling means different runs produce different results
- Creative mode is usually the best default for rich output

### Post-Processing Pipeline

The raw LLM output goes through a deterministic pipeline that ensures quality:

1. **Vocabulary validation** - only tags that exist in the 1M-entry Danbooru CSV pass
2. **Relevance gate** - scores each tag against the description; drops off-topic hallucinations
3. **Safe mode filter** - hard-blocks explicit and suggestive tags in Safe mode
4. **Concept expansion** - `"emo"` expands to allow `pale_skin, eyeliner, choker, studded_belt`
5. **Act expansion** - `"blowjob"` permits `fellatio, penis, saliva, kneeling` (non-Safe only)
6. **Mature wildcards** - injects 2-3 tasteful suggestive tags on SFW descriptions in Mature mode
7. **Backfill** - if the LLM under-delivers, fills from archetype-specific clothing/detail pools
8. **Semantic dedup** - collapses synonyms (`blowjob` → `fellatio`, `drooling` → `saliva`)
9. **Conflict resolution** - mutual exclusion groups (can't be `standing` AND `sitting`)
10. **Sort by popularity** - higher post_count tags appear first

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

- **huihui_ai/qwen3-abliterated:30b-a3b-q4_K_M** - Thinking-mode variant. Reasoning tokens consume the generation budget, produces empty output.
- Any Qwen3 variant without `instruct-2507` in the name (thinking mode wastes tokens)

### Alternatives (sizes verified against the Ollama registry)

| Model | Size | 16 GB card | Notes |
|-------|------|-----------|-------|
| `huihui_ai/mistral-small-abliterated:24b` | 14.33 GB | fits | Mistral Small 24B, **non-thinking** - no `<think>` block to leak. The usual pick for uncensored prose. |
| `huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M` | 8.99 GB | fits | Qwen2.5, no thinking mode. Same footprint as the default. |
| `Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M` | 15.22 GB | tight | Ships a **vision projector** - could in principle serve both stages and avoid the VRAM swap. Captioning quality vs JoyCaption is unverified. |
| `huihui_ai/dolphin3-abliterated:8b` | 4.92 GB | fits | Lightweight; leaves room for a vision model alongside. |
| `huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M` | 18.56 GB | **does not fit** | Previously listed here as "may work" - measured at 18.56 GB, so it spills to system RAM on a 16 GB card. |

> **On thinking models:** the default is a Qwen3 reasoning model driven with `/no_think`, which suppresses the `<think>` block imperfectly - stray `</think>` and `\boxed{}` fragments do reach the output, and the prompt pipeline strips them. A natively non-thinking instruct model removes that failure mode rather than cleaning up after it.

### Measured: qwen3-14b vs mistral-small-24b

Both models on the same caption, timeline style, 5 seconds, two runs each:

| | qwen3-14b (default) | mistral-small-24b |
|---|---|---|
| Beat completeness | 5/5 both runs | 5/5 both runs |
| Reasoning leaked into raw output | **both runs** (`</think>`, `boxed`) | **none** |
| Speed, model warm | **1.8 s** | 8.9 s |
| Speed, cold load | 13 s | 46 s |
| Prompt-rule violations | none | one run used a banned word ("continues") |

Their MMAudio prompts differed in a way worth knowing about. From a caption describing a *bedroom with a window and natural light*:

- qwen3: `distant city sounds, subtle heartbeat, gentle hum of a ceiling fan, soft click of a curtain swaying`
- mistral: `soft breeze through window, distant birds chirping`

qwen3 invented a ceiling fan and city noise the scene never mentioned; MMAudio will render those faithfully. Mistral stayed with what the description supports.

**Neither wins outright.** Batch work favours qwen3 - the ~5× speed gap compounds over a folder, and the pipeline strips its leaks reliably. Single images where quality matters favour mistral: cleaner output and better-grounded audio. The stage-2 model is a dropdown, so this is a per-run choice rather than a setting to commit to.

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
- The build bundles `TERMS.md`, `THIRD_PARTY_NOTICES.txt` and `LICENSE`. After changing `requirements.txt`, regenerate the notices with `python scripts/generate_third_party_notices.py`

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

**`GET /health`** - Health check
```json
{"status": "ok"}
```

**`POST /tag`** - Tag a single image

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
- Vocabulary: `danbooru_tags_post_count.csv` - 1,000,000 tags with post counts
- Post-count threshold: 500 (configurable) - tags below this are filtered out
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
- **Re-run descriptions:** Temperature sampling means each run is different - try 2-3 times
- **Combine both:** Tag an image first, then use those tags as seeds in enrichment mode

---

## Roadmap

These are not confirmed - just things being explored or considered for future versions.

- **Tag translation** - translate Japanese Danbooru tags to English equivalents
- **Preset tag profiles** - saved configurations for common workflows (LoRA training, prompt generation, minimal clean output)
- **Optional online LLM support** - opt-in API key support for GPT-4o / Claude as an alternative to local Ollama for the Description Tagger
- **App localization** - UI translation support for non-English users (long-term)

Have a feature request? Open an issue on [GitHub](https://github.com/Xymoh/img-tagbooru/issues).

---

## Support the Project

If you find Img-Tagbooru useful, consider supporting development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/saekimon)

Your support helps keep this project free, open-source, and actively maintained.

---

## Legal, privacy and attribution

- **License:** MIT - see [LICENSE](LICENSE). Bundled third-party components and their licences are listed in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt); Qt is used through PySide6 under the LGPL-3.0.
- **Terms of use:** [TERMS.md](TERMS.md). The app is for adults (18+) and asks you to confirm that and accept the terms on first launch. Adult output is not filtered - it depends on the modes you pick and the models you install, and complying with local law is your responsibility.
- **The one hard rule:** the app never emits minor age-descriptor tags (`loli`, `shota`, `child`, …) in any mode that can produce suggestive output, and every explicit prompt states that all subjects are adults. This is enforced in the app's own code ([backend/content_policy.py](backend/content_policy.py)), not left to the model. Sexual content involving minors, drawn or otherwise, is a criminal offence in the EU, the US and Poland, and using this tool for it is prohibited.
- **Privacy:** no telemetry, no accounts, no uploads. The only network access is model downloads from Hugging Face, the GitHub update check in `update.bat`, Ollama on localhost, and any image URL you paste yourself. Settings and models live under `~/.img_tagger`.
- **Models:** WD tagger models by [SmilingWolf](https://huggingface.co/SmilingWolf) (Apache 2.0), downloaded on request. Language and vision models are pulled by you through Ollama under their own licences - JoyCaption is Llama 3.1-based (Meta Llama 3.1 Community License), Qwen and Mistral derivatives are Apache 2.0. Read the licence of anything you install.
- **Data:** `danbooru_tags_post_count.csv` contains tag names and post counts from [Danbooru](https://danbooru.donmai.us/), whose terms describe tags as factual, non-copyrightable information. No images are included.

## License

MIT License - see [LICENSE](LICENSE) for details.
