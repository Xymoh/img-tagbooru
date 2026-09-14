from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Custom delegate for table checkbox column with styled checkmark
# ---------------------------------------------------------------------------

class CheckboxDelegate(QtWidgets.QStyledItemDelegate):
    """Draws a styled checkbox with a visible checkmark in the table."""

    def paint(self, painter, option, index):
        painter.save()
        checked = index.data(QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked

        # Draw cell background
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor("#0059b3"))
        else:
            bg = QtGui.QColor("#0d0d0d") if index.row() % 2 == 0 else QtGui.QColor("#1a1a1a")
            painter.fillRect(option.rect, bg)

        # Box dimensions
        box_size = 16
        cx = option.rect.center().x()
        cy = option.rect.center().y()
        box_rect = QtCore.QRect(cx - box_size // 2, cy - box_size // 2, box_size, box_size)

        if checked:
            # Filled blue box
            painter.setBrush(QtGui.QColor("#0059b3"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#4da6ff"), 1.5))
            painter.drawRoundedRect(box_rect, 3, 3)
            # Draw white checkmark
            pen = QtGui.QPen(
                QtGui.QColor("#ffffff"), 2.2,
                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            x, y = box_rect.x(), box_rect.y()
            w, h = box_rect.width(), box_rect.height()
            p1 = QtCore.QPointF(x + w * 0.15, y + h * 0.50)
            p2 = QtCore.QPointF(x + w * 0.42, y + h * 0.75)
            p3 = QtCore.QPointF(x + w * 0.85, y + h * 0.22)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
        else:
            # Empty dark box
            painter.setBrush(QtGui.QColor("#0d0d0d"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#666666"), 1.5))
            painter.drawRoundedRect(box_rect, 3, 3)

        painter.restore()

    def sizeHint(self, option, index):
        return QtCore.QSize(40, 26)


# ---------------------------------------------------------------------------
# Help dialog
# ---------------------------------------------------------------------------

class HelpDialog(QtWidgets.QDialog):
    """Help dialog with usage instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📖 User Guide - Img-Tagbooru")
        self.resize(700, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #0d0d0d;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                color: #ffffff;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("🏷️ Img-Tagbooru - User Guide")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4da6ff; margin-bottom: 10px;")
        layout.addWidget(title)

        # Content browser
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml("""
        <h2 style="color: #4da6ff;">Quick Start Guide</h2>

        <h3 style="color: #66ff66;">📁 Loading Images</h3>
        <ul>
            <li><b>Drag & Drop:</b> Drag image files or folders directly onto the app</li>
            <li><b>Clipboard:</b> Copy an image and press Ctrl+V</li>
            <li><b>Buttons:</b> Use "Open Image" or "Open Folder" buttons</li>
            <li><b>Web Images:</b> Copy image URL and paste with Ctrl+V</li>
        </ul>

        <h3 style="color: #66ff66;">⚙️ Tagging Settings</h3>
        <ul>
            <li><b>🧠 Recognition Model:</b> Choose which ONNX vision model tags your images.
                Use <b>⚙️ Manage</b> to download, update, or delete models (WD SwinV2, ViT,
                ConvNeXt, ViT-Large, EVA02-Large, or a custom Hugging Face repo). Your choice is remembered.</li>
            <li><b>General Threshold (0.25-0.40):</b> Lower = more tags, may include false positives</li>
            <li><b>Character Threshold (0.80-0.95):</b> Higher = only confident character matches</li>
            <li><b>Max Tags:</b> Limit tags per image (40-80 is typical for training)</li>
            <li><b>MCut:</b> Automatic threshold detection (overrides manual settings)</li>
        </ul>

        <h3 style="color: #66ff66;">🏷️ Working with Tags</h3>
        <ul>
            <li><b>Include Column:</b> Uncheck to exclude tags from caption</li>
            <li><b>Rank:</b> Manually order tags (lower = appears first)</li>
            <li><b>Blacklist:</b> Tags to always exclude (e.g., "blurry, lowres")</li>
            <li><b>Whitelist:</b> Only include these tags if specified</li>
            <li><b>Sort Mode:</b> Change how tags are displayed/ordered</li>
        </ul>

        <h3 style="color: #66ff66;">💾 Exporting Results</h3>
        <ul>
            <li><b>Save Current TXT:</b> Save caption for selected image</li>
            <li><b>Save All TXT:</b> Save all captions to a folder</li>
            <li><b>Export ZIP:</b> Download all captions in a ZIP file</li>
            <li><b>Caption Format:</b> "tag1, tag2, tag3" ready for training</li>
        </ul>

        <h3 style="color: #66ff66;">📝 Description Tagger (Tab 2)</h3>
        <ul>
            <li>Describe what you want to see in English</li>
            <li>AI generates Danbooru-style tags from your description</li>
            <li>Requires Ollama installed (see setup below)</li>
            <li>Choose creativity mode: Safe → Creative → Mature</li>
            <li><b>Re-run for variety:</b> The AI uses temperature sampling - running the same prompt again can produce different (sometimes better) results</li>
        </ul>

        <h3 style="color: #66ff66;">🏷️ Output Format: Tags or Natural Language</h3>
        <p>The <b>Output format</b> dropdown picks what the Description Tagger produces. It is
        independent of the input mode, so all four combinations work.</p>
        <ul>
            <li><b>🏷️ Danbooru Tags</b> - comma-separated tags for SDXL-anime models
                (Illustrious, NoobAI, Pony) in ComfyUI or A1111.</li>
            <li><b>📝 Natural Language</b> - a flowing prose paragraph for models that read plain
                English, such as Krea and Flux. No underscores, no <code>(weight:1.3)</code> syntax,
                and no <i>masterpiece, best quality</i> spam - those are booru-model habits that do
                nothing on these encoders.</li>
        </ul>
        <p><b>Natural Language runs in two stages:</b> a Danbooru tag set is generated first, then
        rewritten as prose. That means the prompt inherits everything the tag pipeline adds -
        concept expansion, backfill, dedup and conflict resolution - so "a girl, emo, black hair"
        becomes a paragraph naming the studded belt and chipped nail polish you never typed.
        It takes roughly twice as long as tag output, and the status bar shows which stage is
        running. A second button copies the source tags, so one run gives you both.</p>

        <h3 style="color: #66ff66;">📝 Write Prompt from Image (photos)</h3>
        <p>The ONNX taggers are trained on <b>Danbooru</b> - anime and illustration. On a
        <b>photograph</b> they degrade badly: expect a painting/medium mislabel, mutually exclusive
        garment guesses, and occasionally an anime character projected onto a real person. That is
        not a threshold you can tune; photos are simply out of domain.</p>
        <p>For photos, use <b>📝 Write Prompt from Image</b> in the caption toolbar. A local vision
        model describes the picture directly in natural language - one step, no tags in between -
        so it can report lighting, camera framing, depth of field and the character of the room,
        none of which exist in the booru tag vocabulary.</p>
        <ul>
            <li><b>Setup:</b> requires a vision model pulled through Ollama. The dialog lists the
                JoyCaption quants and shows which are installed. Note that repo has no
                <code>latest</code> tag, so the quant must be named explicitly:<br>
                <code>ollama pull aha2025/llama-joycaption-beta-one-hf-llava:Q8_0</code></li>
            <li><b>No tagging pass needed</b> - load images and caption them straight away.</li>
            <li><b>Caption styles:</b> the descriptive styles produce flowing prose, which is what
                Krea/Flux condition on. "Stable Diffusion prompt" returns comma-separated fragments
                instead.</li>
            <li><b>Speed:</b> the first run loads several GB into VRAM and can take a minute;
                after that each image takes a few seconds.</li>
            <li><b>Saving:</b> prompt files are written as <code>&lt;name&gt;_prompt.txt</code> so
                they never overwrite the <code>&lt;name&gt;.txt</code> tag captions used for LoRA
                training.</li>
        </ul>
        <p><b>Explicit content:</b> the vision model is uncensored, but a <i>formal</i> register
        still euphemises - suggestive imagery comes back as "modest cleavage". Tick
        <b>🔞 Explicit</b> to switch to a casual tone and have it describe anatomy and state of
        dress directly. The separate <b>Vulgar slang</b> option adds profanity, but trades
        descriptive detail for slang and is usually counterproductive for prompting - it is meant
        for captioning training sets.</p>

        <p><b>🎬 Wan 2.2 video prompts:</b> the <b>Output</b> dropdown can also turn the still into
        an image-to-video prompt - either a second-by-second timeline
        (<code>(At 0 seconds: ...) (At 1 seconds: ...)</code>, 2-12 seconds) or a flowing
        paragraph. This runs <i>two</i> models: the vision model describes the first frame, then the
        text model invents the motion. Asking the vision model to plan motion on its own does not
        work - it is a captioner, so it just restates the still. Expect a VRAM swap on the first
        image, and pick the stage-2 text model in the dialog. Saved as
        <code>&lt;name&gt;_video.txt</code>.</p>

        <p><b>🔊 MMAudio prompts:</b> in video modes, <b>Also write an MMAudio prompt</b> (on by
        default) writes the soundtrack for the same clip - a positive soundscape plus a negative
        prompt for exclusions - shown below the video prompt. <b>🔊 Copy Audio</b> copies the
        positive; Shift-click copies the negative. It reuses the text model already loaded for the
        motion stage, so it costs seconds and no extra VRAM swap. Batch saves write it to
        <code>&lt;name&gt;_audio.txt</code>.</p>

        <p><b>✍️ Narration hint:</b> the free-text field steers how the description is worded
        (<i>"cinematic film-noir tone"</i>, <i>"clinical and factual"</i>). It has a much stronger
        effect on <b>video</b> prompts than on image captions - the motion stage runs a general
        instruction-following model, while the captioner is fine-tuned on fixed templates, so free
        text nudges it rather than transforming it. For caption style, the built-in styles and
        option checkboxes are the effective lever. Keep hints light when the output goes straight
        into generation: heavy stylistic wording buys atmosphere at the cost of the concrete motion
        a video model needs.</p>

        <p>The ONNX taggers remain the right tool for anime/illustration LoRA captioning - this is
        an additional path, not a replacement.</p>

        <h3 style="color: #ff9933;">✍️ Writing Better Descriptions</h3>
        <p>The more concrete visual detail you provide, the better the tags. The AI maps descriptions to tags across these dimensions:</p>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Dimension</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Good Example</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Poor Example</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Subject</b><br><span style="color: #888;">who is in the scene?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"a knight", "two elves", "a catgirl"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">"someone", "a character"</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444;"><b>Action</b><br><span style="color: #888;">what are they doing?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"baking cookies", "kissing", "standing"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">"existing", "being"</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Setting</b><br><span style="color: #888;">where does this happen?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"in a forest clearing", "on the train"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no setting mentioned)</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444;"><b>Atmosphere</b><br><span style="color: #888;">mood / lighting</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"cozy", "stormy night", "romantic"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no mood mentioned)</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Clothing</b><br><span style="color: #888;">what are they wearing?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"a maid outfit", "armor and cape"<br><span style="color: #888;">(or implied by archetype: witch → hat)</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no clothing clues)</td>
            </tr>
        </table>

        <h4 style="color: #ffcc66;">Mode Selection Guide</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Mode</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Best For</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Tags Include</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">🟢 Safe</td>
                <td style="padding: 6px; border: 1px solid #444;">SFW, character portraits, scenery</td>
                <td style="padding: 6px; border: 1px solid #444;">No sexual content - groping/kissing gets softened to blush/romance</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444; color: #ff9933;">🟡 Creative</td>
                <td style="padding: 6px; border: 1px solid #444;">Action, atmosphere, mild romance</td>
                <td style="padding: 6px; border: 1px solid #444;">Style/lighting tags, kissing, hand-holding, suggestive - no explicit</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">🔴 Mature</td>
                <td style="padding: 6px; border: 1px solid #444;">NSFW, explicit sexual content</td>
                <td style="padding: 6px; border: 1px solid #444;">Full sexual vocabulary, body language, intimate settings</td>
            </tr>
        </table>

        <h4 style="color: #ff6666;">Common Pitfalls</h4>
        <ul style="font-size: 12px;">
            <li><b>Too vague:</b> "a witch" → few generic tags. Try "a witch flying through a dark storm"</li>
            <li><b>Abstract concepts:</b> "a feeling of dread" → can't map to visual Danbooru tags. Describe what that looks like instead</li>
            <li><b>Franchise names:</b> "Hollow Knight", "Disgaea" → may collide with tag namespace. Use generic descriptions</li>
        </ul>

        <h4 style="color: #66ff66;">If Results Are Poor</h4>
        <ol style="font-size: 12px; margin: 5px 0;">
            <li><b>Add detail:</b> Make sure you have a subject + action + setting</li>
            <li><b>Re-run:</b> Temperature sampling means different runs produce different results</li>
            <li><b>Try another mode:</b> Creative often produces richer atmospheric tags than Safe</li>
            <li><b>Be explicit:</b> If a tag is missing, name the element directly</li>
        </ol>

        <h3 style="color: #ff9933;">💡 Pro Tips</h3>
        <ul>
            <li>Use blacklist to filter out unwanted tags permanently</li>
            <li>Lower general threshold catches more details but may include noise</li>
            <li>Edit tags in the table and caption updates automatically</li>
            <li>Drag multiple images or entire folders at once</li>
            <li>For LoRA training, keep captions clean with 40-80 quality tags</li>
        </ul>

        <h3 style="color: #ff66a3;">🔧 System Requirements</h3>
        <ul>
            <li><b>RAM:</b> 8GB minimum, 16GB recommended (32GB+ for Description Tagger)</li>
            <li><b>GPU:</b> Optional but speeds up tagging significantly (NVIDIA/AMD)</li>
            <li><b>Disk:</b> 5GB for models and temp files (15-25GB for LLM models)</li>
        </ul>

        <h3 style="color: #ff9933;">🤖 Description Tagger Setup</h3>
        <ol style="margin: 5px 0;">
            <li>Install <b>Ollama</b> from <a href='https://ollama.ai' style='color: #4da6ff;'>ollama.ai</a></li>
            <li>For GPU: Install NVIDIA CUDA or AMD ROCm drivers</li>
            <li>Start Ollama: run <code>ollama serve</code> in terminal</li>
            <li>Verify GPU: run <code>ollama ps</code> (shows 'GPU loaded')</li>
            <li>Pull the recommended model: <code>ollama pull richardyoung/qwen3-14b-abliterated</code></li>
        </ol>
        
        <h4 style="color: #66ff66;">⭐ Recommended Model - Tested & Verified</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Size</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Quality</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Pull Command</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;"><b>🏆 Qwen3-14B-Abliterated</b><br><span style="color: #888;">Default · 9.9 avg tags/run · 4s/run · 16GB VRAM</span></td>
                <td style="padding: 6px; border: 1px solid #444;">14B<br><span style="color: #888;">~9GB Q4_K_M</span></td>
                <td style="padding: 6px; border: 1px solid #444;">⭐⭐⭐⭐⭐<br>Recommended</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull richardyoung/qwen3-14b-abliterated</td>
            </tr>
        </table>

        <h4 style="color: #66ff66;">🔁 Tested Alternative</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Size</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Best for</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Pull Command</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #ffcc66;"><b>Goonsai Qwen2.5-3B NSFW</b><br><span style="color: #888;">Purpose-built for image prompts · 7s/run · 7.3 avg tags</span></td>
                <td style="padding: 6px; border: 1px solid #444;">3B<br><span style="color: #888;">~2GB</span></td>
                <td style="padding: 6px; border: 1px solid #444;">Quick re-runs when main model is sparse; thinner atmospheric coverage</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull goonsai/qwen2.5-3B-goonsai-nsfw-100k</td>
            </tr>
        </table>

        <h4 style="color: #888;">Not Recommended (tested, returned empty output)</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Why skip</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #888;"><b>huihui_ai/qwen3-abliterated:30b-a3b-q4_K_M</b></td>
                <td style="padding: 6px; border: 1px solid #444; color: #888;">Thinking-mode variant - reasoning tokens consume the generation budget, produces empty tags. Use the <code>instruct-2507</code> variant instead (untested).</td>
            </tr>
        </table>

        <h4 style="color: #888;">Untested Alternatives (may or may not work)</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Pull Command</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Qwen3 30B-A3B Instruct-2507 Abliterated</b><br><span style="color: #888;">Non-thinking MoE, ~18GB - should run with partial RAM offload</span></td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull huihui_ai/qwen3-abliterated:30b-a3b-instruct-2507-q4_K_M</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Qwen2.5 14B Abliterated Instruct</b><br><span style="color: #888;">Qwen2.5 has no thinking mode by design - clean baseline</span></td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull huihui_ai/qwen2.5-abliterate:14b-instruct-q4_K_M</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Cydonia 24B v4.3 Heretic</b><br><span style="color: #888;">Dense 24B, tuned for uncensored creative writing</span></td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M</td>
            </tr>
        </table>

        <p style='color: #ffcc66; font-size: 11px; margin-top: 8px;'>
            💡 <b>Speed:</b> Qwen3-14B runs at ~4s/run thanks to the <code>/no_think</code> directive that skips reasoning tokens.<br>
            💡 <b>Weak output?</b> Re-run 2-3 times - temperature sampling produces different results. Goonsai-3B is great for quick iterations.<br>
            💡 <b>Explicit content:</b> All listed models are abliterated/uncensored for NSFW tags.<br>
            💡 <b>Avoid thinking-mode Qwen3 variants</b> (anything without <code>instruct-2507</code> in the name) - they waste the output budget on reasoning.
        </p>

        <hr style="border: 1px solid #444;">
        <div style="text-align: center; margin: 12px 0;">
            <p style="color: #ff5e5b; font-size: 13px; font-weight: bold;">☕ Enjoy this tool? Support development!</p>
            <p style="font-size: 12px;">
                <a href="https://ko-fi.com/saekimon" style="color: #ff5e5b; font-size: 14px; font-weight: bold; text-decoration: none;">
                    ☕ Buy me a coffee on Ko-fi
                </a>
            </p>
            <p style="color: #888; font-size: 11px;">
                Your support helps keep this project free, open-source, and actively maintained.
            </p>
        </div>
        <hr style="border: 1px solid #444;">
        <p style="color: #9ecbff; font-size: 11px;">
            Need help? Check the
            <a href="https://github.com/Xymoh/img-tagbooru" style="color: #4da6ff;">GitHub repository</a>
            or review the README.md file.
        </p>
        """)
        layout.addWidget(browser)

        # Close button
        close_btn = QtWidgets.QPushButton("✓ Got it!")
        close_btn.setMinimumHeight(35)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0059b3;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0073e6;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ---------------------------------------------------------------------------
# Bundled text documents (TERMS.md, THIRD_PARTY_NOTICES.txt, LICENSE)
# ---------------------------------------------------------------------------

def find_bundled_file(filename: str):
    """Locate a project-root data file in dev and PyInstaller builds."""
    import sys
    from pathlib import Path

    base = getattr(sys, "_MEIPASS", None)
    candidates = []
    if base:
        candidates.append(Path(base) / filename)
    candidates.append(Path(__file__).resolve().parents[2] / filename)
    candidates.append(Path.cwd() / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_bundled_text(filename: str, fallback: str = "") -> str:
    path = find_bundled_file(filename)
    if path is None:
        return fallback
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback


# Version of TERMS.md the user must have accepted. Bump this whenever the
# terms change materially so the dialog is shown again on next launch.
TERMS_VERSION = "2026-09-13"

_TERMS_FALLBACK = (
    "# Img-Tagbooru - Terms of Use\n\n"
    "The bundled TERMS.md could not be found. The current terms are published at\n"
    "https://github.com/Xymoh/img-tagbooru/blob/main/TERMS.md\n\n"
    "In short: you must be 18 or older; the software is provided as is under the "
    "MIT License; you are responsible for the models you install and for complying "
    "with the law where you live; sexual content involving minors and non-consensual "
    "intimate content of real people are prohibited."
)


class TextViewerDialog(QtWidgets.QDialog):
    """Read-only viewer for a bundled document (Markdown or plain text)."""

    def __init__(self, title: str, text: str, parent=None, markdown: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 640)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a1a; color: #ffffff; }
            QTextBrowser {
                background-color: #0d0d0d; color: #e0e0e0;
                border: 1px solid #333; padding: 10px; font-size: 12px;
            }
            QPushButton {
                background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #444;
                border-radius: 3px; padding: 6px 14px;
            }
            QPushButton:hover { background-color: #3b3b3b; border: 1px solid #4da6ff; }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        if markdown:
            browser.setMarkdown(text)
        else:
            browser.setPlainText(text)
        layout.addWidget(browser)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)


class TermsDialog(QtWidgets.QDialog):
    """First-run gate: age confirmation and acceptance of TERMS.md.

    Shown before the main window is built, and again whenever TERMS_VERSION
    changes. Declining exits the application.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Img-Tagbooru - Terms of Use")
        self.setModal(True)
        self.resize(760, 680)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a1a; color: #ffffff; }
            QLabel { color: #ffffff; }
            QCheckBox { color: #ffffff; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QTextBrowser {
                background-color: #0d0d0d; color: #e0e0e0;
                border: 1px solid #333; padding: 10px; font-size: 12px;
            }
            QPushButton {
                background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #444;
                border-radius: 3px; padding: 6px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3b3b3b; border: 1px solid #4da6ff; }
            QPushButton#accept { background-color: #0059b3; color: white; font-weight: bold; }
            QPushButton#accept:hover { background-color: #0073e6; }
            QPushButton#accept:disabled { background-color: #333; color: #777; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Before you start</b><br>"
            "Img-Tagbooru is free, open-source software that runs entirely on your "
            "computer. It can produce adult text output when you enable those modes, "
            "so it is for adults only. Please read the terms below."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(read_bundled_text("TERMS.md", _TERMS_FALLBACK))
        layout.addWidget(browser, stretch=1)

        self.age_check = QtWidgets.QCheckBox("I confirm that I am at least 18 years old.")
        self.terms_check = QtWidgets.QCheckBox(
            "I have read and accept the Terms of Use, Acceptable Use and Privacy Notice."
        )
        layout.addWidget(self.age_check)
        layout.addWidget(self.terms_check)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        decline_btn = QtWidgets.QPushButton("Decline and exit")
        decline_btn.clicked.connect(self.reject)
        self.accept_btn = QtWidgets.QPushButton("Accept and continue")
        self.accept_btn.setObjectName("accept")
        self.accept_btn.setEnabled(False)
        self.accept_btn.clicked.connect(self.accept)
        buttons.addWidget(decline_btn)
        buttons.addWidget(self.accept_btn)
        layout.addLayout(buttons)

        self.age_check.toggled.connect(self._update_accept)
        self.terms_check.toggled.connect(self._update_accept)

    def _update_accept(self) -> None:
        self.accept_btn.setEnabled(
            self.age_check.isChecked() and self.terms_check.isChecked()
        )

    def reject(self) -> None:  # Escape / window close count as declining
        super().reject()
