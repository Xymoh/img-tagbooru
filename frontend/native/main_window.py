from __future__ import annotations

import io
import logging
import random
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import urljoin
from uuid import uuid4

logger = logging.getLogger(__name__)

# Single source of truth for the version shown in the title bar, the About
# dialog, the splash screen and QApplication metadata.
APP_VERSION = "1.3.7"

import pandas as pd
from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QTimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.description_tagger import (
    DescriptionTagResult,
    NaturalPromptResult,
    get_description_tagger,
)
from backend import tagger as tagger_backend
from backend.tagger import category_label, get_tagger, predict_tags
from backend.tag_utils import (
    IMAGE_EXTENSIONS,
    TaggingResult,
    apply_filters,
    export_zip_from_results,
    extract_ai_metadata,
    frame_from_predictions,
    frame_to_caption,
    metadata_to_tags,
    sort_frame,
    split_tags,
)

from frontend.native.completer import CaptionCompleterMixin
from frontend.native.styles import build_stylesheet
from frontend.native.widgets import (
    HelpDialog,
    TermsDialog,
    TextViewerDialog,
    TERMS_VERSION,
    read_bundled_text,
)
from frontend.native.workers import (
    DescriptionTagWorker,
    ImageLoadWorker,
    ModelOperationWorker,
    TaggerModelWorker,
    VLMCaptionWorker,
)
from backend.vlm_captioner import (
    CAPTION_LENGTHS,
    CAPTION_STYLES,
    DEFAULT_LENGTH,
    DEFAULT_STYLE,
    EXPLICIT_EXTRAS,
    EXTRA_OPTIONS,
    VLMCaptioner,
    list_vlm_models,
)
from backend.video_prompter import (
    DEFAULT_DURATION,
    MAX_DURATION,
    MIN_DURATION,
    RELIABLE_DURATION,
    VIDEO_STYLES,
    VideoPrompter,
)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow, CaptionCompleterMixin):
    def __init__(self, progress_cb: "Callable[[str], None] | None" = None) -> None:
        super().__init__()
        # Reports construction phases to the splash screen. Defaults to a no-op
        # so the window can still be built headlessly (tests, embedding).
        self._boot = progress_cb or (lambda _message: None)
        self._boot("Building interface…")
        self.setWindowTitle(f"Img-Tagbooru v{APP_VERSION}")
        self.resize(1500, 960)
        self.setMinimumSize(1200, 720)
        self.setAcceptDrops(True)

        self.pending_paths: list[Path] = []
        # Vision captioning runs on a QThread that can outlive its dialog, so
        # the reference is held here rather than in the dialog's local scope.
        self._vlm_worker: VLMCaptionWorker | None = None
        # Settings for the image/video prompt dialog. Held on the window so
        # they survive closing and reopening the dialog, and nowhere more
        # durable than that — a fresh app launch starts from the defaults.
        self._vlm_dialog_state: dict[str, object] = {}
        self.results: list[TaggingResult] = []
        self._single_results: dict[int, TaggingResult] = {}
        self._active_result_index = -1
        self._preview_image: Image.Image | None = None
        self._tag_worker: DescriptionTagWorker | None = None
        self._image_load_worker: ImageLoadWorker | None = None
        self._last_description_tags: list[str] = []
        self._last_creativity_mode = "creative"
        self.danbooru_tags: list[str] = []
        self.caption_completer: QtWidgets.QCompleter | None = None

        # Watch-folder auto-tagging
        self._watcher: QtCore.QFileSystemWatcher | None = None
        self._watch_dir: Path | None = None
        self._watch_timer: QTimer = QTimer(self)
        self._watch_timer.setSingleShot(True)
        self._watch_timer.setInterval(800)  # 800ms debounce
        self._watch_timer.timeout.connect(self._on_watch_timer)
        self._watch_pending: set[Path] = set()

        # Load Danbooru tags for autocomplete
        self.danbooru_tags = self._load_danbooru_tags()

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setSpacing(4)
        root_layout.setContentsMargins(6, 6, 6, 6)

        self.setStyleSheet(build_stylesheet())

        # --- Tab widget --------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs)

        # ===== TAB 1: Batch Tagger =====
        batch_tab = QtWidgets.QWidget()
        batch_tab_layout = QtWidgets.QVBoxLayout(batch_tab)
        batch_tab_layout.setContentsMargins(4, 4, 4, 4)
        batch_tab_layout.setSpacing(0)

        # Outer horizontal splitter — user can drag to resize left vs right
        batch_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        batch_splitter.setChildrenCollapsible(False)
        batch_splitter.setHandleWidth(6)
        batch_tab_layout.addWidget(batch_splitter)

        # Vertical splitters for each side — let user balance inner groups
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(6)
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(6)

        batch_splitter.addWidget(left_splitter)
        batch_splitter.addWidget(right_splitter)
        batch_splitter.setStretchFactor(0, 2)
        batch_splitter.setStretchFactor(1, 3)
        # Initial size hint: ~40% / 60%
        batch_splitter.setSizes([600, 900])

        input_group = QtWidgets.QGroupBox("📁 Input - Load Images")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        input_layout.setSpacing(8)

        # --- Row 1: File operations + watch folder ---
        file_row = QtWidgets.QHBoxLayout()
        file_row.setSpacing(6)
        self.open_images_btn = QtWidgets.QPushButton("🖼️ Open Images")
        self.open_images_btn.setToolTip("Select one or more image files to tag")
        self.open_images_btn.clicked.connect(self.open_images)
        self.open_folder_btn = QtWidgets.QPushButton("📂 Open Folder")
        self.open_folder_btn.setToolTip("Select a folder containing images\n(Can include subfolders on user choice)")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.watch_folder_cb = QtWidgets.QCheckBox("👁 Watch Folder (Auto-Load)")
        self.watch_folder_cb.setToolTip(
            "Automatically load new images saved to the last-opened folder.\n"
            "Ideal for ComfyUI output directories — tags appear as images render."
        )
        self.watch_folder_cb.toggled.connect(self._toggle_watch_folder)
        file_row.addWidget(self.open_images_btn)
        file_row.addWidget(self.open_folder_btn)
        file_row.addWidget(self.watch_folder_cb)
        file_row.addStretch(1)
        input_layout.addLayout(file_row)

        # --- Row 2: Tagging actions ---
        tag_row = QtWidgets.QHBoxLayout()
        tag_row.setSpacing(6)
        self.tag_selected_btn = QtWidgets.QPushButton("⚡ Tag Selected")
        self.tag_selected_btn.setObjectName("tagSelectedBtn")
        self.tag_selected_btn.setToolTip("Tag only the currently selected image")
        self.tag_selected_btn.clicked.connect(self.process_single_image)
        self.tag_selected_btn.setEnabled(False)
        self.tag_btn = QtWidgets.QPushButton("🏷️ Tag All Images")
        self.tag_btn.setObjectName("tagBtn")
        self.tag_btn.setToolTip("Start tagging all loaded images with current settings")
        self.tag_btn.clicked.connect(self.process_pending)
        self.ai_meta_btn = QtWidgets.QPushButton("✅ Extract Positive Prompts")
        self.ai_meta_btn.setObjectName("aiMetaBtn")
        self.ai_meta_btn.setToolTip("Read AI generation parameters embedded in the\n"
                                     "currently selected image (SD/A1111/ComfyUI) and treat them as tags, if exists")
        self.ai_meta_btn.clicked.connect(self._extract_ai_metadata_for_current)
        self.ai_meta_btn.setEnabled(False)
        tag_row.addWidget(self.tag_selected_btn)
        tag_row.addWidget(self.tag_btn)
        tag_row.addWidget(self.ai_meta_btn)
        tag_row.addStretch(1)
        input_layout.addLayout(tag_row)

        list_label = QtWidgets.QLabel("📋 Loaded Images:")
        list_label.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        input_layout.addWidget(list_label)

        self.result_list = QtWidgets.QListWidget()
        self.result_list.currentRowChanged.connect(self.show_result)
        self.result_list.currentRowChanged.connect(self._update_tag_selected_button)
        self.result_list.setToolTip("Click an image to preview and edit its tags")
        # Embed the drag-and-drop hint as a placeholder inside the list
        # (visible only when empty, disappears once images are loaded)
        self._list_placeholder = QtWidgets.QLabel(
            "💡 Drag & drop images or folders here\n"
            "or use the buttons above\n"
            "or press Ctrl+V to paste",
            self.result_list,
        )
        self._list_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._list_placeholder.setStyleSheet(
            "color: #555; font-size: 10px; background: transparent;"
        )
        self._list_placeholder.setWordWrap(True)
        self._list_placeholder.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        input_layout.addWidget(self.result_list, 1)

        self.pending_label = QtWidgets.QLabel("")
        self.pending_label.setObjectName("alertLabel")
        self.pending_label.setWordWrap(True)
        self.pending_label.setVisible(False)
        input_layout.addWidget(self.pending_label)
        left_splitter.addWidget(input_group)

        preview_group = QtWidgets.QGroupBox("🖼️ Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(6, 6, 6, 6)

        # Label that re-scales its pixmap whenever its own size changes
        # (handles splitter drags as well as window resizes)
        class _PreviewLabel(QtWidgets.QLabel):
            def __init__(self, parent_win, **kwargs):
                super().__init__(**kwargs)
                self._win = parent_win

            def resizeEvent(self, event):
                super().resizeEvent(event)
                if self._win._preview_pixmap is not None and not self.size().isEmpty():
                    self.setPixmap(
                        self._win._preview_pixmap.scaled(
                            self.size(),
                            QtCore.Qt.KeepAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )

        self._preview_pixmap: QtGui.QPixmap | None = None
        self.image_label = _PreviewLabel(self, alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet(
            "background: #0d0d0d; border: 2px solid #333; border-radius: 8px; padding: 2px;"
        )
        self.image_label.setToolTip("Preview of selected image")
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        preview_layout.addWidget(self.image_label)
        left_splitter.addWidget(preview_group)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)

        settings_group = QtWidgets.QGroupBox("⚙️ Tagging Settings")
        form = QtWidgets.QFormLayout(settings_group)
        form.setSpacing(8)

        # ── Recognition model (ONNX image tagger) ────────────────────────────
        # Mirrors the LLM model selector on the Description tab: choose which
        # vision model recognizes images, and download/update/remove models.
        # Packed into one tight block so the dense settings panel stays compact.
        model_hdr = QtWidgets.QLabel("🧠 Recognition Model")
        model_hdr.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")

        self.tagger_model_selector = QtWidgets.QComboBox()
        self.tagger_model_selector.setToolTip(
            "The ONNX vision model used to recognize images and produce tags.\n"
            "Not-downloaded models are fetched on first use."
        )
        self.tagger_model_selector.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 4px;"
        )

        self.manage_tagger_btn = QtWidgets.QPushButton("⚙️ Manage")
        self.manage_tagger_btn.setToolTip("Download, update, or remove recognition models")
        self.manage_tagger_btn.clicked.connect(self._show_tagger_model_manager)

        model_selector_row = QtWidgets.QHBoxLayout()
        model_selector_row.setContentsMargins(0, 0, 0, 0)
        model_selector_row.setSpacing(4)
        model_selector_row.addWidget(self.tagger_model_selector, 1)
        model_selector_row.addWidget(self.manage_tagger_btn)

        self.tagger_model_desc = QtWidgets.QLabel("")
        self.tagger_model_desc.setWordWrap(True)
        self.tagger_model_desc.setStyleSheet("color: #9ecbff; font-size: 10px;")

        model_sep = QtWidgets.QFrame()
        model_sep.setFrameShape(QtWidgets.QFrame.HLine)
        model_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        model_sep.setStyleSheet("color: #333;")

        model_box = QtWidgets.QWidget()
        model_box_layout = QtWidgets.QVBoxLayout(model_box)
        model_box_layout.setContentsMargins(0, 0, 0, 0)
        model_box_layout.setSpacing(4)
        model_box_layout.addWidget(model_hdr)
        model_box_layout.addLayout(model_selector_row)
        model_box_layout.addWidget(self.tagger_model_desc)
        model_box_layout.addWidget(model_sep)
        form.addRow(model_box)

        # Tracks the currently active repo so we can revert the combo if the
        # user picks a not-downloaded model and then cancels the download.
        self._active_tagger_repo: str = tagger_backend.get_selected_model()
        self._tagger_worker: TaggerModelWorker | None = None
        self.tagger_model_selector.currentIndexChanged.connect(self._on_tagger_model_changed)
        self._refresh_tagger_models()

        self.general_threshold = QtWidgets.QDoubleSpinBox()
        self.general_threshold.setObjectName("generalThreshold")
        self.general_threshold.setRange(0.0, 1.0)
        self.general_threshold.setSingleStep(0.01)
        self.general_threshold.setValue(0.6)
        self.general_threshold.setToolTip(
            "Lower = more tags (0.5-0.7 recommended)\nHigher = only very confident tags"
        )
        self.general_threshold.setStyleSheet("color: #66ff66; font-weight: bold;")

        self.character_threshold = QtWidgets.QDoubleSpinBox()
        self.character_threshold.setObjectName("characterThreshold")
        self.character_threshold.setRange(0.0, 1.0)
        self.character_threshold.setSingleStep(0.01)
        self.character_threshold.setValue(0.85)
        self.character_threshold.setToolTip(
            "Higher = only confident character matches (0.80-0.95)\nLower = may detect false characters"
        )
        self.character_threshold.setStyleSheet("color: #ff66a3; font-weight: bold;")

        self.max_tags = QtWidgets.QSpinBox()
        self.max_tags.setObjectName("maxTags")
        self.max_tags.setRange(5, 200)
        self.max_tags.setValue(40)
        self.max_tags.setToolTip("Limit tags per image (40-80 is typical for training)")
        self.max_tags.setStyleSheet("color: #ffcc66; font-weight: bold;")

        self.sort_mode = QtWidgets.QComboBox()
        self.sort_mode.addItems(["confidence", "alphabetical", "manual rank"])
        self.sort_mode.setToolTip("How to order tags in the results table")

        self.normalize_pixels = QtWidgets.QCheckBox("Normalize pixels to 0-1")
        self.normalize_pixels.setToolTip("Standardize pixel values (usually not needed for WD14)")
        self.use_mcut = QtWidgets.QCheckBox("Use MCut thresholding")
        self.use_mcut.setToolTip("Automatic threshold detection (overrides manual thresholds)")
        self.include_scores = QtWidgets.QCheckBox("Show scores in caption")
        self.include_scores.setToolTip("Include confidence scores in exported captions")

        self.general_enabled = QtWidgets.QCheckBox("General tags")
        self.general_enabled.setChecked(True)
        self.general_enabled.setToolTip("Include general tags (clothing, background, etc.)")
        self.character_enabled = QtWidgets.QCheckBox("Character tags")
        self.character_enabled.setChecked(True)
        self.character_enabled.setToolTip("Include character name tags")

        category_widget = QtWidgets.QWidget()
        category_row = QtWidgets.QHBoxLayout(category_widget)
        category_row.setContentsMargins(0, 0, 0, 0)
        category_row.addWidget(self.general_enabled)
        category_row.addWidget(self.character_enabled)
        category_row.addStretch(1)

        self.blacklist = QtWidgets.QPlainTextEdit()
        self.blacklist.setPlaceholderText("blurry, lowres, bad_anatomy")
        self.blacklist.setFixedHeight(52)
        self.blacklist.setToolTip("Tags to always exclude (comma-separated)")

        self.whitelist = QtWidgets.QPlainTextEdit()
        self.whitelist.setPlaceholderText("1girl, solo (optional)")
        self.whitelist.setFixedHeight(52)
        self.whitelist.setToolTip("Only include these tags if specified (comma-separated)")

        self.caption_prefix = QtWidgets.QLineEdit()
        self.caption_prefix.setPlaceholderText("e.g. masterpiece, best_quality")
        self.caption_prefix.setToolTip("Tags prepended before every caption (e.g. quality booster tags)")
        self.caption_prefix.setClearButtonEnabled(True)

        self.caption_postfix = QtWidgets.QLineEdit()
        self.caption_postfix.setPlaceholderText("e.g. from_above, dutch_angle")
        self.caption_postfix.setToolTip("Tags appended after every caption (e.g. camera angle, style tags)")
        self.caption_postfix.setClearButtonEnabled(True)

        self.initial_caption = QtWidgets.QLineEdit()
        self.initial_caption.setPlaceholderText("e.g. 1girl, solo")
        self.initial_caption.setToolTip("Trigger words used as the caption when no tags are generated yet")
        self.initial_caption.setClearButtonEnabled(True)

        # ── Thresholds + max tags in one compact row ─────────────────────────
        thresholds_widget = QtWidgets.QWidget()
        thresholds_row = QtWidgets.QHBoxLayout(thresholds_widget)
        thresholds_row.setContentsMargins(0, 0, 0, 0)
        thresholds_row.setSpacing(10)

        gen_col = QtWidgets.QVBoxLayout()
        gen_col.setSpacing(2)
        gen_lbl = QtWidgets.QLabel("🟢 General")
        gen_lbl.setStyleSheet("color: #66ff66; font-size: 10px; font-weight: bold;")
        gen_lbl.setAlignment(QtCore.Qt.AlignCenter)
        gen_col.addWidget(gen_lbl)
        gen_col.addWidget(self.general_threshold)
        thresholds_row.addLayout(gen_col, 1)

        char_col = QtWidgets.QVBoxLayout()
        char_col.setSpacing(2)
        char_lbl = QtWidgets.QLabel("🩷 Character")
        char_lbl.setStyleSheet("color: #ff66a3; font-size: 10px; font-weight: bold;")
        char_lbl.setAlignment(QtCore.Qt.AlignCenter)
        char_col.addWidget(char_lbl)
        char_col.addWidget(self.character_threshold)
        thresholds_row.addLayout(char_col, 1)

        maxtags_col = QtWidgets.QVBoxLayout()
        maxtags_col.setSpacing(2)
        maxtags_lbl = QtWidgets.QLabel("🟡 Max tags")
        maxtags_lbl.setStyleSheet("color: #ffcc66; font-size: 10px; font-weight: bold;")
        maxtags_lbl.setAlignment(QtCore.Qt.AlignCenter)
        maxtags_col.addWidget(maxtags_lbl)
        maxtags_col.addWidget(self.max_tags)
        thresholds_row.addLayout(maxtags_col, 1)

        form.addRow(thresholds_widget)

        form.addRow(QtWidgets.QLabel("Sort by:"), self.sort_mode)
        form.addRow(QtWidgets.QLabel("Categories:"), category_widget)
        form.addRow(QtWidgets.QLabel("Blacklist:"), self.blacklist)
        form.addRow(QtWidgets.QLabel("Whitelist:"), self.whitelist)

        # Processing toggles share one row — the wide panel fits them side by
        # side, keeping the settings column short.
        options_widget = QtWidgets.QWidget()
        options_row = QtWidgets.QHBoxLayout(options_widget)
        options_row.setContentsMargins(0, 0, 0, 0)
        options_row.setSpacing(16)
        options_row.addWidget(self.normalize_pixels)
        options_row.addWidget(self.use_mcut)
        options_row.addWidget(self.include_scores)
        options_row.addStretch(1)
        form.addRow(QtWidgets.QLabel("Options:"), options_widget)
        right_splitter.addWidget(settings_group)

        table_group = QtWidgets.QGroupBox("🏷️ Tags")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["✓ Include", "Rank", "Tag", "Confidence", "Category"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "Uncheck 'Include' to exclude tags from caption\nEdit 'Rank' to change tag order"
        )
        self.table.itemChanged.connect(self.on_table_changed)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(2, 350)
        self.table.setColumnWidth(4, 90)
        table_layout.addWidget(self.table)
        right_splitter.addWidget(table_group)

        caption_group = QtWidgets.QGroupBox("📝 Generated Caption")
        caption_layout = QtWidgets.QVBoxLayout(caption_group)
        caption_layout.setSpacing(8)

        # ── Caption Wrapping collapsible section ──────────────────────────────
        self._wrapping_group = QtWidgets.QGroupBox("🔧 Caption Wrapping")
        self._wrapping_group.setCheckable(True)
        self._wrapping_group.setChecked(False)   # collapsed by default
        self._wrapping_group.setToolTip(
            "Wrap every exported caption with fixed prefix/postfix tags,\n"
            "and set trigger words used when no tags are generated."
        )
        wrapping_form = QtWidgets.QFormLayout(self._wrapping_group)
        wrapping_form.setSpacing(6)
        wrapping_form.setContentsMargins(10, 6, 10, 8)

        prefix_lbl = QtWidgets.QLabel("🟢 Prefix tags:")
        prefix_lbl.setToolTip("Added at the START of every caption")
        prefix_lbl.setStyleSheet("color: #66ff66; font-size: 11px;")
        wrapping_form.addRow(prefix_lbl, self.caption_prefix)

        postfix_lbl = QtWidgets.QLabel("🟠 Postfix tags:")
        postfix_lbl.setToolTip("Added at the END of every caption")
        postfix_lbl.setStyleSheet("color: #ff9966; font-size: 11px;")
        wrapping_form.addRow(postfix_lbl, self.caption_postfix)

        trigger_lbl = QtWidgets.QLabel("🔵 Trigger words:")
        trigger_lbl.setToolTip("Used as the caption when no tags have been generated yet")
        trigger_lbl.setStyleSheet("color: #66ccff; font-size: 11px;")
        wrapping_form.addRow(trigger_lbl, self.initial_caption)

        # Live preview bar — shows the assembled caption structure
        self.affix_preview_label = QtWidgets.QLabel()
        self.affix_preview_label.setStyleSheet(
            "color: #9ecbff; font-size: 10px; padding: 4px 6px; "
            "background: #0d0d0d; border: 1px solid #333; border-radius: 4px;"
        )
        self.affix_preview_label.setWordWrap(True)
        self.affix_preview_label.setVisible(False)
        wrapping_form.addRow(self.affix_preview_label)

        caption_layout.addWidget(self._wrapping_group)

        # Collapse/expand the inner content when the checkbox is toggled
        def _toggle_wrapping(checked: bool) -> None:
            self._wrapping_group.setFlat(not checked)
            for i in range(wrapping_form.rowCount()):
                item_label = wrapping_form.itemAt(i, QtWidgets.QFormLayout.LabelRole)
                item_field = wrapping_form.itemAt(i, QtWidgets.QFormLayout.FieldRole)
                item_span  = wrapping_form.itemAt(i, QtWidgets.QFormLayout.SpanningRole)
                for item in (item_label, item_field, item_span):
                    if item and item.widget():
                        item.widget().setVisible(checked)
        self._wrapping_group.toggled.connect(_toggle_wrapping)
        # Apply initial collapsed state
        _toggle_wrapping(False)

        self.caption_edit = QtWidgets.QPlainTextEdit()
        self.caption_edit.setPlaceholderText(
            "Tags will appear here as comma-separated values...\n"
            "Example: 1girl, smile, blue_eyes, long_hair"
        )
        self.caption_edit.setToolTip(
            "Edit caption text directly, then click 'Apply' to sync with table.\n"
            "Ctrl+Z / Ctrl+Y to undo/redo changes."
        )
        caption_layout.addWidget(self.caption_edit)

        self.apply_caption_btn = QtWidgets.QPushButton("🔄 Apply")
        self.apply_caption_btn.setToolTip("Update table from edited caption text")
        self.apply_caption_btn.clicked.connect(self.apply_caption_text)
        self.copy_prompt_btn = QtWidgets.QPushButton("📋 Copy Prompt")
        self.copy_prompt_btn.setObjectName("copyPromptBtn")
        self.copy_prompt_btn.setToolTip(
            "Copy current caption as a ComfyUI-compatible prompt string\n"
            "(underscores → spaces). Paste directly into ComfyUI."
        )
        self.copy_prompt_btn.clicked.connect(self._copy_as_prompt)
        self.neg_prompt_btn = QtWidgets.QPushButton("🚫 Negative")
        self.neg_prompt_btn.setObjectName("negPromptBtn")
        self.neg_prompt_btn.setToolTip(
            "Build a Negative prompt from excluded/blacklisted tags"
        )
        self.neg_prompt_btn.clicked.connect(self._build_negative_prompt)
        self.vlm_prompt_btn = QtWidgets.QPushButton("📝 Image → Prompt")
        self.vlm_prompt_btn.setObjectName("vlmPromptBtn")
        self.vlm_prompt_btn.setToolTip(
            "Describe the image itself with a vision model, in natural language,\n"
            "for Krea / Flux-style prompts.\n\n"
            "This bypasses the ONNX tagger entirely — useful for photographs,\n"
            "which the Danbooru-trained taggers handle poorly."
        )
        self.vlm_prompt_btn.clicked.connect(self._show_image_prompt_dialog)
        self.export_btn = QtWidgets.QPushButton("💾 Save")
        self.export_btn.setToolTip("Save caption for selected image as .txt file")
        self.export_btn.clicked.connect(self.export_caption)
        self.export_beside_btn = QtWidgets.QPushButton("💾 Save Beside")
        self.export_beside_btn.setToolTip(
            "Save all captions as .txt files next to their source images"
        )
        self.export_beside_btn.clicked.connect(self.export_beside_source)
        self.export_zip_btn = QtWidgets.QPushButton("📦 Export ZIP")
        self.export_zip_btn.setToolTip("Download all captions as ZIP file")
        self.export_zip_btn.clicked.connect(self.export_zip)
        self.tag_freq_btn = QtWidgets.QPushButton("📊 Tag Stats")
        self.tag_freq_btn.setObjectName("tagFreqBtn")
        self.tag_freq_btn.setToolTip("View tag frequency across all loaded results")
        self.tag_freq_btn.clicked.connect(self._show_tag_frequency)
        # Two-row caption toolbar. Eight buttons need ~1230px of natural
        # width; the caption pane is ~940px at the default window size, so a
        # single row could only fit by eliding labels ("Save Beside Sour").
        # Row 1 is caption/prompt actions, row 2 is save/export/analysis.
        def _separator() -> QtWidgets.QFrame:
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.VLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            line.setStyleSheet("color: #444;")
            line.setFixedWidth(2)
            return line

        toolbar_rows = QtWidgets.QVBoxLayout()
        toolbar_rows.setSpacing(4)

        # --- Row 1: caption editing and prompt building ---
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(0)
        row1.addWidget(self.apply_caption_btn)
        row1.addSpacing(8)
        row1.addWidget(_separator())
        row1.addSpacing(8)
        row1.addWidget(self.copy_prompt_btn)
        row1.addSpacing(4)
        row1.addWidget(self.neg_prompt_btn)
        row1.addSpacing(4)
        row1.addWidget(self.vlm_prompt_btn)
        row1.addStretch(1)
        toolbar_rows.addLayout(row1)

        # --- Row 2: saving, export and analysis ---
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(0)
        row2.addWidget(self.export_btn)
        row2.addSpacing(4)
        row2.addWidget(self.export_beside_btn)
        row2.addSpacing(4)
        row2.addWidget(self.export_zip_btn)
        row2.addSpacing(8)
        row2.addWidget(_separator())
        row2.addSpacing(8)
        row2.addWidget(self.tag_freq_btn)
        row2.addStretch(1)
        toolbar_rows.addLayout(row2)

        # Qt elides a button's label when the layout squeezes it below its
        # size hint, which is how "Save Beside Source" became "Save Beside
        # Sour". Pinning each button to its natural width makes that
        # impossible.
        #
        # Deferred to the first event-loop turn on purpose: a button's size
        # hint is not final during construction — neither at creation nor
        # after ensurePolished() — and reading it early under-sizes the
        # minimum by a few pixels, which is exactly enough to clip a trailing
        # character.
        self._toolbar_buttons = (
            self.apply_caption_btn, self.copy_prompt_btn, self.neg_prompt_btn,
            self.vlm_prompt_btn, self.export_btn, self.export_beside_btn,
            self.export_zip_btn, self.tag_freq_btn,
        )
        QtCore.QTimer.singleShot(0, self._pin_toolbar_button_widths)

        caption_layout.addLayout(toolbar_rows)
        right_splitter.addWidget(caption_group)
        right_splitter.setStretchFactor(0, 2)  # settings
        right_splitter.setStretchFactor(1, 4)  # tags table
        right_splitter.setStretchFactor(2, 3)  # caption
        # Explicit initial heights so the Tags table (primary output) gets a
        # fair share on launch instead of being squeezed by the tall settings
        # panel's size hint. Users can still drag the handles to rebalance.
        right_splitter.setSizes([420, 340, 220])

        self.caption_prefix.textChanged.connect(self._update_affix_preview)
        self.caption_postfix.textChanged.connect(self._update_affix_preview)
        self.initial_caption.textChanged.connect(self._update_affix_preview)

        self._boot("Loading tag vocabulary…")
        self.tabs.addTab(batch_tab, "Batch Tagger")

        # ===== TAB 2: Description Tagger =====
        desc_tab = QtWidgets.QWidget()
        desc_tab_outer = QtWidgets.QVBoxLayout(desc_tab)
        desc_tab_outer.setContentsMargins(4, 4, 4, 4)
        desc_tab_outer.setSpacing(0)

        # Two-column splitter: left = config, right = input + output
        desc_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        desc_splitter.setChildrenCollapsible(False)
        desc_splitter.setHandleWidth(6)
        desc_tab_outer.addWidget(desc_splitter)

        desc_left_widget = QtWidgets.QWidget()
        desc_left_layout = QtWidgets.QVBoxLayout(desc_left_widget)
        desc_left_layout.setContentsMargins(0, 0, 0, 0)
        desc_left_layout.setSpacing(8)

        desc_right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        desc_right_splitter.setChildrenCollapsible(False)
        desc_right_splitter.setHandleWidth(6)

        desc_splitter.addWidget(desc_left_widget)
        desc_splitter.addWidget(desc_right_splitter)
        desc_splitter.setStretchFactor(0, 2)
        desc_splitter.setStretchFactor(1, 3)
        desc_splitter.setSizes([560, 900])

        # `desc_layout` alias kept so existing `desc_layout.addWidget(...)` calls
        # below continue to target the left config column.
        desc_layout = desc_left_layout

        # --- Input mode + output format — shown first ---
        # Packed as one two-column row so the config column stays short enough
        # for a 1080p screen.
        input_mode_group = QtWidgets.QGroupBox("📥 Input & Output")
        input_mode_layout = QtWidgets.QGridLayout(input_mode_group)
        input_mode_layout.setSpacing(6)
        input_mode_layout.setColumnStretch(0, 1)
        input_mode_layout.setColumnStretch(1, 1)

        combo_css = "background-color: #1a1a1a; color: #ffffff; padding: 5px;"
        field_label_css = "color: #4da6ff; font-size: 10px; font-weight: bold;"

        input_label = QtWidgets.QLabel("Input:")
        input_label.setStyleSheet(field_label_css)
        input_mode_layout.addWidget(input_label, 0, 0)

        output_label = QtWidgets.QLabel("Output format:")
        output_label.setStyleSheet(field_label_css)
        input_mode_layout.addWidget(output_label, 0, 1)

        self.desc_input_mode = QtWidgets.QComboBox()
        self.desc_input_mode.setStyleSheet(combo_css)
        self.desc_input_mode.addItem("📝 From Description", "description")
        self.desc_input_mode.addItem("🏷️ From Seed Tags", "seed_tags")
        self.desc_input_mode.setCurrentIndex(0)
        self.desc_input_mode.currentIndexChanged.connect(self._on_input_mode_changed)
        input_mode_layout.addWidget(self.desc_input_mode, 1, 0)

        self.desc_output_format = QtWidgets.QComboBox()
        self.desc_output_format.setStyleSheet(combo_css)
        self.desc_output_format.addItem("🏷️ Danbooru Tags", "tags")
        self.desc_output_format.addItem("📝 Natural Language", "natural")
        self.desc_output_format.setCurrentIndex(0)
        self.desc_output_format.setToolTip(
            "Danbooru Tags: comma-separated tags for SDXL-anime models\n"
            "(Illustrious, NoobAI, Pony) via ComfyUI/A1111.\n\n"
            "Natural Language: a flowing prose paragraph for models that read\n"
            "plain English (Krea, Flux and similar). Runs in two stages —\n"
            "tags are generated first, then rewritten as prose, so the prompt\n"
            "inherits all the detail the tag pipeline adds."
        )
        self.desc_output_format.currentIndexChanged.connect(
            self._on_output_format_changed
        )
        input_mode_layout.addWidget(self.desc_output_format, 1, 1)

        self.desc_io_hint = QtWidgets.QLabel()
        self.desc_io_hint.setStyleSheet("color: #9ecbff; font-size: 10px; padding: 2px;")
        self.desc_io_hint.setWordWrap(True)
        input_mode_layout.addWidget(self.desc_io_hint, 2, 0, 1, 2)
        desc_layout.addWidget(input_mode_group)

        model_group = QtWidgets.QGroupBox("🤖 LLM Model Selection")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        model_label = QtWidgets.QLabel("Select AI Model for Tag Generation:")
        model_label.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        model_layout.addWidget(model_label)

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 5px;"
        )
        self.model_selector.addItem("(Loading models...)", None)
        model_layout.addWidget(self.model_selector)
        self._boot("Checking for local models…")
        self._refresh_available_models()

        # --- Model management buttons ---
        model_btn_row = QtWidgets.QHBoxLayout()
        model_btn_row.setSpacing(4)
        refresh_models_btn = QtWidgets.QPushButton("🔄 Refresh Models")
        refresh_models_btn.setToolTip("Check for newly installed Ollama models")
        refresh_models_btn.clicked.connect(self._refresh_available_models)
        model_btn_row.addWidget(refresh_models_btn)

        self.manage_models_btn = QtWidgets.QPushButton("⚙️ Manage Models")
        self.manage_models_btn.setToolTip("Pull new models, view installed models, or delete unused ones")
        self.manage_models_btn.clicked.connect(self._show_model_manager)
        model_btn_row.addWidget(self.manage_models_btn)
        model_layout.addLayout(model_btn_row)

        desc_layout.addWidget(model_group)

        creativity_group = QtWidgets.QGroupBox("🎨 Creativity Mode")
        creativity_layout = QtWidgets.QVBoxLayout(creativity_group)
        creativity_layout.setSpacing(8)

        creativity_label = QtWidgets.QLabel("How imaginative should the generated tags be?")
        creativity_label.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        creativity_layout.addWidget(creativity_label)

        self.creativity_selector = QtWidgets.QComboBox()
        self.creativity_selector.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 5px;"
        )
        self.creativity_selector.addItem("🛡️ Safe (literal, conservative)", "safe")
        self.creativity_selector.addItem("✨ Creative (balanced)", "creative")
        self.creativity_selector.addItem("🔞 Mature (explicit, nsfw)", "mature")
        self.creativity_selector.setCurrentIndex(1)
        creativity_layout.addWidget(self.creativity_selector)

        creativity_hint = QtWidgets.QLabel(
            "<b>Mode descriptions:</b><br>"
            "🛡️ <b>Safe:</b> Literal, conservative tags — no explicit content<br>"
            "✨ <b>Creative:</b> Balanced, richer scenes with context cues<br>"
            "🔞 <b>Mature:</b> Adult/explicit content (fellatio, sex, etc.)"
        )
        creativity_hint.setStyleSheet("color: #9ecbff; font-size: 10px; padding: 5px;")
        creativity_hint.setWordWrap(True)
        creativity_layout.addWidget(creativity_hint)

        desc_layout.addWidget(creativity_group)

        # --- Post-count threshold (advanced filter) ---
        threshold_group = QtWidgets.QGroupBox("🔧 Tag Quality Filter")
        threshold_layout = QtWidgets.QFormLayout(threshold_group)
        threshold_layout.setSpacing(6)

        self.post_count_threshold = QtWidgets.QSpinBox()
        self.post_count_threshold.setRange(0, 100000)
        self.post_count_threshold.setSingleStep(500)
        self.post_count_threshold.setValue(500)
        self.post_count_threshold.setToolTip(
            "Minimum post_count a tag must have on Danbooru to be included.\n"
            "Higher = only the most established tags. Lower = more variety.\n"
            "500 means a tag must appear in at least 500 Danbooru posts."
        )
        self.post_count_threshold.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addRow("Min Post Count:", self.post_count_threshold)

        threshold_hint = QtWidgets.QLabel(
            "Filters out rare/obscure tags. Set to 0 to disable filtering."
        )
        threshold_hint.setStyleSheet("color: #9ecbff; font-size: 10px; padding: 2px;")
        threshold_hint.setWordWrap(True)
        threshold_layout.addRow(threshold_hint)

        desc_layout.addWidget(threshold_group)

        # Add a stretch at the bottom of the left config column
        desc_left_layout.addStretch(1)

        self.input_desc_group = QtWidgets.QGroupBox("✍️ Input")
        input_desc_layout = QtWidgets.QVBoxLayout(self.input_desc_group)
        input_desc_layout.setSpacing(8)

        self.desc_hint_label = QtWidgets.QLabel(
            "Describe what you want to see, and AI will generate Danbooru tags:"
        )
        self.desc_hint_label.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        input_desc_layout.addWidget(self.desc_hint_label)

        self.description_input = QtWidgets.QPlainTextEdit()
        self.description_input.setPlaceholderText(
            "Examples:\n"
            "• A girl with long black hair and red eyes, wearing a maid outfit\n"
            "• Anime boy with blue eyes and blonde hair, holding a sword\n"
            "• Beautiful landscape with mountains and sunset in fantasy art style\n"
            "• Character with animal ears, tail, and wearing school uniform"
        )
        self.description_input.setMinimumHeight(140)
        self.description_input.setStyleSheet(
            "background-color: #0d0d0d; color: #ffffff; border: 1px solid #444; "
            "border-radius: 5px; padding: 8px;"
        )
        input_desc_layout.addWidget(self.description_input, 1)

        self.generate_from_desc_btn = QtWidgets.QPushButton("✨ Generate Tags from Description")
        self.generate_from_desc_btn.setObjectName("tagBtn")
        self.generate_from_desc_btn.setMinimumHeight(45)
        self.generate_from_desc_btn.setToolTip(
            "AI will analyze your description and generate matching Danbooru tags"
        )
        self.generate_from_desc_btn.setStyleSheet("""
            QPushButton#tagBtn {
                background-color: #0059b3;
                font-weight: bold;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton#tagBtn:hover {
                background-color: #0073e6;
            }
        """)
        self.generate_from_desc_btn.clicked.connect(self._generate_tags_from_description)
        input_desc_layout.addWidget(self.generate_from_desc_btn)

        desc_right_splitter.addWidget(self.input_desc_group)

        tags_group = QtWidgets.QGroupBox("🏷️ Generated Tags")
        self.desc_output_group = tags_group
        tags_layout = QtWidgets.QVBoxLayout(tags_group)

        self.desc_tags_display = QtWidgets.QPlainTextEdit()
        self.desc_tags_display.setReadOnly(True)
        # Keep Ctrl+A / Shift+arrow selection working on this read-only view.
        self.desc_tags_display.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        self.desc_tags_display.setPlaceholderText(
            "Generated Danbooru tags will appear here after processing..."
        )
        self.desc_tags_display.setStyleSheet(
            "background-color: #0d0d0d; color: #66ff66; font-family: monospace; "
            "font-size: 11px; border-radius: 5px; padding: 8px;"
        )
        self.desc_tags_display.setMinimumHeight(80)
        tags_layout.addWidget(self.desc_tags_display, 1)

        copy_row = QtWidgets.QHBoxLayout()
        copy_row.setSpacing(4)

        copy_tags_btn = QtWidgets.QPushButton("📋 Copy Tags to Clipboard")
        copy_tags_btn.setToolTip("Copy generated tags as comma-separated list")
        copy_tags_btn.setStyleSheet(
            "background-color: #ff9933; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        copy_tags_btn.clicked.connect(self._copy_description_tags)
        self._copy_tags_btn = copy_tags_btn
        copy_row.addWidget(copy_tags_btn, 2)

        # Only meaningful in natural-language mode, where the run also produces
        # an intermediate tag set worth keeping.
        self._copy_source_tags_btn = QtWidgets.QPushButton("🏷️ Copy Source Tags")
        self._copy_source_tags_btn.setToolTip(
            "Copy the Danbooru tags the prompt was written from"
        )
        self._copy_source_tags_btn.setStyleSheet(
            "background-color: #444; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        self._copy_source_tags_btn.clicked.connect(self._copy_prompt_source_tags)
        self._copy_source_tags_btn.setVisible(False)
        copy_row.addWidget(self._copy_source_tags_btn, 1)

        tags_layout.addLayout(copy_row)

        desc_right_splitter.addWidget(tags_group)
        desc_right_splitter.setStretchFactor(0, 3)  # input
        desc_right_splitter.setStretchFactor(1, 2)  # generated tags

        # Sync every input/output-dependent label now that the whole tab exists.
        self._refresh_desc_io_labels()

        self._boot("Finishing up…")
        self.tabs.addTab(desc_tab, "Description Tagger")

        # --- Status bar --------------------------------------------------------
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready.")

        self.help_btn = QtWidgets.QPushButton("❓ Help")
        self.help_btn.setMaximumHeight(25)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3b3b3b;
                border: 1px solid #4da6ff;
            }
        """)
        self.help_btn.clicked.connect(self.show_help)
        self.statusbar.addPermanentWidget(self.help_btn)

        self.about_btn = QtWidgets.QPushButton("ℹ️ About")
        self.about_btn.setMaximumHeight(25)
        self.about_btn.setToolTip("Version, licence, terms of use and third-party notices")
        self.about_btn.setStyleSheet(self.help_btn.styleSheet())
        self.about_btn.clicked.connect(self.show_about)
        self.statusbar.addPermanentWidget(self.about_btn)

        self.kofi_btn = QtWidgets.QPushButton("☕ Support")
        self.kofi_btn.setMaximumHeight(25)
        self.kofi_btn.setToolTip("Support development on Ko-fi")
        self.kofi_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff5e5b;
                color: #ffffff;
                border: 1px solid #ff5e5b;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff7875;
                border: 1px solid #ffffff;
            }
        """)
        self.kofi_btn.clicked.connect(self._open_kofi)
        self.statusbar.addPermanentWidget(self.kofi_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(250)
        self.progress.setFormat("Processing: %p%")
        self.statusbar.addPermanentWidget(self.progress)

        self._set_export_enabled(False)

        # Setup caption completer
        self._setup_caption_completer()

        # Show the drag-and-drop hint in the empty list on startup
        QtCore.QTimer.singleShot(0, self._update_list_placeholder)

        # --- Drop overlay ------------------------------------------------------
        self._drop_overlay = QtWidgets.QLabel(self)
        self._drop_overlay.setText("📥\n\nDrop images here\nto upload")
        self._drop_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self._drop_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 210);
                color: #4da6ff;
                font-size: 28px;
                font-weight: bold;
                border: 3px dashed #4da6ff;
                border-radius: 20px;
                padding: 40px;
            }
        """)
        self._drop_overlay.setVisible(False)

        self._loading_overlay = QtWidgets.QFrame(self)
        self._loading_overlay.setObjectName("loadingOverlay")
        self._loading_overlay.setVisible(False)
        loading_layout = QtWidgets.QVBoxLayout(self._loading_overlay)
        loading_layout.setAlignment(QtCore.Qt.AlignCenter)
        loading_spinner = QtWidgets.QLabel("\u23f3")
        loading_spinner.setStyleSheet("font-size: 48px; color: #4da6ff; background: transparent;")
        loading_spinner.setAlignment(QtCore.Qt.AlignCenter)
        loading_label = QtWidgets.QLabel("Loading images...")
        loading_label.setStyleSheet(
            "font-size: 16px; color: #4da6ff; font-weight: bold; background: transparent;"
        )
        loading_label.setAlignment(QtCore.Qt.AlignCenter)
        self._loading_detail = QtWidgets.QLabel("")
        self._loading_detail.setStyleSheet(
            "font-size: 12px; color: #9ecbff; background: transparent;"
        )
        self._loading_detail.setAlignment(QtCore.Qt.AlignCenter)
        self._loading_progress = QtWidgets.QProgressBar()
        self._loading_progress.setMaximumWidth(300)
        self._loading_progress.setFormat("Validating: %p%")
        self._loading_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #4da6ff; border-radius: 4px; "
            "background-color: #0d0d0d; text-align: center; color: white; }"
            "QProgressBar::chunk { background-color: #0059b3; border-radius: 3px; }"
        )
        loading_layout.addWidget(loading_spinner)
        loading_layout.addWidget(loading_label)
        loading_layout.addWidget(self._loading_detail)
        loading_layout.addSpacing(10)
        loading_layout.addWidget(self._loading_progress, 0, QtCore.Qt.AlignCenter)
        self._loading_overlay.setStyleSheet("""
            QFrame#loadingOverlay {
                background-color: rgba(0, 0, 0, 210);
                border-radius: 20px;
            }
        """)

        paste_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Paste, self)
        paste_shortcut.activated.connect(self._handle_paste)

        # --- Undo / Redo -------------------------------------------------------
        self._undo_stack: list[tuple[pd.DataFrame, str]] = []
        self._redo_stack: list[tuple[pd.DataFrame, str]] = []
        undo_action = QtGui.QAction("Undo", self)
        undo_action.setShortcut(QtGui.QKeySequence.Undo)
        undo_action.triggered.connect(self._undo)
        self.addAction(undo_action)
        redo_action = QtGui.QAction("Redo", self)
        redo_action.setShortcut(QtGui.QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self._redo)
        self.addAction(redo_action)

    # ==================================================================
    # event filter (overrides QMainWindow)
    # ==================================================================

    def eventFilter(self, obj, event) -> bool:
        """Route caption-completer events before default handling."""
        if self._completer_event_filter(obj, event):
            return True
        return super().eventFilter(obj, event)

    # ==================================================================
    # Help / About
    # ==================================================================

    def show_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def _open_kofi(self) -> None:
        """Open the Ko-fi donation page in the default browser."""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/saekimon"))

    def show_about(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About Img-Tagbooru")
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText(
            f"<b>Img-Tagbooru v{APP_VERSION}</b><br>"
            "Local Danbooru-style image tagger for anime images and LoRA training.<br><br>"
            "© 2026 Szymon Ruszkiewicz — released under the <b>MIT License</b>.<br>"
            "Provided \"as is\", without warranty of any kind.<br><br>"
            "<b>Privacy:</b> no telemetry, no accounts, no uploads. Everything runs on "
            "this computer; the only network access is model downloads from "
            "Hugging Face, the GitHub update check, and any image URL you paste.<br><br>"
            "<b>Models:</b> WD taggers by SmilingWolf (Apache 2.0). Language and "
            "vision models are installed by you through Ollama under their own "
            "licences. Tag vocabulary: Danbooru (factual tag data).<br><br>"
            "Built with Python, PySide6 / Qt (LGPL-3.0) and ONNX Runtime.<br><br>"
            "☕ <a href='https://ko-fi.com/saekimon'>ko-fi.com/saekimon</a> &nbsp; "
            "🐙 <a href='https://github.com/Xymoh/img-tagboru-ai'>github.com/Xymoh/img-tagboru-ai</a>"
        )
        terms_btn = box.addButton("Terms of Use", QtWidgets.QMessageBox.ActionRole)
        notices_btn = box.addButton("Third-party notices", QtWidgets.QMessageBox.ActionRole)
        license_btn = box.addButton("MIT License", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is terms_btn:
            TextViewerDialog(
                "Terms of Use — Img-Tagbooru",
                read_bundled_text("TERMS.md", "TERMS.md not found. See the GitHub repository."),
                self,
            ).exec()
        elif clicked is notices_btn:
            TextViewerDialog(
                "Third-party notices — Img-Tagbooru",
                read_bundled_text("THIRD_PARTY_NOTICES.txt", "THIRD_PARTY_NOTICES.txt not found."),
                self,
                markdown=False,
            ).exec()
        elif clicked is license_btn:
            TextViewerDialog(
                "MIT License — Img-Tagbooru",
                read_bundled_text("LICENSE", "LICENSE not found."),
                self,
                markdown=False,
            ).exec()

    # ==================================================================
    # UI helpers
    # ==================================================================

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)
        self.export_beside_btn.setEnabled(enabled)
        self.export_zip_btn.setEnabled(enabled)

    def _update_list_placeholder(self) -> None:
        """Show the drag-and-drop hint inside the list when it is empty."""
        empty = self.result_list.count() == 0
        self._list_placeholder.setVisible(empty)
        if empty:
            # Centre the label inside the list viewport
            vp = self.result_list.viewport()
            self._list_placeholder.setGeometry(vp.rect())
        if hasattr(self, "pending_label"):
            count = self.result_list.count()
            if count:
                self.pending_label.setText(f"Loaded {count} image(s).")
                self.pending_label.setVisible(True)
            else:
                self.pending_label.setVisible(False)

    def _selected_categories(self) -> set[str]:
        selected: set[str] = set()
        if self.general_enabled.isChecked():
            selected.add("general")
        if self.character_enabled.isChecked():
            selected.add("character")
        return selected

    # ==================================================================
    # Image / path loading
    # ==================================================================

    def _open_image_safe(self, path: Path) -> Image.Image | None:
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

    def _show_loading_overlay(self, detail: str = "") -> None:
        if hasattr(self, '_loading_overlay'):
            rect = self.centralWidget().rect()
            self._loading_overlay.setGeometry(rect)
            self._loading_detail.setText(detail)
            self._loading_progress.setValue(0)
            self._loading_overlay.setVisible(True)
            self._loading_overlay.raise_()

    def _hide_loading_overlay(self) -> None:
        if hasattr(self, '_loading_overlay'):
            self._loading_overlay.setVisible(False)

    def _load_paths(self, paths: Sequence[Path]) -> None:
        if self._image_load_worker is not None:
            self._image_load_worker.progress.disconnect()
            self._image_load_worker.finished.disconnect()
            self._image_load_worker.quit()
            self._image_load_worker.wait(1000)
            self._image_load_worker = None

        self._show_loading_overlay(f"Validating {len(paths)} file(s)...")
        self._hide_drop_overlay()

        self._image_load_worker = ImageLoadWorker(list(paths))
        self._image_load_worker.progress.connect(self._loading_progress.setValue)
        self._image_load_worker.finished.connect(self._on_images_loaded)
        self._image_load_worker.start()

    def _on_images_loaded(self, valid_paths: list[Path], skipped: list[str]) -> None:
        self._hide_loading_overlay()
        if self._image_load_worker is not None:
            self._image_load_worker.wait()
            self._image_load_worker = None

        self.results = []
        self._single_results = {}
        self._active_result_index = -1
        self.table.setRowCount(0)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText("")
        self.caption_edit.blockSignals(False)
        self._set_export_enabled(False)

        self.pending_paths = valid_paths
        self.pending_label.setText(f"Loaded {len(self.pending_paths)} image(s).")
        self.tag_btn.setEnabled(bool(self.pending_paths))

        self.result_list.blockSignals(True)
        self.result_list.clear()
        for p in self.pending_paths:
            self.result_list.addItem(p.name)
        self.result_list.blockSignals(False)
        self._update_list_placeholder()
        if self.pending_paths:
            self.result_list.setCurrentRow(0)
            self.show_result(0)
        elif skipped:
            self.statusbar.showMessage("No valid images found in drop/paste input.", 7000)

    def open_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if paths:
            self._load_paths([Path(path) for path in paths])

    # ==================================================================
    # Drag & drop
    # ==================================================================

    def _show_drop_overlay(self) -> None:
        """Position and show the drop overlay to cover the central widget."""
        if hasattr(self, '_drop_overlay'):
            rect = self.centralWidget().rect()
            self._drop_overlay.setGeometry(rect)
            self._drop_overlay.setVisible(True)
            self._drop_overlay.raise_()

    def _hide_drop_overlay(self) -> None:
        """Hide the drop overlay."""
        if hasattr(self, '_drop_overlay'):
            self._drop_overlay.setVisible(False)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if (
            mime_data.hasUrls()
            or mime_data.hasImage()
            or mime_data.hasHtml()
            or mime_data.hasText()
            or any(fmt.startswith("image/") for fmt in mime_data.formats())
            or mime_data.hasFormat("application/octet-stream")
        ):
            event.acceptProposedAction()
            self._show_drop_overlay()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self._hide_drop_overlay()
        super().dragLeaveEvent(event)

    def _save_qimage_temp(self, image: QtGui.QImage, prefix: str) -> Path | None:
        if image.isNull():
            return None
        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-clipboard"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{prefix}_{uuid4().hex[:8]}.png"
        if image.save(str(temp_path), "PNG"):
            return temp_path
        return None

    @staticmethod
    def _origin_referer(url: str) -> str:
        """Derive a plausible Referer from *url*'s origin (scheme + host)."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/"

    # Maximum bytes to download for a single web image (100 MiB safety cap).
    _MAX_WEB_IMAGE_BYTES: int = 100 * 1024 * 1024

    def _download_web_image_to_temp(self, url: str) -> Path | None:
        """Download a web image to a temp file. Shows loading overlay while downloading."""
        self._show_loading_overlay(f"Downloading web image…\n{url}")
        QtCore.QCoreApplication.processEvents()
        try:
            logger.debug("Starting download of: %s", url)

            referer = self._origin_referer(url)
            headers_variants = [
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": referer,
                    "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Sec-Fetch-Dest": "image",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "same-site",
                    "Cache-Control": "max-age=0",
                },
                {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                {},
            ]

            for attempt, headers in enumerate(headers_variants):
                try:
                    logger.debug("Attempt %d with headers: %s", attempt + 1, list(headers.keys()))
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        # --- Content-Length safety check -----------------------
                        content_length_raw = response.headers.get("Content-Length")
                        if content_length_raw is not None:
                            try:
                                content_length = int(content_length_raw)
                            except ValueError:
                                content_length = -1
                            if content_length > self._MAX_WEB_IMAGE_BYTES:
                                logger.warning(
                                    "Rejected URL (Content-Length %d > %d max): %s",
                                    content_length,
                                    self._MAX_WEB_IMAGE_BYTES,
                                    url,
                                )
                                continue
                        # ------------------------------------------------------
                        content_type = (response.headers.get("Content-Type") or "").lower()
                        data = response.read()
                        logger.debug("Downloaded %d bytes, Content-Type: %s", len(data), content_type)

                        if (
                            content_type
                            and content_type not in ("application/octet-stream", "")
                            and "image/" not in content_type
                        ):
                            logger.debug("Rejected due to Content-Type: %s", content_type)
                            continue

                        try:
                            img = Image.open(io.BytesIO(data))
                            img.load()
                            logger.debug("PIL validation successful, format: %s", img.format)
                        except (UnidentifiedImageError, OSError) as e:
                            logger.debug("PIL validation failed: %s", e)
                            continue

                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        ext = (img.format or "jpg").lower() if hasattr(img, "format") else "jpg"
                        ext = ext if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif") else "jpg"
                        temp_path = temp_dir / f"web_{uuid4().hex[:8]}.{ext}"
                        temp_path.write_bytes(data)
                        logger.debug("Saved to %s", temp_path)
                        self._hide_loading_overlay()
                        return temp_path
                except urllib.error.HTTPError as e:
                    logger.debug("HTTP Error %s on attempt %d", e.code, attempt + 1)
                    if attempt == len(headers_variants) - 1:
                        raise
                    continue
                except Exception as e:
                    logger.debug("Error on attempt %d: %s", attempt + 1, e)
                    if attempt == len(headers_variants) - 1:
                        raise
                    continue
        except Exception as e:
            logger.debug("Download failed with exception: %s: %s", type(e).__name__, e)
            self._hide_loading_overlay()
            return None

    _PAGE_URL_PATTERNS = [
        re.compile(p) for p in [
            r'index\.php\?.*page=post',
            r'/post/show/',
            r'/posts/\d+/?$',
            r'\.php\?',
        ]
    ]
    _IMAGE_EXT_PATTERN = re.compile(r'\.(png|jpe?g|webp|bmp|gif)(\?.*)?$', re.IGNORECASE)

    @classmethod
    def _is_image_url(cls, url: str) -> bool:
        """Return True if *url* looks like a direct image URL."""
        return bool(cls._IMAGE_EXT_PATTERN.search(url))

    @classmethod
    def _is_page_url(cls, url: str) -> bool:
        """Return True if *url* looks like an HTML page, not an image."""
        return any(pat.search(url) for pat in cls._PAGE_URL_PATTERNS)

    def _extract_web_image_candidates(self, mime_data: QtCore.QMimeData) -> list[str]:
        candidates: list[str] = []

        if mime_data.hasUrls():
            for url in mime_data.urls():
                if url.scheme() in ("http", "https"):
                    candidates.append(url.toString())

        if mime_data.hasHtml():
            html = mime_data.html()
            src_matches = re.findall(r'src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
            # Also capture href attributes of <a> tags wrapping images
            href_matches = re.findall(
                r'<a\b[^>]*href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE
            )
            base_match = re.search(
                r'<base[^>]*href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE
            )
            base_url = base_match.group(1) if base_match else ""

            for src in src_matches:
                if src.startswith(("http://", "https://")):
                    candidates.append(src)
                elif base_url:
                    candidates.append(urljoin(base_url, src))

            for href in href_matches:
                if href.startswith(("http://", "https://")):
                    candidates.append(href)
                elif base_url:
                    candidates.append(urljoin(base_url, href))

        if mime_data.hasText():
            text = mime_data.text().strip()
            if text.startswith(("http://", "https://")):
                candidates.append(text)

        # Separate image-looking URLs from page-looking URLs
        image_urls: list[str] = []
        page_urls: list[str] = []
        other_urls: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            if self._is_image_url(item):
                image_urls.append(item)
            elif self._is_page_url(item):
                page_urls.append(item)
            else:
                other_urls.append(item)

        # Image URLs first, then ambiguous others, page URLs last
        unique = image_urls + other_urls + page_urls
        logger.debug("URL candidates: %d total (image=%d, other=%d, page=%d)",
                     len(unique), len(image_urls), len(other_urls), len(page_urls))
        return unique

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        self._hide_drop_overlay()
        mime_data = event.mimeData()

        # 1) Local files/folders
        if mime_data.hasUrls():
            urls = mime_data.urls()
            local_paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
            if local_paths:
                files: list[Path] = []
                for path in local_paths:
                    if path.is_dir():
                        files.extend(
                            [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
                        )
                    else:
                        files.append(path)
                if files:
                    self._load_paths(files)
                    self.statusbar.showMessage(f"Loaded {len(files)} dropped file(s).", 5000)
                    event.acceptProposedAction()
                    return

        # 2) Qt native image
        if mime_data.hasImage():
            logger.debug("Attempting to use Qt's native image support via hasImage()")
            try:
                image = QtGui.QImage(mime_data.imageData())
                if not image.isNull():
                    logger.debug("Successfully got QImage from mime_data")
                    temp_path = self._save_qimage_temp(image, "dragged")
                    if temp_path is not None:
                        logger.debug("Saved QImage to %s", temp_path)
                        self._load_paths([temp_path])
                        self.statusbar.showMessage("Loaded dropped image.", 5000)
                        event.acceptProposedAction()
                        return
            except Exception as e:
                logger.debug("Qt image handling failed: %s", e)

        # 3) Raw image data from any MIME type
        logger.debug("Available MIME formats: %s", mime_data.formats())
        for fmt in mime_data.formats():
            logger.debug("Trying to extract image from format: %s", fmt)
            try:
                image_bytes = mime_data.data(fmt)
                if image_bytes and len(image_bytes) > 500:
                    logger.debug("Found %d bytes in format %s", len(image_bytes), fmt)

                    logger.debug("Attempting QImage.fromData() on %s", fmt)
                    qt_image = QtGui.QImage.fromData(image_bytes)
                    if not qt_image.isNull():
                        logger.debug("QImage.fromData() succeeded for %s", fmt)
                        temp_path = self._save_qimage_temp(qt_image, "dragged")
                        if temp_path is not None:
                            logger.debug("Saved QImage to %s", temp_path)
                            self._load_paths([temp_path])
                            self.statusbar.showMessage("Loaded dropped image.", 5000)
                            event.acceptProposedAction()
                            return

                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        img.load()
                        logger.debug("Successfully parsed as %s", img.format)

                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        ext = (img.format or "jpg").lower() if hasattr(img, "format") else "jpg"
                        ext = ext if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif") else "jpg"
                        temp_path = temp_dir / f"web_{uuid4().hex[:8]}.{ext}"
                        temp_path.write_bytes(image_bytes)
                        logger.debug("Saved extracted image to %s", temp_path)
                        self._load_paths([temp_path])
                        self.statusbar.showMessage("Loaded dropped image.", 5000)
                        event.acceptProposedAction()
                        return
                    except Exception as e:
                        logger.debug("Failed to parse %s as PIL image: %s", fmt, e)
                        continue
            except Exception as e:
                logger.debug("Error extracting %s: %s", fmt, e)
                continue

        # 4) Web URLs
        candidates = list(self._extract_web_image_candidates(mime_data))
        if candidates:
            logger.debug("Found web image candidates: %s", candidates)
        for url in candidates:
            logger.debug("Downloading web image from URL: %s", url)
            downloaded = self._download_web_image_to_temp(url)
            if downloaded and downloaded.suffix.lower() in IMAGE_EXTENSIONS | {".png"}:
                logger.debug("Successfully downloaded to %s", downloaded)
                self._load_paths([downloaded])
                self.statusbar.showMessage("Loaded dropped image from web URL.", 5000)
                event.acceptProposedAction()
                return
            else:
                self.statusbar.showMessage(
                    f"⚠️ Failed to download image from URL. The server may have rejected the request.",
                    7000,
                )
                event.acceptProposedAction()
                return

        all_formats = mime_data.formats()
        logger.debug("Drop not recognized. Available MIME types: %s", all_formats)
        self.statusbar.showMessage(
            "Drop not recognized. Try: drag image file, drag from website, or Ctrl+V.",
            7000,
        )
        super().dropEvent(event)

    # ==================================================================
    # Keyboard / paste
    # ==================================================================

    def _handle_paste(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasUrls():
            paths = [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()]
            if paths:
                self._load_paths(paths)
                self.statusbar.showMessage("Loaded pasted file path(s).", 5000)
                return

        for url in self._extract_web_image_candidates(mime_data):
            downloaded = self._download_web_image_to_temp(url)
            if downloaded is not None:
                self._load_paths([downloaded])
                self.statusbar.showMessage("Loaded pasted web image URL.", 5000)
                return
            else:
                self.statusbar.showMessage(
                    f"⚠️ Failed to download image from URL. The server may have rejected the request.",
                    7000,
                )
                return

        image = clipboard.image()
        temp_path = self._save_qimage_temp(image, "clipboard")
        if temp_path is not None:
            self._load_paths([temp_path])
            self.statusbar.showMessage("Loaded pasted image data.", 5000)
            return

        self.statusbar.showMessage(
            "Clipboard does not contain an image. Copy an image or an image URL and press Ctrl+V.",
            7000,
        )

    # ==================================================================
    # Folder picker
    # ==================================================================

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
        if not folder:
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Include Subfolders?")
        msg_box.setText("Do you want to include images from subfolders?")
        msg_box.setInformativeText(f"Selected folder:\n{folder}")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Cancel:
            return

        root = Path(folder)
        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            paths = [
                path
                for path in sorted(root.rglob("*"))
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            paths = [
                path
                for path in sorted(root.glob("*"))
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]

        self._load_paths(paths)

    # ==================================================================
    # Tagging (core)
    # ==================================================================

    def _tag_image(self, image: Image.Image) -> list:
        tagger = get_tagger()
        return predict_tags(
            tagger,
            image,
            general_threshold=self.general_threshold.value(),
            character_threshold=self.character_threshold.value(),
            normalize_pixels=self.normalize_pixels.isChecked(),
            use_mcut=self.use_mcut.isChecked(),
            limit=self.max_tags.value(),
        )

    def process_pending(self) -> None:
        if not self.pending_paths:
            return

        self.results = []
        self._active_result_index = -1
        self.table.blockSignals(True)
        self.result_list.blockSignals(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusbar.showMessage("Tagging images...")

        blacklist = split_tags(self.blacklist.toPlainText())
        whitelist = split_tags(self.whitelist.toPlainText())
        allowed_categories = self._selected_categories()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            total = len(self.pending_paths)
            for index, path in enumerate(self.pending_paths, start=1):
                image = self._open_image_safe(path)
                if image is None:
                    self.progress.setValue(int(index / max(1, total) * 100))
                    QtWidgets.QApplication.processEvents()
                    continue
                predictions = self._tag_image(image)
                frame = frame_from_predictions(predictions)
                if not frame.empty:
                    if allowed_categories:
                        frame = frame[frame["category"].isin(allowed_categories)].copy()
                    frame = apply_filters(frame, blacklist, whitelist)
                    frame = sort_frame(frame, self.sort_mode.currentText())
                caption = frame_to_caption(frame, include_scores=self.include_scores.isChecked())
                self.results.append(
                    TaggingResult(name=path.name, path=path, image=image, frame=frame, caption=caption)
                )
                try:
                    item = self.result_list.item(index - 1)
                    if item is not None:
                        item.setText(path.name)
                except Exception:
                    pass
                self.progress.setValue(int(index / max(1, total) * 100))
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setVisible(False)
            self.table.blockSignals(False)
            self.result_list.blockSignals(False)

        self._set_export_enabled(bool(self.results))
        self.statusbar.showMessage(f"Tagged {len(self.results)} image(s).")
        if self.results:
            current_row = self.result_list.currentRow()
            if current_row >= 0:
                self.show_result(current_row)
            else:
                self.result_list.setCurrentRow(0)

    def _update_tag_selected_button(self) -> None:
        index = self.result_list.currentRow()
        has_pending = 0 <= index < len(self.pending_paths)
        has_result = 0 <= index < len(self.results)
        self.tag_selected_btn.setEnabled(has_pending or has_result)
        self.ai_meta_btn.setEnabled(has_pending or has_result)

    def process_single_image(self) -> None:
        index = self.result_list.currentRow()

        if index < 0:
            self.statusbar.showMessage("No image selected.", 3000)
            return

        if 0 <= index < len(self.results):
            existing = self.results[index]
            path = existing.path
            image = existing.image
        elif index in self._single_results:
            existing = self._single_results[index]
            path = existing.path
            image = existing.image
        elif 0 <= index < len(self.pending_paths):
            path = self.pending_paths[index]
            image = self._open_image_safe(path)
            if image is None:
                self.statusbar.showMessage(f"Cannot load image: {path.name}", 3000)
                return
        else:
            self.statusbar.showMessage("No image selected.", 3000)
            return

        self.statusbar.showMessage(f"Tagging {path.name}...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            predictions = self._tag_image(image)
            frame = frame_from_predictions(predictions)
            if not frame.empty:
                allowed_categories = self._selected_categories()
                if allowed_categories:
                    frame = frame[frame["category"].isin(allowed_categories)].copy()
                blacklist = split_tags(self.blacklist.toPlainText())
                whitelist = split_tags(self.whitelist.toPlainText())
                frame = apply_filters(frame, blacklist, whitelist)
                frame = sort_frame(frame, self.sort_mode.currentText())
            else:
                frame = pd.DataFrame(
                    columns=["include", "rank", "tag", "confidence", "category"]
                )
            caption = frame_to_caption(frame, include_scores=self.include_scores.isChecked())

            new_result = TaggingResult(
                name=path.name, path=path, image=image, frame=frame, caption=caption
            )
            if 0 <= index < len(self.results):
                self.results[index] = new_result
            else:
                self._single_results[index] = new_result

            self._active_result_index = index
            self._frame_to_table(frame)
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(caption)
            self.caption_edit.blockSignals(False)
            self._set_export_enabled(True)

            tag_count = len(frame) if not frame.empty else 0
            self.statusbar.showMessage(f"Tagged {path.name} ({tag_count} tags)")
        except Exception as e:
            self.statusbar.showMessage(f"Error tagging: {str(e)}", 5000)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _extract_ai_metadata_for_current(self) -> None:
        """Read AI generation parameters from the currently selected image."""
        result = self._current_result()
        image = None
        if result is not None:
            image = result.image
        else:
            index = self.result_list.currentRow()
            if 0 <= index < len(self.pending_paths):
                image = self._open_image_safe(self.pending_paths[index])

        if image is None:
            self.statusbar.showMessage("No image selected.", 3000)
            return

        self.statusbar.showMessage("Extracting AI metadata…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            params = extract_ai_metadata(image)
            if not params:
                self.statusbar.showMessage(
                    "No AI generation metadata found in this image.", 5000
                )
                return

            tags = metadata_to_tags(params)
            if not tags:
                self.statusbar.showMessage(
                    "AI metadata found but no parseable generation parameters.", 5000
                )
                return

            # Import pandas for DataFrame construction
            caption_text = ", ".join(tags)
            # Build a frame with these metadata tags
            rows: list[dict] = []
            for idx, tag in enumerate(tags):
                rows.append({
                    "include": True,
                    "rank": idx + 1,
                    "tag": tag,
                    "confidence": 0.0,
                    "category": "ai_metadata",
                })
            frame = pd.DataFrame(rows)

            index = self.result_list.currentRow()
            new_result = TaggingResult(
                name=(result.name if result else self.pending_paths[index].name),
                path=(result.path if result else self.pending_paths[index]),
                image=image,
                frame=frame,
                caption=caption_text,
            )
            if 0 <= index < len(self.results):
                self.results[index] = new_result
            else:
                self._single_results[index] = new_result

            self._active_result_index = index
            self._frame_to_table(frame)
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(caption_text)
            self.caption_edit.blockSignals(False)
            self._set_export_enabled(True)

            self.statusbar.showMessage(
                f"✓ Extracted {len(tags)} AI metadata tags from image.", 5000
            )
        except Exception as e:
            self.statusbar.showMessage(f"Error extracting AI metadata: {str(e)}", 5000)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ==================================================================
    # Result selection / navigation
    # ==================================================================

    def _current_index(self) -> int:
        return self.result_list.currentRow()

    def _current_result(self) -> TaggingResult | None:
        index = self._current_index()
        if 0 <= index < len(self.results):
            return self.results[index]
        if index in self._single_results:
            return self._single_results[index]
        return None

    def _strip_affixes(self, text: str) -> str:
        """Remove prefix and postfix from caption text to recover the base caption."""
        prefix = self.caption_prefix.text().strip()
        postfix = self.caption_postfix.text().strip()
        result = text.strip()
        if postfix and result.endswith(", " + postfix):
            result = result[:-(len(", " + postfix))]
        elif postfix and result == postfix:
            result = ""
        if prefix and result.startswith(prefix + ", "):
            result = result[len(prefix + ", "):]
        elif prefix and result == prefix:
            result = ""
        return result

    def _sync_current_result(self) -> None:
        result = self._current_result()
        if result is None:
            return
        result.frame = self._table_to_frame()
        result.caption = self._strip_affixes(self.caption_edit.toPlainText())

    def _effective_caption(self, caption: str | None = None) -> str:
        """Return caption with prefix and postfix applied (or initial caption if empty)."""
        prefix = self.caption_prefix.text().strip()
        postfix = self.caption_postfix.text().strip()
        initial = self.initial_caption.text().strip()

        if caption is None:
            result = self._current_result()
            caption = result.caption if result else ""

        base = caption.strip()
        if not base:
            base = initial

        parts = [p for p in (prefix, base, postfix) if p]
        return ", ".join(parts)

    def _update_affix_preview(self) -> None:
        """Refresh the affix preview label inside the Caption Wrapping group."""
        prefix = self.caption_prefix.text().strip()
        postfix = self.caption_postfix.text().strip()
        initial = self.initial_caption.text().strip()

        parts: list[str] = []
        if prefix:
            parts.append(f"<span style='color:#66ff66'><b>[{prefix}]</b></span>")
        parts.append("<span style='color:#888'>… tags …</span>")
        if postfix:
            parts.append(f"<span style='color:#ff9966'><b>[{postfix}]</b></span>")

        preview_html = " → ".join(parts)
        if initial:
            preview_html += (
                f"&nbsp;&nbsp;<span style='color:#aaa'>|</span>&nbsp;&nbsp;"
                f"<span style='color:#66ccff'>Trigger: <b>{initial}</b></span>"
            )

        if prefix or postfix or initial:
            self.affix_preview_label.setText(preview_html)
            self.affix_preview_label.setVisible(True)
        else:
            self.affix_preview_label.setVisible(False)

    def show_result(self, index: int) -> None:
        if not (0 <= index < len(self.results)):
            if index in self._single_results:
                if 0 <= self._active_result_index < len(self.results) and self.table.rowCount() > 0:
                    prev = self.results[self._active_result_index]
                    prev.frame = self._table_to_frame()
                    prev.caption = self._strip_affixes(self.caption_edit.toPlainText())
                elif self._active_result_index in self._single_results and self.table.rowCount() > 0:
                    prev = self._single_results[self._active_result_index]
                    prev.frame = self._table_to_frame()
                    prev.caption = self._strip_affixes(self.caption_edit.toPlainText())
                result = self._single_results[index]
                self._active_result_index = index
                self._set_image(result.image)
                self._frame_to_table(result.frame)
                self.caption_edit.blockSignals(True)
                self.caption_edit.setPlainText(self._effective_caption(result.caption))
                self.caption_edit.blockSignals(False)
                self._update_affix_preview()
                return
            if 0 <= index < len(self.pending_paths):
                self._active_result_index = -1
                img = self._open_image_safe(self.pending_paths[index])
                if img is not None:
                    self._set_image(img)
                else:
                    self.image_label.clear()
                    self._preview_pixmap = None
                    self.statusbar.showMessage("Could not preview this image file.", 7000)
                self.table.setRowCount(0)
                self.caption_edit.blockSignals(True)
                self.caption_edit.setPlainText("")
                self.caption_edit.blockSignals(False)
                self.affix_preview_label.setVisible(False)
            return

        if 0 <= self._active_result_index < len(self.results) and self.table.rowCount() > 0:
            prev = self.results[self._active_result_index]
            prev.frame = self._table_to_frame()
            prev.caption = self._strip_affixes(self.caption_edit.toPlainText())

        result = self.results[index]
        self._active_result_index = index
        self._set_image(result.image)
        self._frame_to_table(result.frame)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(self._effective_caption(result.caption))
        self.caption_edit.blockSignals(False)
        self._update_affix_preview()

    def _set_image(self, pil: Image.Image) -> None:
        self._preview_image = pil
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        self._preview_pixmap = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(buf.getvalue()))
        if not self.image_label.size().isEmpty():
            self.image_label.setPixmap(
                self._preview_pixmap.scaled(
                    self.image_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Recentre the list placeholder
        if hasattr(self, "_list_placeholder") and self._list_placeholder.isVisible():
            vp = self.result_list.viewport()
            self._list_placeholder.setGeometry(vp.rect())
        # Keep the drop overlay sized to the central widget
        if hasattr(self, '_drop_overlay') and self._drop_overlay.isVisible():
            rect = self.centralWidget().rect()
            self._drop_overlay.setGeometry(rect)
        if hasattr(self, '_loading_overlay') and self._loading_overlay.isVisible():
            rect = self.centralWidget().rect()
            self._loading_overlay.setGeometry(rect)

    # ==================================================================
    # Undo / Redo
    # ==================================================================

    def _push_undo_state(self) -> None:
        """Snapshot current result before a mutation for undo support."""
        r = self._current_result()
        if r is None:
            return
        state = (r.frame.copy(), r.caption)
        if self._undo_stack and self._undo_stack[-1][1] == state[1]:
            return  # debounce duplicate captions
        self._undo_stack.append(state)
        self._redo_stack.clear()

    def _undo(self) -> None:
        """Restore the previous frame + caption state."""
        if not self._undo_stack:
            return
        r = self._current_result()
        if r:
            self._redo_stack.append((r.frame.copy(), r.caption))
        frame, caption = self._undo_stack.pop()
        if r:
            r.frame = frame
            r.caption = caption
            self._frame_to_table(frame)
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(caption)
            self.caption_edit.blockSignals(False)

    def _redo(self) -> None:
        """Re-apply a previously undone state."""
        if not self._redo_stack:
            return
        r = self._current_result()
        if r:
            self._undo_stack.append((r.frame.copy(), r.caption))
        frame, caption = self._redo_stack.pop()
        if r:
            r.frame = frame
            r.caption = caption
            self._frame_to_table(frame)
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(caption)
            self.caption_edit.blockSignals(False)

    # ==================================================================
    # Table <-> frame <-> caption sync
    # ==================================================================

    def _frame_to_table(self, frame: pd.DataFrame) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row_index, row in enumerate(frame.itertuples(index=False)):
            self.table.insertRow(row_index)

            include_item = QtWidgets.QTableWidgetItem()
            include_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            include_item.setCheckState(
                QtCore.Qt.Checked if bool(row.include) else QtCore.Qt.Unchecked
            )
            self.table.setItem(row_index, 0, include_item)

            rank_item = QtWidgets.QTableWidgetItem(str(int(row.rank)))
            rank_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
            )
            self.table.setItem(row_index, 1, rank_item)

            tag_item = QtWidgets.QTableWidgetItem(str(row.tag))
            tag_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 2, tag_item)

            confidence_item = QtWidgets.QTableWidgetItem(f"{float(row.confidence):.4f}")
            confidence_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 3, confidence_item)

            category_item = QtWidgets.QTableWidgetItem(str(row.category))
            category_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 4, category_item)
        self.table.blockSignals(False)

    def _table_to_frame(self) -> pd.DataFrame:
        rows: list[dict] = []
        for row_index in range(self.table.rowCount()):
            include_item = self.table.item(row_index, 0)
            rows.append(
                {
                    "include": (
                        include_item.checkState() == QtCore.Qt.Checked
                        if include_item
                        else False
                    ),
                    "rank": (
                        int(rank_item.text())
                        if (rank_item := self.table.item(row_index, 1))
                        and rank_item.text().strip().isdigit()
                        else row_index + 1
                    ),
                    "tag": (
                        (tag_item := self.table.item(row_index, 2)).text()
                        if (tag_item := self.table.item(row_index, 2))
                        else ""
                    ),
                    "confidence": (
                        float(confidence_item.text())
                        if (confidence_item := self.table.item(row_index, 3))
                        else 0.0
                    ),
                    "category": (
                        (category_item := self.table.item(row_index, 4)).text()
                        if (category_item := self.table.item(row_index, 4))
                        else ""
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = sort_frame(frame, self.sort_mode.currentText())
        return frame

    def on_table_changed(self, *_args) -> None:
        """Surgically update the caption when a checkbox or rank is edited.

        Instead of rebuilding the whole caption from the frame (which
        would discard user-added custom tags), we only add or remove
        the specific tag that was toggled.
        """
        self._push_undo_state()
        result = self._current_result()
        if result is None:
            return
        new_frame = self._table_to_frame()
        old_frame = result.frame

        # Current caption tags (preserves user customizations)
        caption_tags = split_tags(self._strip_affixes(self.caption_edit.toPlainText()))
        caption_lower = {t.lower() for t in caption_tags}

        # Diff include flags: add newly-checked tags, remove newly-unchecked
        for _, new_row in new_frame.iterrows():
            tag = str(new_row["tag"])
            tag_lower = tag.lower()
            new_included = bool(new_row["include"])

            old_match = old_frame[
                old_frame["tag"].astype(str).str.lower() == tag_lower
            ]
            old_included = (
                bool(old_match.iloc[0]["include"])
                if not old_match.empty
                else new_included
            )

            if new_included and not old_included:
                if tag_lower not in caption_lower:
                    caption_tags.append(tag)
            elif not new_included and old_included:
                caption_tags = [t for t in caption_tags if t.lower() != tag_lower]

        result.frame = new_frame
        result.caption = ", ".join(caption_tags) if caption_tags else ""
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(self._effective_caption(result.caption))
        self.caption_edit.blockSignals(False)

    def apply_caption_text(self) -> None:
        """Sync user-edited caption back to the table.

        The caption text is treated as the source of truth.  Existing
        rows get their ``include`` and ``rank`` updated.  Tags the user
        typed that have no matching row are added as new rows with
        category ``custom`` so the table fully reflects the caption.
        """
        self._push_undo_state()
        result = self._current_result()
        if result is None:
            return
        caption_text = self._strip_affixes(self.caption_edit.toPlainText())
        result.caption = caption_text
        tags = split_tags(caption_text)
        tag_order = {tag.lower(): idx + 1 for idx, tag in enumerate(tags)}
        lower_tags = {tag.lower() for tag in tags}
        frame = result.frame.copy()

        # --- user-typed tags not yet in the frame -> add as custom rows ---
        existing_lower = set(frame["tag"].astype(str).str.lower())
        new_rows: list[dict] = []
        for idx, tag in enumerate(tags):
            if tag.lower() not in existing_lower:
                new_rows.append({
                    "include": True,
                    "rank": idx + 1,
                    "tag": tag,
                    "confidence": 0.0,
                    "category": "custom",
                })
        if new_rows:
            frame = pd.concat(
                [frame, pd.DataFrame(new_rows)], ignore_index=True
            )

        # --- update existing rows ---
        frame["include"] = frame["tag"].astype(str).str.lower().isin(lower_tags)
        frame["rank"] = [
            tag_order.get(str(tag).lower(), 9999)
            for tag in frame["tag"]
        ]
        frame = sort_frame(frame, self.sort_mode.currentText())
        result.frame = frame
        self._frame_to_table(frame)

    # ==================================================================
    # Description tagger (Tab 2)
    # ==================================================================

    def _refresh_available_models(self) -> None:
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        try:
            tagger = get_description_tagger()
            if not tagger.check_connection():
                self.model_selector.addItem("(Ollama not running)", None)
            else:
                models = tagger.list_available_models()
                for model in models:
                    self.model_selector.addItem(model, model)
                # Prefer our tested/recommended qwen3-14b-abliterated, then any qwen3 abliterated, then any abliterated/uncensored
                preferred_model = next(
                    (
                        model
                        for model in models
                        if "qwen3-14b-abliterated" in model.lower()
                    ),
                    next(
                        (
                            model
                            for model in models
                            if "qwen3" in model.lower() and "abliterated" in model.lower()
                        ),
                        next(
                            (
                                model
                                for model in models
                                if any(kw in model.lower() for kw in ["abliterated", "uncensored", "heretic", "derestricted"])
                            ),
                            models[0] if models else None,
                        ),
                    ),
                )
                if preferred_model:
                    idx = self.model_selector.findData(preferred_model)
                    if idx >= 0:
                        self.model_selector.setCurrentIndex(idx)
        except Exception:
            self.model_selector.addItem("(error loading)", None)
        finally:
            self.model_selector.blockSignals(False)

    def _on_threshold_changed(self, value: int) -> None:
        """Update the tagger's post_count threshold and clear prompt cache."""
        try:
            tagger = get_description_tagger()
            tagger.set_post_count_threshold(value)
        except Exception:
            pass  # Tagger not initialized yet — fine

    # Output-panel styling per format. Tags want a dense monospace list; prose
    # is a paragraph and reads better in the UI font at a comfortable size.
    _TAGS_DISPLAY_CSS = (
        "background-color: #0d0d0d; color: #66ff66; font-family: monospace; "
        "font-size: 11px; border-radius: 5px; padding: 8px;"
    )
    _PROSE_DISPLAY_CSS = (
        "background-color: #0d0d0d; color: #d6e9ff; font-size: 12px; "
        "border-radius: 5px; padding: 8px;"
    )

    def _on_input_mode_changed(self) -> None:
        """Refresh labels when the description/seed-tags input mode changes."""
        self._refresh_desc_io_labels()

    def _on_output_format_changed(self) -> None:
        """Refresh labels when the tags/natural-language output format changes."""
        self._refresh_desc_io_labels()

    def _refresh_desc_io_labels(self) -> None:
        """Sync every Description-tab label to the current input/output pair.

        Input mode and output format are independent, so all four combinations
        are valid: description or seed tags in, Danbooru tags or prose out.
        """
        seed_mode = self.desc_input_mode.currentData() == "seed_tags"
        natural = self.desc_output_format.currentData() == "natural"

        # --- input side ---
        if seed_mode:
            self.input_desc_group.setTitle("🏷️ Seed Tags Input")
            self.description_input.setPlaceholderText(
                "Examples:\n"
                "• 1girl, beach, volleyball\n"
                "• 1girl, witch_hat, forest\n"
                "• 1girl, 1boy, bedroom"
            )
        else:
            self.input_desc_group.setTitle("✍️ Description Input")
            self.description_input.setPlaceholderText(
                "Examples:\n"
                "• A girl with long black hair and red eyes, wearing a maid outfit\n"
                "• Anime boy with blue eyes and blonde hair, holding a sword\n"
                "• Beautiful landscape with mountains and sunset in fantasy art style\n"
                "• Character with animal ears, tail, and wearing school uniform"
            )

        if natural:
            source = "seed tags" if seed_mode else "description"
            self.desc_hint_label.setText(
                f"Write a {source}, and AI will produce a natural-language prompt:"
            )
            self.generate_from_desc_btn.setText(
                "✨ Write Prompt from Seed Tags" if seed_mode
                else "✨ Generate Prompt from Description"
            )
            self.generate_from_desc_btn.setToolTip(
                "Two-stage: AI builds a Danbooru tag set first, then rewrites it "
                "as a flowing prose prompt for Krea/Flux-style models"
            )
        elif seed_mode:
            self.desc_hint_label.setText(
                "Paste Danbooru tags and AI will add complementary tags:"
            )
            self.generate_from_desc_btn.setText("✨ Enrich Seed Tags")
            self.generate_from_desc_btn.setToolTip(
                "AI will analyze your seed tags and generate complementary Danbooru tags"
            )
        else:
            self.desc_hint_label.setText(
                "Describe what you want to see, and AI will generate Danbooru tags:"
            )
            self.generate_from_desc_btn.setText("✨ Generate Tags from Description")
            self.generate_from_desc_btn.setToolTip(
                "AI will analyze your description and generate matching Danbooru tags"
            )

        # --- combined input/output hint ---
        input_line = (
            "<b>Seed Tags:</b> paste existing tags → AI expands them"
            if seed_mode
            else "<b>Description:</b> write a scene in English"
        )
        output_line = (
            "<b>Natural Language:</b> prose paragraph for Krea / Flux "
            "(two-stage, slower)"
            if natural
            else "<b>Danbooru Tags:</b> comma-separated, for SDXL-anime models"
        )
        self.desc_io_hint.setText(f"{input_line}<br>{output_line}")

        # --- output side ---
        if natural:
            self.desc_output_group.setTitle("📝 Natural-Language Prompt")
            self.desc_tags_display.setStyleSheet(self._PROSE_DISPLAY_CSS)
            self.desc_tags_display.setPlaceholderText(
                "The generated prompt will appear here after processing..."
            )
            self._copy_tags_btn.setText("📋 Copy Prompt to Clipboard")
            self._copy_tags_btn.setToolTip("Copy the natural-language prompt")
        else:
            self.desc_output_group.setTitle("🏷️ Generated Tags")
            self.desc_tags_display.setStyleSheet(self._TAGS_DISPLAY_CSS)
            self.desc_tags_display.setPlaceholderText(
                "Generated Danbooru tags will appear here after processing..."
            )
            self._copy_tags_btn.setText("📋 Copy Tags to Clipboard")
            self._copy_tags_btn.setToolTip("Copy generated tags as comma-separated list")

        # The source-tag button only has something to copy after a prose run.
        self._copy_source_tags_btn.setVisible(
            natural and bool(getattr(self, "_last_prompt_source_tags", None))
        )

    def _generate_tags_from_description(self) -> None:
        description = self.description_input.toPlainText().strip()
        if not description:
            self.statusbar.showMessage("Input is empty. Please enter a description or seed tags.", 5000)
            return

        selected_model = self.model_selector.currentData()
        if not selected_model:
            self.desc_tags_display.setPlainText(
                "⚠️ Error:\n\nNo model selected. Refresh models or start Ollama."
            )
            self.statusbar.showMessage("Tag generation failed.", 5000)
            return

        selected_creativity = self.creativity_selector.currentData() or "creative"
        self._last_creativity_mode = selected_creativity

        enrich_mode = self.desc_input_mode.currentData() == "seed_tags"
        output_format = self.desc_output_format.currentData() or "tags"
        natural = output_format == "natural"

        if natural:
            tips = [
                "💡 Tip: This runs two passes — tags first, then prose. Give it a moment",
                "💡 Tip: Natural-language models read plain English — no weights, no tag spam",
                "💡 Tip: The source tags are kept too; copy them with the second button",
                "💡 Tip: Creative mode gives the prose more material to work with",
                "💡 Tip: Re-run for a different phrasing of the same scene",
            ]
        elif enrich_mode:
            tips = [
                "💡 Tip: Add more seed tags for richer expansion results",
                "💡 Tip: Try different creativity modes for varied complementary tags",
                "💡 Tip: Creative mode adds style/lighting tags — try it for richer atmosphere",
                "💡 Tip: The AI preserves your seed tags and only adds new ones",
                "💡 Tip: Re-run for different variations",
            ]
        else:
            tips = [
                "💡 Tip: Include a subject, action, and setting for best results",
                "💡 Tip: Re-running the same prompt can produce different (better) tags",
                "💡 Tip: Creative mode adds style/lighting tags — try it for richer atmosphere",
                "💡 Tip: If tags miss a key element, name it explicitly in your description",
                "💡 Tip: Concrete visual details beat abstract concepts",
            ]
        tip = random.choice(tips)

        self.generate_from_desc_btn.setEnabled(False)
        self._copy_source_tags_btn.setVisible(False)
        self._last_prompt_source_tags = []

        if natural:
            action_label = "Generating"
            target_label = "natural-language prompt"
        else:
            action_label = "Enriching" if enrich_mode else "Generating"
            target_label = "tags"

        self.desc_tags_display.setPlainText(
            f"⏳ {action_label} {target_label}... (this may take a while)\n\n"
            f"Mode: {selected_creativity.capitalize()}\n"
            f"{tip}"
        )
        self.statusbar.showMessage(
            f"Connecting to Ollama and {action_label.lower()} {target_label} "
            f"in {selected_creativity} mode..."
        )

        threshold = self.post_count_threshold.value()
        self._tag_worker = DescriptionTagWorker(
            description, selected_model, selected_creativity, threshold,
            enrich_mode=enrich_mode,
            output_format=output_format,
        )
        self._tag_worker.finished.connect(self._on_tags_generated)
        self._tag_worker.prompt_finished.connect(self._on_prompt_generated)
        self._tag_worker.stage.connect(self._on_generation_stage)
        self._tag_worker.error.connect(self._on_tag_generation_error)
        # Use Qt's own thread-finished signal (fires after the OS thread exits)
        # to schedule cleanup — never destroy a QThread while it's still running.
        self._tag_worker.finished.connect(self._tag_worker.quit)
        self._tag_worker.prompt_finished.connect(self._tag_worker.quit)
        self._tag_worker.error.connect(self._tag_worker.quit)
        self._tag_worker.start()

    def _on_generation_stage(self, label: str) -> None:
        """Surface multi-stage progress from the worker in the status bar."""
        self.statusbar.showMessage(label)

    def _on_tags_generated(self, result: DescriptionTagResult) -> None:
        self._last_description_tags = result.tags

        tags_output = ", ".join(result.tags)

        if len(result.tags) == 0:
            display_text = (
                "⚠️ No output generated\n\n"
                "The AI couldn't produce tags from this description. Try:\n"
                "  1. Adding a clear subject (who is in the scene?)\n"
                "  2. Adding an action (what are they doing?)\n"
                "  3. Adding a setting (where does this happen?)\n"
                "  4. Re-running — temperature variance may help\n\n"
                "Example: instead of \"a witch\" try \"a witch flying through a dark storm\""
            )
            self._copy_tags_btn.setEnabled(False)
        else:
            display_text = (
                f"✓ Generated prompt ({len(result.tags)} terms) "
                f"[{self._last_creativity_mode.capitalize()} mode]:\n\n{tags_output}"
            )
            self._copy_tags_btn.setEnabled(True)

        self.desc_tags_display.setPlainText(display_text)
        self.statusbar.showMessage(
            f"✓ Generated {len(result.tags)} prompt terms in {self._last_creativity_mode} mode. "
            "Not satisfied? Re-run for alternative results.",
            8000,
        )
        self.generate_from_desc_btn.setEnabled(True)
        if self._tag_worker is not None:
            self._tag_worker.wait()
            self._tag_worker = None

    def _on_prompt_generated(self, result: NaturalPromptResult) -> None:
        """Display a finished natural-language prompt and its source tags."""
        self._last_natural_prompt = result.prompt
        self._last_prompt_source_tags = list(result.tags)
        # Keep the tag list available too — a prose run produces both.
        self._last_description_tags = list(result.tags)

        if not result.prompt:
            self.desc_tags_display.setPlainText(
                "⚠️ No prompt generated\n\n"
                "The AI produced tags but couldn't turn them into prose. Try:\n"
                "  1. Re-running — temperature variance often fixes it\n"
                "  2. A different model (a non-thinking instruct model works best)\n"
                "  3. Switching output format to Danbooru Tags to check stage 1"
            )
            self._copy_tags_btn.setEnabled(False)
            self._copy_source_tags_btn.setVisible(False)
            self.statusbar.showMessage("Prompt generation produced no prose.", 5000)
            self.generate_from_desc_btn.setEnabled(True)
            if self._tag_worker is not None:
                self._tag_worker.wait()
                self._tag_worker = None
            return

        self.desc_tags_display.setPlainText(
            f"✓ Generated prompt ({result.word_count} words) "
            f"[{self._last_creativity_mode.capitalize()} mode]:\n\n{result.prompt}\n\n"
            f"— written from {len(result.tags)} source tags —"
        )
        self._copy_tags_btn.setEnabled(True)
        self._copy_source_tags_btn.setVisible(True)
        self.statusbar.showMessage(
            f"✓ Generated a {result.word_count}-word prompt from "
            f"{len(result.tags)} tags. Re-run for a different phrasing.",
            8000,
        )
        self.generate_from_desc_btn.setEnabled(True)
        if self._tag_worker is not None:
            self._tag_worker.wait()
            self._tag_worker = None

    def _on_tag_generation_error(self, error_msg: str) -> None:
        self.desc_tags_display.setPlainText(f"⚠️ Error:\n\n{error_msg}")
        self.statusbar.showMessage("Tag generation failed.", 5000)
        self.generate_from_desc_btn.setEnabled(True)
        self._copy_tags_btn.setEnabled(False)
        self._copy_source_tags_btn.setVisible(False)
        if self._tag_worker is not None:
            self._tag_worker.wait()
            self._tag_worker = None

    def _copy_description_tags(self) -> None:
        """Copy the primary output — prose in natural mode, tags otherwise."""
        if self.desc_output_format.currentData() == "natural":
            prompt = getattr(self, "_last_natural_prompt", "")
            if not prompt:
                self.statusbar.showMessage("No prompt to copy.", 2000)
                return
            QtWidgets.QApplication.clipboard().setText(prompt)
            self.statusbar.showMessage(
                f"✓ Copied {len(prompt.split())}-word prompt to clipboard", 3000
            )
            return

        if not getattr(self, "_last_description_tags", None):
            self.statusbar.showMessage("No tags to copy.", 2000)
            return
        tags_text = ", ".join(self._last_description_tags)
        QtWidgets.QApplication.clipboard().setText(tags_text)
        self.statusbar.showMessage(
            f"✓ Copied {len(self._last_description_tags)} tags to clipboard", 3000
        )

    def _copy_prompt_source_tags(self) -> None:
        """Copy the intermediate Danbooru tags a prose prompt was built from."""
        tags = getattr(self, "_last_prompt_source_tags", None)
        if not tags:
            self.statusbar.showMessage("No source tags to copy.", 2000)
            return
        QtWidgets.QApplication.clipboard().setText(", ".join(tags))
        self.statusbar.showMessage(
            f"✓ Copied {len(tags)} source tags to clipboard", 3000
        )

    # ==================================================================
    # Watch-folder auto-tagging
    # ==================================================================

    def _toggle_watch_folder(self, enabled: bool) -> None:
        """Start/stop watching the last-opened folder for new images."""
        if enabled:
            if not self.pending_paths:
                self.watch_folder_cb.setChecked(False)
                self.statusbar.showMessage(
                    "Open a folder first, then enable Watch Folder.", 5000
                )
                return
            watch_dir = self.pending_paths[0].parent
            self._start_watching(watch_dir)
        else:
            self._stop_watching()

    def _start_watching(self, directory: Path) -> None:
        """Begin filesystem monitoring on *directory*."""
        self._stop_watching()
        self._watch_dir = directory
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_watch_directory_changed)
        self._watcher.addPath(str(directory))
        self._watch_pending.clear()
        self.statusbar.showMessage(
            f"👁 Watching {directory} — new images will auto-load.", 6000
        )

    def _stop_watching(self) -> None:
        """Stop filesystem monitoring."""
        if self._watcher is not None:
            self._watcher.directoryChanged.disconnect(self._on_watch_directory_changed)
            self._watcher.deleteLater()
            self._watcher = None
        self._watch_dir = None
        self._watch_pending.clear()
        self._watch_timer.stop()
        self.statusbar.showMessage("Watch folder stopped.", 3000)

    def _on_watch_directory_changed(self, path: str) -> None:
        """Called by QFileSystemWatcher when files change in the watched dir."""
        self._watch_timer.start()

    def _on_watch_timer(self) -> None:
        """Debounced: scan watch dir for new image files, auto-load them."""
        if self._watch_dir is None or not self._watch_dir.exists():
            return
        new: list[Path] = []
        for entry in self._watch_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                if entry not in self.pending_paths and entry not in self._watch_pending:
                    new.append(entry)
                    self._watch_pending.add(entry)
        if not new:
            return

        self.pending_paths = sorted(
            set(self.pending_paths) | set(new), key=lambda p: p.stat().st_mtime
        )
        self.statusbar.showMessage(
            f"👁 Auto-loaded {len(new)} new image(s) — "
            f"{len(self.pending_paths)} total. Click 'Tag All Images'.", 6000
        )

        self._hide_drop_overlay()
        self.table.setRowCount(0)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText("")
        self.caption_edit.blockSignals(False)
        self._set_export_enabled(False)
        self.results = []
        self._single_results = {}
        self._active_result_index = -1

        self.pending_label.setText(f"Loaded {len(self.pending_paths)} image(s).")
        self.tag_btn.setEnabled(bool(self.pending_paths))

        self.result_list.blockSignals(True)
        self.result_list.clear()
        for p in self.pending_paths:
            self.result_list.addItem(p.name)
        self.result_list.blockSignals(False)
        self._update_list_placeholder()
        if self.pending_paths:
            self.result_list.setCurrentRow(0)
            self.show_result(0)

    # ==================================================================
    # Copy as Prompt / Negative Prompt builder
    # ==================================================================

    def _copy_as_prompt(self) -> None:
        """Copy the current caption as a ComfyUI-ready prompt (underscores->spaces)."""
        text = self._effective_caption()
        if not text:
            self.statusbar.showMessage("No caption to copy.", 3000)
            return
        prompt = text.replace("_", " ")
        QtWidgets.QApplication.clipboard().setText(prompt)
        self.statusbar.showMessage(
            "✓ Copied as prompt (underscores -> spaces). Paste into ComfyUI.", 4000
        )

    def _build_negative_prompt(self) -> None:
        """Build a Negative prompt from excluded/blacklisted tags and copy to clipboard."""
        excluded_tags: list[str] = []

        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, 0)
            tag_item = self.table.item(row, 2)
            if include_item is None or tag_item is None:
                continue
            if include_item.checkState() != QtCore.Qt.Checked:
                excluded_tags.append(tag_item.text().strip())

        blacklist_raw = self.blacklist.toPlainText().strip()
        if blacklist_raw:
            excluded_tags.extend(split_tags(blacklist_raw))

        excluded_tags = sorted(set(excluded_tags))

        if not excluded_tags:
            self.statusbar.showMessage(
                "No excluded tags found. Uncheck some tags or add a blacklist.", 4000
            )
            return

        negative = ", ".join(excluded_tags)
        QtWidgets.QApplication.clipboard().setText(negative)
        self.statusbar.showMessage(
            f"✓ Built negative prompt with {len(excluded_tags)} tags. Copied to clipboard.", 4000
        )

    # ==================================================================
    # Tag frequency dashboard
    # ==================================================================

    def _show_tag_frequency(self) -> None:
        """Display a dialog listing all tags across results with their counts."""
        counter: dict[str, int] = {}
        for r in self.results:
            tags = split_tags(r.caption)
            for t in tags:
                counter[t] = counter.get(t, 0) + 1

        if not counter:
            self.statusbar.showMessage("No tagged results to analyze.", 3000)
            return

        sorted_tags = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("📊 Tag Frequency")
        dlg.resize(500, 500)
        dlg.setModal(True)
        dlg.setStyleSheet(self.styleSheet())

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(
            f"{len(self.results)} image(s), {len(sorted_tags)} unique tags, "
            f"{sum(counter.values())} total occurrences"
        )
        header.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        header.setWordWrap(True)
        layout.addWidget(header)

        table = QtWidgets.QTableWidget(len(sorted_tags), 3)
        table.setHorizontalHeaderLabels(["Tag", "Count", "Coverage"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setColumnWidth(0, 260)
        table.setColumnWidth(1, 70)
        total = max(1, len(self.results))

        for row, (tag, count) in enumerate(sorted_tags):
            tag_item = QtWidgets.QTableWidgetItem(tag)
            tag_item.setToolTip(tag)
            table.setItem(row, 0, tag_item)

            count_item = QtWidgets.QTableWidgetItem(str(count))
            count_item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row, 1, count_item)

            pct = f"{count / total * 100:.0f}%"
            pct_item = QtWidgets.QTableWidgetItem(pct)
            pct_item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row, 2, pct_item)

        layout.addWidget(table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("📋 Copy as CSV")
        copy_btn.clicked.connect(
            lambda: self._copy_freq_to_clipboard(sorted_tags, counter)
        )
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addStretch(1)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.exec()

    def _copy_freq_to_clipboard(
        self,
        sorted_tags: list[tuple[str, int]],
        counter: dict[str, int],
    ) -> None:
        """Copy the frequency table as CSV to clipboard."""
        lines = ["tag,count,coverage_pct"]
        total = max(1, len(self.results))
        for tag, count in sorted_tags:
            pct = count / total * 100
            lines.append(f"{tag},{count},{pct:.1f}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.statusbar.showMessage("✓ Copied frequency table as CSV.", 3000)

    # ==================================================================
    # Export
    # ==================================================================

    def export_caption(self) -> None:
        """Save the current caption as a .txt file (with file dialog)."""
        result = self._current_result()
        if result is None or result.path is None:
            return
        self._sync_current_result()
        default_path = str(result.path.with_suffix(".txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save caption", default_path, "Text (*.txt)"
        )
        if not path:
            return
        Path(path).write_text(self._effective_caption(result.caption), encoding="utf-8")
        self.statusbar.showMessage(f"Saved caption to {path}")

    def export_beside_source(self) -> None:
        """Save every caption as a .txt file next to its source image."""
        if not self.results:
            return
        self._sync_current_result()
        saved = 0
        for r in self.results:
            if r.path is None:
                continue
            txt_path = r.path.with_suffix(".txt")
            try:
                txt_path.write_text(self._effective_caption(r.caption), encoding="utf-8")
                saved += 1
            except OSError as e:
                self.statusbar.showMessage(f"Error saving {txt_path.name}: {e}", 5000)
                return
        self.statusbar.showMessage(f"Saved {saved} caption(s) beside source images.")

    def export_zip(self) -> None:
        if not self.results:
            return
        self._sync_current_result()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save captions zip",
            str(Path.cwd() / "captions.zip"),
            "Zip (*.zip)",
        )
        if not path:
            return
        # Build ZIP using effective captions (prefix/postfix/initial applied)
        pairs = [(r.caption_filename, self._effective_caption(r.caption)) for r in self.results]
        from backend.tag_utils import export_zip
        Path(path).write_bytes(export_zip(pairs))
        self.statusbar.showMessage(f"Saved zip to {path}")


    # ==================================================================
    # Paste URLs from clipboard (batch import)
    # ==================================================================
    # Model management dialog (pull / list / delete)
    # ==================================================================

    # ==================================================================
    # Recognition (ONNX) model management — Batch Tagger
    # ==================================================================

    def _refresh_tagger_models(self) -> None:
        """Repopulate the recognition-model dropdown from the backend registry."""
        self.tagger_model_selector.blockSignals(True)
        self.tagger_model_selector.clear()
        active_idx = 0
        for idx, info in enumerate(tagger_backend.list_models()):
            downloaded = tagger_backend.is_model_downloaded(info.repo_id)
            star = "⭐ " if info.recommended else ""
            if downloaded:
                status = "✓ installed"
            elif info.approx_size_mb:
                status = f"⬇ ~{info.approx_size_mb} MB"
            else:
                status = "⬇ not installed"
            self.tagger_model_selector.addItem(f"{star}{info.name} — {status}", info.repo_id)
            if info.repo_id == self._active_tagger_repo:
                active_idx = idx
        self.tagger_model_selector.setCurrentIndex(active_idx)
        self.tagger_model_selector.blockSignals(False)
        self._update_tagger_model_desc(self._active_tagger_repo)

    def _update_tagger_model_desc(self, repo_id: str) -> None:
        info = tagger_backend.find_model(repo_id)
        if info is None:
            self.tagger_model_desc.setText(repo_id)
            return
        # Keep this to a single line — the dropdown already shows install state.
        self.tagger_model_desc.setText(f"<b>{info.name}</b> — {info.description}")

    def _sync_combo_to_active(self) -> None:
        """Reset the dropdown to the active model (e.g. after a cancelled switch)."""
        self.tagger_model_selector.blockSignals(True)
        idx = self.tagger_model_selector.findData(self._active_tagger_repo)
        if idx >= 0:
            self.tagger_model_selector.setCurrentIndex(idx)
        self.tagger_model_selector.blockSignals(False)
        self._update_tagger_model_desc(self._active_tagger_repo)

    def _set_active_tagger_model(self, repo_id: str) -> None:
        tagger_backend.set_selected_model(repo_id)
        self._active_tagger_repo = repo_id
        self._refresh_tagger_models()
        info = tagger_backend.find_model(repo_id)
        self.statusbar.showMessage(
            f"Recognition model set to {info.name if info else repo_id}.", 5000
        )

    def _on_tagger_model_changed(self, _index: int) -> None:
        repo_id = self.tagger_model_selector.currentData()
        if not repo_id:
            return
        if repo_id == self._active_tagger_repo:
            self._update_tagger_model_desc(repo_id)
            return
        if tagger_backend.is_model_downloaded(repo_id):
            self._set_active_tagger_model(repo_id)
            return
        # Not downloaded — offer to fetch it now.
        info = tagger_backend.find_model(repo_id)
        name = info.name if info else repo_id
        size = f" (~{info.approx_size_mb} MB)" if info and info.approx_size_mb else ""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Download Recognition Model",
            f"'{name}' is not downloaded yet{size}.\n\n"
            "Download it now and set it as the active model?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._sync_combo_to_active()
            return
        self._start_tagger_download(repo_id, activate=True)

    def _start_tagger_download(self, repo_id: str, activate: bool = True, done_cb=None) -> None:
        """Download a model in the background, showing a modal busy dialog."""
        if self._tagger_worker is not None and self._tagger_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Please Wait", "Another model operation is already in progress."
            )
            return

        info = tagger_backend.find_model(repo_id)
        name = info.name if info else repo_id

        busy = QtWidgets.QDialog(self)
        busy.setWindowTitle("Downloading Model")
        busy.setModal(True)
        busy.setStyleSheet(self.styleSheet())
        busy.setFixedWidth(440)
        busy.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        v = QtWidgets.QVBoxLayout(busy)
        v.setSpacing(10)
        label = QtWidgets.QLabel(
            f"⬇ Downloading <b>{name}</b>…<br>"
            "First download can take several minutes depending on model size."
        )
        label.setWordWrap(True)
        v.addWidget(label)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)  # indeterminate
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        v.addWidget(bar)
        hide_btn = QtWidgets.QPushButton("Hide (keeps downloading)")
        hide_btn.clicked.connect(busy.accept)
        v.addWidget(hide_btn)

        def _finished(success: bool, message: str, done_repo: str) -> None:
            busy.accept()
            self._tagger_worker = None
            if success:
                if activate:
                    self._set_active_tagger_model(done_repo)
                else:
                    self._refresh_tagger_models()
                self.statusbar.showMessage(f"{name}: {message}", 6000)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Download Failed", f"Could not download {name}:\n\n{message}"
                )
                self._sync_combo_to_active()
            if done_cb is not None:
                done_cb(success)

        self._tagger_worker = TaggerModelWorker("download", repo_id)
        self._tagger_worker.finished.connect(_finished)
        self._tagger_worker.finished.connect(self._tagger_worker.quit)
        self.statusbar.showMessage(f"Downloading {name}…")
        self._tagger_worker.start()
        busy.exec()

    def _show_tagger_model_manager(self) -> None:
        """Dialog to download / update / activate / delete recognition models."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("⚙️ Recognition Model Manager")
        dlg.resize(660, 540)
        dlg.setModal(True)
        dlg.setStyleSheet(self.styleSheet())
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Choose which ONNX vision model recognizes your images. Download new "
            "models, update to the latest weights, set the active model, or delete "
            "ones you no longer use to reclaim disk space."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9ecbff; font-size: 11px;")
        layout.addWidget(intro)

        list_group = QtWidgets.QGroupBox("📦 Available Models")
        list_layout = QtWidgets.QVBoxLayout(list_group)

        model_list = QtWidgets.QListWidget()
        model_list.setStyleSheet(
            "background-color: #0d0d0d; color: #e0e0e0; border: 1px solid #444; "
            "border-radius: 5px; padding: 5px;"
        )
        list_layout.addWidget(model_list, 1)

        desc_label = QtWidgets.QLabel("")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #9ecbff; font-size: 10px; padding: 2px;")
        list_layout.addWidget(desc_label)

        status_label = QtWidgets.QLabel("")
        status_label.setStyleSheet("color: #9ecbff; font-size: 10px;")
        list_layout.addWidget(status_label)

        def _refresh_list() -> None:
            model_list.blockSignals(True)
            model_list.clear()
            for info in tagger_backend.list_models():
                downloaded = tagger_backend.is_model_downloaded(info.repo_id)
                active = info.repo_id == self._active_tagger_repo
                marks: list[str] = []
                if active:
                    marks.append("● active")
                if downloaded:
                    marks.append("✓ installed")
                elif info.approx_size_mb:
                    marks.append(f"~{info.approx_size_mb} MB")
                else:
                    marks.append("not installed")
                if info.recommended:
                    marks.append("recommended")
                if not info.builtin:
                    marks.append("custom")
                item = QtWidgets.QListWidgetItem(f"{info.name}   [{' · '.join(marks)}]")
                item.setData(QtCore.Qt.UserRole, info.repo_id)
                item.setToolTip(f"{info.repo_id}\n\n{info.description}")
                model_list.addItem(item)
            model_list.blockSignals(False)
            # Keep the main-window dropdown in sync with any changes made here.
            self._refresh_tagger_models()
            _on_row_changed()

        def _on_row_changed() -> None:
            item = model_list.currentItem()
            if item is None:
                desc_label.setText("Select a model to see details.")
                dl_btn.setEnabled(False)
                set_active_btn.setEnabled(False)
                delete_btn.setEnabled(False)
                return
            repo = item.data(QtCore.Qt.UserRole)
            info = tagger_backend.find_model(repo)
            downloaded = tagger_backend.is_model_downloaded(repo)
            is_active = repo == self._active_tagger_repo
            dl_btn.setEnabled(True)
            dl_btn.setText("🔁 Update" if downloaded else "⬇ Download")
            set_active_btn.setEnabled(downloaded and not is_active)
            # Can remove a downloaded model or a custom entry, but not the active one.
            removable = (downloaded or (info is not None and not info.builtin)) and not is_active
            delete_btn.setEnabled(removable)
            if info is not None:
                state = "installed" if downloaded else "not installed"
                desc_label.setText(f"<b>{info.repo_id}</b><br>{info.description} <i>({state})</i>")
            else:
                desc_label.setText(repo)

        model_list.currentRowChanged.connect(lambda _row: _on_row_changed())

        # --- Action buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)

        dl_btn = QtWidgets.QPushButton("⬇ Download")
        dl_btn.setToolTip("Download the selected model, or update it to the latest weights")

        def _on_download() -> None:
            item = model_list.currentItem()
            if item is None:
                return
            repo = item.data(QtCore.Qt.UserRole)
            self._start_tagger_download(repo, activate=False, done_cb=lambda ok: _refresh_list())

        dl_btn.clicked.connect(_on_download)
        btn_row.addWidget(dl_btn)

        set_active_btn = QtWidgets.QPushButton("✅ Set Active")
        set_active_btn.setToolTip("Use the selected model for image tagging")

        def _on_set_active() -> None:
            item = model_list.currentItem()
            if item is None:
                return
            repo = item.data(QtCore.Qt.UserRole)
            self._set_active_tagger_model(repo)
            status_label.setText(f"Active model: {repo}")
            _refresh_list()

        set_active_btn.clicked.connect(_on_set_active)
        btn_row.addWidget(set_active_btn)

        delete_btn = QtWidgets.QPushButton("🗑 Delete")
        delete_btn.setStyleSheet(
            "QPushButton { background-color: #5c1a1a; color: #ffaaaa; "
            "border: 1px solid #9a3a3a; }"
            "QPushButton:hover { background-color: #702828; border: 1px solid #cc4444; }"
        )
        delete_btn.setToolTip("Remove the selected model's downloaded files")

        def _on_delete() -> None:
            item = model_list.currentItem()
            if item is None:
                return
            repo = item.data(QtCore.Qt.UserRole)
            info = tagger_backend.find_model(repo)
            name = info.name if info else repo
            reply = QtWidgets.QMessageBox.question(
                dlg,
                "Confirm Delete",
                f"Delete downloaded files for '{name}'?\n\n"
                "This frees disk space. You can download it again later.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                tagger_backend.delete_model_files(repo)
            except Exception as e:  # pragma: no cover - defensive
                QtWidgets.QMessageBox.warning(dlg, "Delete Failed", str(e))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            status_label.setText(f"Deleted {name}.")
            _refresh_list()

        delete_btn.clicked.connect(_on_delete)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch(1)
        list_layout.addLayout(btn_row)
        layout.addWidget(list_group, 1)

        # --- Custom Hugging Face repo ---
        custom_group = QtWidgets.QGroupBox("➕ Add Custom Model (Advanced)")
        custom_layout = QtWidgets.QHBoxLayout(custom_group)
        custom_input = QtWidgets.QLineEdit()
        custom_input.setPlaceholderText("Hugging Face repo, e.g. SmilingWolf/wd-vit-large-tagger-v3")
        custom_input.setStyleSheet(
            "background-color: #0d0d0d; color: #ffffff; padding: 6px; "
            "border: 1px solid #444; border-radius: 4px;"
        )
        custom_layout.addWidget(custom_input, 1)

        def _on_add_custom() -> None:
            repo = custom_input.text().strip()
            if not repo:
                return
            if "/" not in repo:
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "Invalid Repo",
                    "Enter a full Hugging Face repo id in the form 'owner/model'.\n"
                    "The repo must contain 'model.onnx' and 'selected_tags.csv'.",
                )
                return
            custom_input.clear()
            self._start_tagger_download(repo, activate=False, done_cb=lambda ok: _refresh_list())

        add_custom_btn = QtWidgets.QPushButton("⬇ Download")
        add_custom_btn.clicked.connect(_on_add_custom)
        custom_layout.addWidget(add_custom_btn)
        layout.addWidget(custom_group)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        _refresh_list()
        if model_list.count():
            model_list.setCurrentRow(0)
        _on_row_changed()
        dlg.exec()

    # ==================================================================
    # Shutdown
    # ==================================================================

    def _pin_toolbar_button_widths(self) -> None:
        """Stop the caption toolbar from eliding button labels.

        Runs once the widgets are laid out and their size hints are final.
        """
        for button in getattr(self, "_toolbar_buttons", ()):
            try:
                button.setMinimumWidth(button.sizeHint().width())
            except RuntimeError:
                pass  # widget already destroyed

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop background threads before the window (and its widgets) die.

        Qt aborts the process if a QThread is still running when it is
        destroyed, so every worker is disconnected and joined here.
        """
        workers = [
            getattr(self, name, None)
            for name in ("_vlm_worker", "_tag_worker", "_image_worker")
        ]
        for w in workers:
            if w is None:
                continue
            try:
                if hasattr(w, "cancel"):
                    w.cancel()
                w.disconnect()  # detach every slot before teardown
            except (RuntimeError, TypeError):
                pass
            try:
                if w.isRunning():
                    w.quit()
                    w.wait(5000)
            except RuntimeError:
                pass  # already destroyed by Qt
        super().closeEvent(event)

    # ==================================================================
    # Image → natural-language prompt (vision model)
    # ==================================================================

    def _vlm_image_sources(self) -> list[tuple[int, str, object]]:
        """Collect images available for captioning, newest workflow first.

        Tagged results are preferred when present, but untagged loaded images
        work too — the vision model does not need tags, and running the ONNX
        tagger first is pointless for photographs.
        """
        sources: list[tuple[int, str, object]] = []
        if self.results:
            for i, r in enumerate(self.results):
                # Prefer the path; pasted/dropped images live only in memory.
                sources.append((i, r.name, r.path if r.path else r.image))
        elif self.pending_paths:
            for i, p in enumerate(self.pending_paths):
                sources.append((i, p.name, p))
        return sources

    def _show_image_prompt_dialog(self) -> None:
        """Caption images directly with a vision model, bypassing the tagger."""
        sources = self._vlm_image_sources()
        if not sources:
            QtWidgets.QMessageBox.information(
                self,
                "No images",
                "Load some images first (drag & drop, Open Images, or paste).\n\n"
                "You do not need to tag them — the vision model reads the image "
                "directly.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("📝 Write Prompt from Image")
        dlg.setMinimumSize(940, 620)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setSpacing(8)

        intro = QtWidgets.QLabel(
            "Describes the image itself in natural language for "
            "<b>Krea / Flux-style</b> models — no Danbooru tags in between. "
            "Works on photographs, which the ONNX taggers do not."
        )
        intro.setStyleSheet("color: #9ecbff; font-size: 11px;")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Settings row (packed horizontally to stay short on 1080p) ---
        settings = QtWidgets.QGroupBox("⚙️ Settings")
        grid = QtWidgets.QGridLayout(settings)
        grid.setSpacing(6)
        combo_css = "background-color: #1a1a1a; color: #ffffff; padding: 4px;"
        lbl_css = "color: #4da6ff; font-size: 10px; font-weight: bold;"

        for col, text in enumerate(("Vision model:", "Caption style:", "Length:")):
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet(lbl_css)
            grid.addWidget(lab, 0, col)
            grid.setColumnStretch(col, 1 if col == 0 else 0)

        model_combo = QtWidgets.QComboBox()
        model_combo.setStyleSheet(combo_css)
        installed: set[str] = set()
        detected: list[str] = []
        try:
            _cap = VLMCaptioner()
            installed = set(_cap.list_installed_models())
            # Anything Ollama reports as vision-capable, so a newer or more
            # accurate captioner appears without a code change here.
            detected = _cap.list_vision_models()
        except Exception:
            pass

        curated_ids = set()
        for info in list_vlm_models():
            curated_ids.add(info.model_id)
            mark = "✓ installed" if info.model_id in installed else f"~{info.approx_size_gb:.1f} GB"
            model_combo.addItem(f"{info.name} — {mark}", info.model_id)
            model_combo.setItemData(
                model_combo.count() - 1, info.description, QtCore.Qt.ToolTipRole
            )

        # Other installed vision models the curated list does not know about.
        for name in detected:
            if name in curated_ids:
                continue
            model_combo.addItem(f"{name} — ✓ installed", name)
            model_combo.setItemData(
                model_combo.count() - 1,
                "Detected on your machine: Ollama reports vision support for "
                "this model. Quality for captioning is unverified.",
                QtCore.Qt.ToolTipRole,
            )

        # Default to an installed model when there is one.
        for i in range(model_combo.count()):
            if model_combo.itemData(i) in installed:
                model_combo.setCurrentIndex(i)
                break
        grid.addWidget(model_combo, 1, 0)

        style_combo = QtWidgets.QComboBox()
        style_combo.setStyleSheet(combo_css)
        for key, (label, _) in CAPTION_STYLES.items():
            style_combo.addItem(label, key)
        style_combo.setCurrentIndex(
            max(0, list(CAPTION_STYLES).index(DEFAULT_STYLE))
        )
        style_combo.setToolTip(
            "Descriptive styles produce flowing prose — the right shape for\n"
            "Krea/Flux. 'Stable Diffusion prompt' returns comma-separated\n"
            "fragments instead."
        )
        grid.addWidget(style_combo, 1, 1)

        length_combo = QtWidgets.QComboBox()
        length_combo.setStyleSheet(combo_css)
        for key in CAPTION_LENGTHS:
            length_combo.addItem(key.replace("_", " ").title(), key)
        length_combo.setCurrentIndex(max(0, list(CAPTION_LENGTHS).index(DEFAULT_LENGTH)))
        grid.addWidget(length_combo, 1, 2)

        # --- Extra instructions (JoyCaption's own option strings) ---
        # Packed as two compact rows so the dialog stays usable on 1080p.
        extra_boxes: dict[str, QtWidgets.QCheckBox] = {}
        extra_row = QtWidgets.QGridLayout()
        extra_row.setSpacing(4)
        pos = 0  # own counter so skipped options don't leave holes in the grid
        for key, (label, text) in EXTRA_OPTIONS.items():
            if key in EXPLICIT_EXTRAS:
                continue  # driven by the Explicit toggle below
            cb = QtWidgets.QCheckBox(label)
            if key == "vulgar":
                cb.setToolTip(
                    text
                    + "\n\nNote: tuned for training-caption realism, not prompt "
                    "quality.\nIn testing it traded descriptive detail for slang "
                    "(\"natural light\nfrom a window on the right\" became "
                    "\"lighting's bright\"), and\nprofanity is not a visual "
                    "descriptor a diffusion model can use."
                )
            else:
                cb.setToolTip(text)
            cb.setStyleSheet("font-size: 10px;")
            extra_boxes[key] = cb
            extra_row.addWidget(cb, pos // 3, pos % 3)
            pos += 1
        # Lighting and camera angle are what Krea-style prompts live on.
        extra_boxes["lighting"].setChecked(True)
        extra_boxes["camera_angle"].setChecked(True)
        grid.addLayout(extra_row, 2, 0, 1, 3)

        explicit_cb = QtWidgets.QCheckBox("🔞 Explicit — describe directly, no euphemisms")
        explicit_cb.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 11px;")
        explicit_cb.setToolTip(
            "JoyCaption is uncensored, but a formal register still euphemises —\n"
            "that is what turns explicit imagery into \"modest cleavage\".\n\n"
            "Switches to a casual tone and tells the model to describe anatomy\n"
            "and state of dress bluntly instead of reaching for polite wording.\n\n"
            "For profanity as well, tick 'Vulgar slang' — but note it trades\n"
            "descriptive detail for slang, which usually makes a worse prompt."
        )

        def sync_explicit() -> None:
            # "Keep it PG" and Explicit are contradictory; never both.
            pg = extra_boxes.get("keep_pg")
            if pg is not None and explicit_cb.isChecked():
                pg.setChecked(False)
            if pg is not None:
                pg.setEnabled(not explicit_cb.isChecked())

        explicit_cb.toggled.connect(sync_explicit)
        if "keep_pg" in extra_boxes:
            extra_boxes["keep_pg"].toggled.connect(
                lambda on: explicit_cb.setChecked(False) if on else None
            )
        grid.addWidget(explicit_cb, 3, 0, 1, 3)

        def selected_extras() -> list[str]:
            return [k for k, cb in extra_boxes.items() if cb.isChecked()]

        # --- Output: still-image prompt, or a Wan 2.2 video prompt ---
        out_row = QtWidgets.QHBoxLayout()
        out_row.setSpacing(6)
        out_label = QtWidgets.QLabel("Output:")
        out_label.setStyleSheet(lbl_css)
        out_row.addWidget(out_label)

        output_combo = QtWidgets.QComboBox()
        output_combo.setStyleSheet(combo_css)
        output_combo.addItem("🖼️ Image prompt (Krea / Flux)", "")
        for key, label in VIDEO_STYLES.items():
            output_combo.addItem(f"🎬 {label}", key)
        output_combo.setToolTip(
            "Video modes add a second stage: the vision model describes the\n"
            "first frame, then the text model invents the motion.\n\n"
            "The vision model is a captioner — asked to plan motion directly it\n"
            "just restates the still — so this is split across two models."
        )
        out_row.addWidget(output_combo, 2)

        duration_label = QtWidgets.QLabel("Seconds:")
        duration_label.setStyleSheet(lbl_css)
        out_row.addWidget(duration_label)
        duration_spin = QtWidgets.QSpinBox()
        duration_spin.setRange(MIN_DURATION, MAX_DURATION)
        duration_spin.setValue(DEFAULT_DURATION)
        duration_spin.setToolTip(
            "Clip length. One timeline beat per second.\n\n"
            f"Past about {RELIABLE_DURATION}s the model often drops the last "
            "beat or two;\nthe result reports how many it actually produced."
        )
        out_row.addWidget(duration_spin)

        text_model_combo = QtWidgets.QComboBox()
        text_model_combo.setStyleSheet(combo_css)
        text_model_combo.setToolTip(
            "Stage-2 text model: writes the motion from the first-frame caption.\n\n"
            "This is a language task, so it needs an instruction-following text\n"
            "model. Vision captioners (JoyCaption) are excluded — asked for a\n"
            "timeline they either restate the still or fail outright."
        )
        # Curated captioners are the wrong tool here and were the cause of a
        # silent failure: JoyCaption picked as stage 2 raised on 1 run in 3
        # and produced caption-restating beats on the other two.
        captioner_ids = {info.model_id for info in list_vlm_models()}
        try:
            from backend.description_tagger import get_description_tagger

            for name in get_description_tagger().list_available_models():
                if name in captioner_ids:
                    continue
                text_model_combo.addItem(name, name)
        except Exception:
            pass
        if text_model_combo.count() == 0:
            text_model_combo.addItem(VideoPrompter.DEFAULT_MODEL, VideoPrompter.DEFAULT_MODEL)
        # Prefer the default; otherwise the first remaining entry — never
        # leave the selection on something that cannot do the job.
        for i in range(text_model_combo.count()):
            if text_model_combo.itemData(i) == VideoPrompter.DEFAULT_MODEL:
                text_model_combo.setCurrentIndex(i)
                break
        else:
            text_model_combo.setCurrentIndex(0)
        out_row.addWidget(text_model_combo, 2)

        grid.addLayout(out_row, 4, 0, 1, 3)

        # --- Free-text narration steer ---
        hint_label = QtWidgets.QLabel("Narration hint (optional):")
        hint_label.setStyleSheet(lbl_css)
        grid.addWidget(hint_label, 7, 0, 1, 3)

        custom_hint = QtWidgets.QLineEdit()
        custom_hint.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 5px; "
            "border: 1px solid #444; border-radius: 3px;"
        )
        custom_hint.setPlaceholderText(
            "e.g. cinematic film-noir tone · clinical and factual · "
            "describe her from the viewer's POV · emphasise texture and fabric"
        )
        custom_hint.setMaxLength(400)
        custom_hint.setToolTip(
            "Free-text steering for how the description is written.\n\n"
            "Folded into the caption request before the option lines, because\n"
            "position drives register here — the same words appended last are\n"
            "measurably weaker.\n\n"
            "In video modes it also steers how the motion is worded, but it\n"
            "never overrides the output format or the plausibility limits.\n"
            "It is not applied to the MMAudio prompt, which has its own strict\n"
            "format and a 77-token budget."
        )
        grid.addWidget(custom_hint, 8, 0, 1, 3)

        audio_cb = QtWidgets.QCheckBox("🔊 Also write an MMAudio prompt")
        audio_cb.setChecked(True)
        audio_cb.setToolTip(
            "Writes MMAudio conditioning for the same clip: a positive\n"
            "soundscape description plus a negative prompt for exclusions.\n\n"
            "Uses the text model that is already loaded for the motion stage,\n"
            "so it costs a few seconds and no extra VRAM swap."
        )
        audio_cb.setStyleSheet("font-size: 10px;")
        grid.addWidget(audio_cb, 9, 0, 1, 3)

        vram_note = QtWidgets.QLabel(
            "Video mode runs two models. On a 16 GB card they cannot both stay "
            "resident, so the first image after switching pays a reload."
        )
        vram_note.setStyleSheet("color: #ffb366; font-size: 10px;")
        vram_note.setWordWrap(True)
        vram_note.setVisible(False)
        grid.addWidget(vram_note, 5, 0, 1, 3)

        def video_style() -> str:
            return output_combo.currentData() or ""

        def sync_output_mode() -> None:
            is_video = bool(video_style())
            for wdg in (duration_label, duration_spin, text_model_combo):
                wdg.setVisible(is_video)
            audio_cb.setVisible(is_video)
            duration_spin.setEnabled(video_style() == "timeline")
            duration_label.setEnabled(video_style() == "timeline")
            vram_note.setVisible(is_video)

        output_combo.currentIndexChanged.connect(sync_output_mode)
        sync_output_mode()

        layout.addWidget(settings)

        # --- Results: list on the left, prompt on the right ---
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)

        img_list = QtWidgets.QListWidget()
        img_list.setStyleSheet("background-color: #0d0d0d; color: #ffffff;")
        for _, name, _img in sources:
            img_list.addItem(name)
        img_list.setCurrentRow(min(self._current_index(), len(sources) - 1)
                               if self._current_index() >= 0 else 0)
        split.addWidget(img_list)

        prompt_view = QtWidgets.QPlainTextEdit()
        # Read-only on purpose. Copy and Save read from the result object, not
        # from this widget, so an edit here would be silently discarded — and
        # the view also carries labelled context (first frame, MMAudio) that is
        # not part of any single copyable value. Selection and Ctrl+C still
        # work, as does the right-click Copy / Select All menu.
        prompt_view.setReadOnly(True)
        # setReadOnly alone drops keyboard selection, which would break the
        # obvious Ctrl+A / Ctrl+C gesture. Put both selection modes back.
        prompt_view.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        prompt_view.setPlaceholderText(
            "Select an image and click “Caption Selected”.\n\n"
            "The first run loads several GB into VRAM and can take a minute; "
            "after that each image takes a few seconds."
        )
        prompt_view.setStyleSheet(
            "background-color: #0d0d0d; color: #d6e9ff; font-size: 12px; "
            "border-radius: 5px; padding: 8px;"
        )
        split.addWidget(prompt_view)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([240, 700])
        layout.addWidget(split, 1)

        progress = QtWidgets.QProgressBar()
        progress.setVisible(False)
        layout.addWidget(progress)

        status = QtWidgets.QLabel("")
        status.setStyleSheet("color: #9ecbff; font-size: 10px;")
        status.setWordWrap(True)
        layout.addWidget(status)

        # Captions accumulate here, keyed by the source index.
        captions: dict[int, object] = {}

        def result_text(result) -> str:
            """The video prompt when one was requested, else the caption.

            In video mode the caption is still shown underneath as the
            first-frame description the motion was written from.
            """
            if getattr(result, "video_prompt", ""):
                blocks = [result.video_prompt]
                # A timeline that came up short would otherwise be silent —
                # the video model would just get fewer seconds than asked.
                beats = result.video_prompt.count("(At ")
                wanted = getattr(result, "video_duration", 0)
                if wanted and beats and beats < wanted:
                    blocks.append(
                        f"⚠️ {beats} of {wanted} beats — the model came up "
                        "short. Re-run, or use a shorter clip length."
                    )
                if getattr(result, "audio_prompt", ""):
                    blocks.append(
                        "— 🔊 MMAudio prompt —\n"
                        f"{result.audio_prompt}"
                    )
                    if result.audio_negative:
                        blocks.append(
                            "— 🔊 MMAudio negative prompt —\n"
                            f"{result.audio_negative}"
                        )
                blocks.append(
                    f"— first frame ({result.word_count} words) —\n{result.caption}"
                )
                return "\n\n".join(blocks)
            if getattr(result, "video_style", ""):
                # A video was requested but stage 2 produced nothing. Say so
                # in the output itself, so a bare caption is never mistaken
                # for the video prompt.
                return (
                    "⚠️ VIDEO PROMPT NOT GENERATED — the motion stage failed "
                    "(see the warning). Only the first-frame caption is "
                    "shown below.\n\n"
                    f"— first frame ({result.word_count} words) —\n"
                    f"{result.caption}"
                )
            return result.caption

        def primary_text(result) -> str:
            """Just the thing to copy — never the trailing context."""
            return getattr(result, "video_prompt", "") or result.caption

        def audio_file_text(result) -> str:
            """MMAudio conditioning as a standalone file body."""
            lines = [getattr(result, "audio_prompt", "")]
            if getattr(result, "audio_negative", ""):
                lines.append("")
                lines.append(f"NEGATIVE: {result.audio_negative}")
            return "\n".join(lines)

        def show_selected() -> None:
            row = img_list.currentRow()
            if row < 0 or row >= len(sources):
                return
            idx = sources[row][0]
            result = captions.get(idx)
            if result is None:
                prompt_view.setPlainText("")
                prompt_view.setPlaceholderText(
                    "Not captioned yet — click “Caption Selected”."
                )
            else:
                prompt_view.setPlainText(result_text(result))
            copy_audio_btn.setVisible(
                bool(result is not None and getattr(result, "audio_prompt", ""))
            )

        img_list.currentRowChanged.connect(show_selected)

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)
        caption_btn = QtWidgets.QPushButton("✨ Caption Selected")
        caption_btn.setStyleSheet(
            "background-color: #0059b3; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        caption_all_btn = QtWidgets.QPushButton(f"✨ Caption All ({len(sources)})")
        copy_btn = QtWidgets.QPushButton("📋 Copy")
        copy_audio_btn = QtWidgets.QPushButton("🔊 Copy Audio")
        copy_audio_btn.setToolTip(
            "Copy the MMAudio prompt on its own.\n"
            "Hold Shift to copy the negative prompt instead."
        )
        copy_audio_btn.setVisible(False)
        save_btn = QtWidgets.QPushButton("💾 Save .txt")
        save_all_btn = QtWidgets.QPushButton("💾 Save All")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (
            caption_btn, caption_all_btn, copy_btn, copy_audio_btn,
            save_btn, save_all_btn,
        ):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def set_busy(busy: bool) -> None:
            for b in (caption_btn, caption_all_btn):
                b.setEnabled(not busy)
            progress.setVisible(busy)

        def on_item_done(index: int, result) -> None:
            captions[index] = result
            row = img_list.currentRow()
            if 0 <= row < len(sources) and sources[row][0] == index:
                prompt_view.setPlainText(result_text(result))
                copy_audio_btn.setVisible(bool(getattr(result, "audio_prompt", "")))
            # Mark captioned items in the list.
            for r, (idx, name, _i) in enumerate(sources):
                if idx == index:
                    img_list.item(r).setText(f"✓ {name}")
                    break

        def on_progress(pct: int, label: str) -> None:
            progress.setValue(pct)
            status.setText(label)

        def release_worker() -> None:
            """Drop the worker reference only once its OS thread has exited.

            `finished` is emitted from inside run(), while the thread is still
            alive. Dropping the last reference there lets Qt destroy a running
            QThread, which crashes the process — so always wait() first.
            """
            w = self._vlm_worker
            if w is None:
                return
            w.wait()
            self._vlm_worker = None

        def on_finished(results: list, failures: list) -> None:
            set_busy(False)
            if failures:
                # A per-stage failure with a successful caption used to be
                # invisible — the user saw a caption where a timeline should
                # be and concluded video mode was broken.
                status.setText(
                    f"⚠️ {len(results)} captioned, but {len(failures)} stage(s) "
                    "failed — see details."
                )
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "Some stages failed",
                    "The image caption succeeded, but a later stage did not:\n\n"
                    + "\n".join(failures[:6])
                    + (
                        f"\n\n…and {len(failures) - 6} more." if len(failures) > 6 else ""
                    )
                    + "\n\nThe caption is kept. Check the stage-2 text model — "
                    "it must be an instruction-following text model, not a "
                    "vision captioner.",
                )
            else:
                status.setText(
                    f"✓ Captioned {len(results)} image(s). "
                    "Re-run for a different phrasing."
                )
            release_worker()

        def on_error(msg: str) -> None:
            set_busy(False)
            status.setText("")
            QtWidgets.QMessageBox.warning(dlg, "Captioning failed", msg)
            release_worker()

        def start(items: list) -> None:
            if self._vlm_worker is not None and self._vlm_worker.isRunning():
                return
            model = model_combo.currentData()
            if not model:
                QtWidgets.QMessageBox.warning(
                    dlg, "No model", "Select a vision model first."
                )
                return
            set_busy(True)
            progress.setValue(0)
            status.setText("Loading vision model (first run can take a minute)...")
            w = VLMCaptionWorker(
                items,
                model,
                style_combo.currentData(),
                length_combo.currentData(),
                extras=selected_extras(),
                explicit=explicit_cb.isChecked(),
                video_style=video_style(),
                video_duration=duration_spin.value(),
                text_model=text_model_combo.currentData() or "",
                want_audio=audio_cb.isChecked(),
                custom_instruction=custom_hint.text().strip(),
            )
            w.item_done.connect(on_item_done)
            w.progress.connect(on_progress)
            w.finished.connect(on_finished)
            w.error.connect(on_error)
            w.finished.connect(w.quit)
            w.error.connect(w.quit)
            # Held on self, not in a dialog-local closure: the dialog's widgets
            # are destroyed when it closes, and a worker outliving that scope
            # must still have a live Python reference.
            self._vlm_worker = w
            w.start()

        def caption_selected() -> None:
            row = img_list.currentRow()
            if row < 0 or row >= len(sources):
                return
            idx, _name, image = sources[row]
            start([(idx, image)])

        def caption_all() -> None:
            start([(idx, image) for idx, _n, image in sources])

        def current_result():
            row = img_list.currentRow()
            if row < 0 or row >= len(sources):
                return None
            return captions.get(sources[row][0])

        def copy_audio() -> None:
            result = current_result()
            if result is None:
                return
            shift = bool(
                QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier
            )
            text = (
                getattr(result, "audio_negative", "")
                if shift
                else getattr(result, "audio_prompt", "")
            )
            if not text:
                status.setText("No MMAudio prompt for this image yet.")
                return
            QtWidgets.QApplication.clipboard().setText(text)
            status.setText(
                "✓ Copied MMAudio " + ("negative prompt." if shift else "prompt.")
            )

        def copy_current() -> None:
            result = current_result()
            if result is None:
                return
            # Copy the prompt alone — never the first-frame context shown below it.
            text = primary_text(result).strip()
            if not text:
                return
            QtWidgets.QApplication.clipboard().setText(text)
            status.setText(f"✓ Copied {len(text.split())} words to clipboard.")

        def save_current() -> None:
            row = img_list.currentRow()
            result = current_result()
            if row < 0 or result is None:
                return
            text = primary_text(result).strip()
            if not text:
                return
            stem = Path(sources[row][1]).stem
            suffix = "_video" if getattr(result, "video_prompt", "") else "_prompt"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save prompt", f"{stem}{suffix}.txt", "Text files (*.txt)"
            )
            if path:
                body = text
                if getattr(result, "audio_prompt", ""):
                    body += f"\n\nMMAUDIO: {result.audio_prompt}"
                    if result.audio_negative:
                        body += f"\nMMAUDIO NEGATIVE: {result.audio_negative}"
                Path(path).write_text(body, encoding="utf-8")
                status.setText(f"✓ Saved {Path(path).name}")

        def save_all() -> None:
            if not captions:
                QtWidgets.QMessageBox.information(
                    dlg, "Nothing to save", "Caption some images first."
                )
                return
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Save all prompts to folder"
            )
            if not folder:
                return
            written = 0
            for idx, name, _img in sources:
                result = captions.get(idx)
                if result is None:
                    continue
                # Suffixed so these never overwrite tag captions, which use the
                # bare "<stem>.txt" LoRA convention — and so video prompts do
                # not overwrite image prompts for the same source.
                stem = Path(name).stem
                suffix = "_video" if getattr(result, "video_prompt", "") else "_prompt"
                Path(folder, f"{stem}{suffix}.txt").write_text(
                    primary_text(result), encoding="utf-8"
                )
                written += 1
                # Audio goes to its own file — a batch feeding a pipeline wants
                # them separable, not one blob to re-split.
                if getattr(result, "audio_prompt", ""):
                    Path(folder, f"{stem}_audio.txt").write_text(
                        audio_file_text(result), encoding="utf-8"
                    )
                    written += 1
            status.setText(f"✓ Saved {written} prompt file(s) to {folder}")

        caption_btn.clicked.connect(caption_selected)
        caption_all_btn.clicked.connect(caption_all)
        copy_btn.clicked.connect(copy_current)
        copy_audio_btn.clicked.connect(copy_audio)
        save_btn.clicked.connect(save_current)
        save_all_btn.clicked.connect(save_all)
        close_btn.clicked.connect(dlg.accept)

        # --- Session persistence of every setting in this dialog ---------
        # Snapshot on close, restore on open. Combos are stored by their data
        # value rather than index, so a model installed between two opens
        # (which shifts the list) still restores the right selection.
        def _select_data(combo: QtWidgets.QComboBox, value: object) -> None:
            for i in range(combo.count()):
                if combo.itemData(i) == value:
                    combo.setCurrentIndex(i)
                    return

        def snapshot_settings() -> None:
            try:
                self._vlm_dialog_state = {
                    "vision_model": model_combo.currentData(),
                    "style": style_combo.currentData(),
                    "length": length_combo.currentData(),
                    "extras": {k: cb.isChecked() for k, cb in extra_boxes.items()},
                    "explicit": explicit_cb.isChecked(),
                    "output": output_combo.currentData(),
                    "duration": duration_spin.value(),
                    "text_model": text_model_combo.currentData(),
                    "audio": audio_cb.isChecked(),
                    "hint": custom_hint.text(),
                }
            except RuntimeError:
                pass  # widgets already torn down; keep the previous snapshot

        def restore_settings() -> None:
            s = self._vlm_dialog_state
            if not s:
                return
            if s.get("vision_model") is not None:
                _select_data(model_combo, s["vision_model"])
            if s.get("style") is not None:
                _select_data(style_combo, s["style"])
            if s.get("length") is not None:
                _select_data(length_combo, s["length"])
            for k, on in (s.get("extras") or {}).items():
                if k in extra_boxes:
                    extra_boxes[k].setChecked(bool(on))
            # Explicit before keep_pg's exclusivity handler runs, so the
            # restored pair cannot fight each other.
            explicit_cb.setChecked(bool(s.get("explicit", False)))
            if s.get("output") is not None:
                _select_data(output_combo, s["output"])
            if "duration" in s:
                duration_spin.setValue(int(s["duration"]))
            if s.get("text_model") is not None:
                _select_data(text_model_combo, s["text_model"])
            audio_cb.setChecked(bool(s.get("audio", True)))
            custom_hint.setText(str(s.get("hint", "")))

        def on_close() -> None:
            """Detach the worker before this dialog's widgets are destroyed.

            Any signal still queued would otherwise be delivered into deleted
            C++ widgets and crash the process, so disconnect first, then ask
            the worker to stop. If it is mid-image we deliberately do NOT drop
            the reference — a running QThread must never be garbage collected.
            """
            snapshot_settings()
            w = self._vlm_worker
            if w is None:
                return
            for signal in (w.item_done, w.progress, w.finished, w.error):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass  # nothing connected
            w.cancel()
            if w.wait(3000):
                self._vlm_worker = None
            # Still running: the reference stays on self so Qt cannot destroy
            # it mid-flight. It exits on its own now that it is cancelled, and
            # its signals are already disconnected.

        dlg.finished.connect(lambda _r: on_close())
        # Restore after every widget and handler exists, so the change
        # signals (output mode -> visibility, explicit -> keep_pg) fire
        # against a fully built dialog.
        restore_settings()
        show_selected()
        dlg.exec()

    def _show_model_manager(self) -> None:
        """Open a dialog to pull, list or delete Ollama models."""

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("⚙️ Model Manager — Ollama")
        dlg.resize(600, 480)
        dlg.setModal(True)
        dlg.setStyleSheet(self.styleSheet())

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setSpacing(10)

        # --- Pull section ---
        pull_group = QtWidgets.QGroupBox("⬇ Pull New Model")
        pull_layout = QtWidgets.QHBoxLayout(pull_group)
        pull_input = QtWidgets.QLineEdit()
        pull_input.setPlaceholderText("e.g. qwen2:7b, llama3.1:8b, gemma2:9b")
        pull_input.setStyleSheet(
            "background-color: #0d0d0d; color: #ffffff; padding: 6px; "
            "border: 1px solid #444; border-radius: 4px;"
        )
        pull_layout.addWidget(pull_input, 1)

        # Progress bar shared by pull and delete operations
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 0)  # indeterminate
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)
        progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1a1a1a; border: 1px solid #444; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #4da6ff; border-radius: 2px; }"
        )
        progress_bar.hide()

        self._model_worker: ModelOperationWorker | None = None

        def _on_worker_finished(success: bool, message: str) -> None:
            progress_bar.hide()
            pull_btn.setEnabled(True)
            pull_btn.setText("⬇ Pull")
            delete_btn.setEnabled(True)
            delete_btn.setText("🗑 Delete Selected")
            if success:
                QtWidgets.QMessageBox.information(dlg, "Success", message)
                status_label.setText(message)
            else:
                QtWidgets.QMessageBox.warning(dlg, "Operation Failed", message)
                status_label.setText(f"Error: {message}")
            _refresh_list()

        def _on_pull() -> None:
            model_name = pull_input.text().strip()
            if not model_name:
                return
            pull_btn.setEnabled(False)
            pull_btn.setText("⏳ Pulling…")
            delete_btn.setEnabled(False)
            progress_bar.show()
            status_label.setText(f"Pulling {model_name} — this may take several minutes…")
            self._model_worker = ModelOperationWorker("pull", model_name)
            self._model_worker.finished.connect(_on_worker_finished)
            self._model_worker.start()

        pull_btn = QtWidgets.QPushButton("⬇ Pull")
        pull_btn.clicked.connect(_on_pull)
        pull_layout.addWidget(pull_btn)
        layout.addWidget(pull_group)

        # --- Installed models ---
        list_group = QtWidgets.QGroupBox("📦 Installed Models")
        list_layout = QtWidgets.QVBoxLayout(list_group)

        model_list = QtWidgets.QListWidget()
        model_list.setStyleSheet(
            "background-color: #0d0d0d; color: #e0e0e0; border: 1px solid #444; "
            "border-radius: 5px; padding: 5px;"
        )
        list_layout.addWidget(model_list, 1)

        status_label = QtWidgets.QLabel("")
        status_label.setStyleSheet("color: #9ecbff; font-size: 10px;")
        list_layout.addWidget(status_label)

        list_layout.addWidget(progress_bar)

        list_btn_row = QtWidgets.QHBoxLayout()
        list_btn_row.setSpacing(4)

        def _refresh_list() -> None:
            model_list.clear()
            try:
                tagger = get_description_tagger()
                if not tagger.check_connection():
                    status_label.setText("⚠ Ollama not running. Start it with 'ollama serve'.")
                    return
                models = tagger.list_available_models()
                for m in models:
                    model_list.addItem(m)
                status_label.setText(f"{len(models)} model(s) installed.")
            except Exception as e:
                status_label.setText(f"Error: {e}")

        refresh_list_btn = QtWidgets.QPushButton("🔄 Refresh")
        refresh_list_btn.clicked.connect(_refresh_list)
        list_btn_row.addWidget(refresh_list_btn)

        def _on_delete() -> None:
            current = model_list.currentItem()
            if current is None:
                return
            model_name = current.text().strip()
            reply = QtWidgets.QMessageBox.question(
                dlg,
                "Confirm Delete",
                f"Delete model '{model_name}'?\n\n"
                "This frees disk space. You can re-pull it later.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            delete_btn.setEnabled(False)
            delete_btn.setText("⏳ Deleting…")
            pull_btn.setEnabled(False)
            progress_bar.show()
            status_label.setText(f"Deleting {model_name}…")
            self._model_worker = ModelOperationWorker("delete", model_name)
            self._model_worker.finished.connect(_on_worker_finished)
            self._model_worker.start()

        delete_btn = QtWidgets.QPushButton("🗑 Delete Selected")
        delete_btn.setStyleSheet(
            "QPushButton { background-color: #5c1a1a; color: #ffaaaa; "
            "border: 1px solid #9a3a3a; }"
            "QPushButton:hover { background-color: #702828; border: 1px solid #cc4444; }"
        )
        delete_btn.clicked.connect(_on_delete)
        list_btn_row.addWidget(delete_btn)
        list_btn_row.addStretch(1)
        list_layout.addLayout(list_btn_row)
        layout.addWidget(list_group)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        _refresh_list()
        dlg.exec()

    # ==================================================================
    # Entry point
    # ==================================================================

def _build_splash_pixmap() -> QtGui.QPixmap:
    """Draw the startup splash.

    Painted rather than shipped as an asset so it survives PyInstaller builds
    without extra --add-data wiring.
    """
    width, height = 520, 240
    scale = 2  # render at 2x so it stays crisp on HiDPI displays
    pixmap = QtGui.QPixmap(width * scale, height * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

    card = QtCore.QRectF(0, 0, width, height)

    gradient = QtGui.QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, QtGui.QColor("#16181d"))
    gradient.setColorAt(1.0, QtGui.QColor("#0d0e11"))
    painter.setBrush(QtGui.QBrush(gradient))
    painter.setPen(QtGui.QPen(QtGui.QColor("#2b3038"), 1))
    painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)

    # Accent bar, echoing the app's blue.
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor("#0073e6"))
    painter.drawRoundedRect(QtCore.QRectF(0, 0, width, 4), 2, 2)

    # Explicit family stack rather than painter.font(): the default font is
    # not guaranteed to resolve in a frozen build, and an unresolved family
    # renders every glyph as a tofu box.
    families = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]

    title_font = QtGui.QFont()
    title_font.setFamilies(families)
    title_font.setPointSize(26)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QtGui.QColor("#ffffff"))
    painter.drawText(
        QtCore.QRectF(0, 58, width, 44),
        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        "Img-Tagbooru",
    )

    sub_font = QtGui.QFont()
    sub_font.setFamilies(families)
    sub_font.setPointSize(10)
    painter.setFont(sub_font)
    painter.setPen(QtGui.QColor("#8fb8e6"))
    painter.drawText(
        QtCore.QRectF(0, 102, width, 24),
        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        "Local image tagging, captioning and prompt generation",
    )

    ver_font = QtGui.QFont()
    ver_font.setFamilies(families)
    ver_font.setPointSize(9)
    painter.setFont(ver_font)
    painter.setPen(QtGui.QColor("#6b7280"))
    painter.drawText(
        QtCore.QRectF(0, 126, width, 20),
        QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        f"v{APP_VERSION}  ·  runs entirely on your machine",
    )

    painter.end()
    return pixmap


def main() -> None:
    # Windowed builds (pythonw / PyInstaller --windowed) run without a console,
    # so sys.stdout/sys.stderr are None. Libraries that write to them (tqdm,
    # logging, print) would crash — route None streams to a null sink.
    import os
    for _stream_name in ("stdout", "stderr"):
        if getattr(sys, _stream_name, None) is None:
            try:
                setattr(sys, _stream_name, open(os.devnull, "w", encoding="utf-8"))
            except Exception:
                pass

    # Suppress the HuggingFace unauthenticated-request warning — it's noise
    # for end users who don't have an HF_TOKEN set.
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    # Belt-and-suspenders: also disable HF progress bars via env for any
    # subprocess / re-import path that runs before backend.tagger is imported.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*unauthenticated.*",
        category=UserWarning,
    )

    app = QtWidgets.QApplication([])
    app.setApplicationName("Img-Tagbooru")
    app.setApplicationDisplayName("Img-Tagbooru")
    app.setApplicationVersion(APP_VERSION)

    # Age confirmation and terms acceptance. Shown once, and again whenever
    # TERMS_VERSION changes. Declining exits before anything else loads.
    if tagger_backend.get_config_value("terms_accepted_version") != TERMS_VERSION:
        terms = TermsDialog()
        if terms.exec() != QtWidgets.QDialog.Accepted:
            sys.exit(0)
        tagger_backend.set_config_value("terms_accepted_version", TERMS_VERSION)
        tagger_backend.set_config_value("terms_accepted_age_confirmed", True)

    # Splash while the window builds. Construction is not instant — it wires up
    # several tabs and probes Ollama for installed models — so without this the
    # app looks hung for a second or two after launch.
    splash = QtWidgets.QSplashScreen(
        _build_splash_pixmap(), QtCore.Qt.WindowStaysOnTopHint
    )
    splash.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def boot(message: str) -> None:
        splash.showMessage(
            f"  {message}",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
            QtGui.QColor("#9ecbff"),
        )
        app.processEvents()

    splash.show()
    boot("Starting…")

    window = None
    try:
        window = MainWindow(progress_cb=boot)
        boot("Ready")
        window.show()
    finally:
        # finish() ties the splash to the window; close() covers the case where
        # construction raised and there is no window to hand off to.
        if window is not None:
            splash.finish(window)
        else:
            splash.close()

    app.exec()


if __name__ == "__main__":
    main()
