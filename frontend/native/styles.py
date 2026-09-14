from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Temp-directory management - written once, cleaned up on exit
# ---------------------------------------------------------------------------

_TMP_DIR = Path(tempfile.gettempdir()) / "img-tagger-assets"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_temp_assets() -> None:
    """Remove the img-tagger-assets directory on process exit."""
    try:
        if _TMP_DIR.exists():
            shutil.rmtree(_TMP_DIR, ignore_errors=True)
    except Exception:
        pass  # best-effort cleanup


atexit.register(_cleanup_temp_assets)


# ---------------------------------------------------------------------------
# Checkmark SVG
# ---------------------------------------------------------------------------

def _write_checkmark_svg() -> str:
    """Write a white checkmark SVG to a temp file and return its path.

    The file is only written once; subsequent calls return the cached path.
    """
    svg_path = _TMP_DIR / "checkmark.svg"
    if not svg_path.exists():
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
            '<polyline points="2,7 6,11 12,3" stroke="white" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
            '</svg>'
        )
        svg_path.write_text(svg_content, encoding="utf-8")
    return str(svg_path).replace("\\", "/")


def _write_arrow_svgs(
    color: str = "#e0e0e0",
    hover_color: str = "#ffffff",
    suffix: str = "",
) -> tuple[str, str, str, str]:
    """Write up/down chevron SVGs and return their temp-file paths
    as (up_normal, up_hover, down_normal, down_hover).

    *color* and *hover_color* are SVG stroke colours; *suffix* is appended
    to the file name so we can generate distinct sets per spinbox.

    Each SVG file is written only once; subsequent calls return cached paths.
    """

    def _svg(points: str, stroke: str) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
            f'<polyline points="{points}" stroke="{stroke}" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
            '</svg>'
        )

    up_normal = _TMP_DIR / f"arrow_up{suffix}.svg"
    if not up_normal.exists():
        up_normal.write_text(_svg("1,5 5,1 9,5", color), encoding="utf-8")

    up_hover = _TMP_DIR / f"arrow_up_hover{suffix}.svg"
    if not up_hover.exists():
        up_hover.write_text(_svg("1,5 5,1 9,5", hover_color), encoding="utf-8")

    down_normal = _TMP_DIR / f"arrow_down{suffix}.svg"
    if not down_normal.exists():
        down_normal.write_text(_svg("1,1 5,5 9,1", color), encoding="utf-8")

    down_hover = _TMP_DIR / f"arrow_down_hover{suffix}.svg"
    if not down_hover.exists():
        down_hover.write_text(_svg("1,1 5,5 9,1", hover_color), encoding="utf-8")

    return (
        str(up_normal).replace("\\", "/"),
        str(up_hover).replace("\\", "/"),
        str(down_normal).replace("\\", "/"),
        str(down_hover).replace("\\", "/"),
    )


CHECKMARK_SVG_PATH = _write_checkmark_svg()

# Default arrow set (unused by spinboxes but kept as fallback)
ARROW_UP_SVG, ARROW_UP_HOVER_SVG, ARROW_DOWN_SVG, ARROW_DOWN_HOVER_SVG = _write_arrow_svgs()

# Per-spinbox coloured arrow sets
_GREEN_UP, _GREEN_UP_H, _GREEN_DOWN, _GREEN_DOWN_H = _write_arrow_svgs(
    "#66ff66", "#a0ffa0", "_green"
)
_PINK_UP, _PINK_UP_H, _PINK_DOWN, _PINK_DOWN_H = _write_arrow_svgs(
    "#ff66a3", "#ff99c2", "_pink"
)
_YELLOW_UP, _YELLOW_UP_H, _YELLOW_DOWN, _YELLOW_DOWN_H = _write_arrow_svgs(
    "#ffcc66", "#ffe099", "_yellow"
)


# ---------------------------------------------------------------------------
# Application stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """Return the complete dark-theme Qt stylesheet."""

    return """
        QMainWindow {
            background-color: #0d0d0d;
        }
        QWidget {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        QSplitter::handle {
            background-color: #333;
            border-radius: 2px;
        }
        QSplitter::handle:hover {
            background-color: #4da6ff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #333;
            margin-top: 10px;
            padding-top: 16px;
            border-radius: 8px;
            background-color: #1f1f1f;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #4da6ff;
            font-size: 12px;
        }
        QPushButton {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #444;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #3b3b3b;
            border: 1px solid #4da6ff;
        }
        QPushButton:pressed {
            background-color: #1a1a1a;
        }
        QPushButton#tagBtn {
            background-color: #0059b3;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
        }
        QPushButton#tagBtn:hover {
            background-color: #0073e6;
        }
        QPushButton#tagSelectedBtn {
            background-color: #1a5c1a;
            color: #aaffaa;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            border: 1px solid #2d8a2d;
        }
        QPushButton#tagSelectedBtn:hover {
            background-color: #236b23;
            border: 1px solid #44bb44;
            color: #ccffcc;
        }
        QPushButton#tagSelectedBtn:pressed {
            background-color: #0f3d0f;
        }
        QPushButton#tagSelectedBtn:disabled {
            background-color: #1a2e1a;
            color: #557755;
            border: 1px solid #2a3d2a;
        }
        QPushButton#aiMetaBtn {
            background-color: #4a1a5c;
            color: #ddaaff;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            border: 1px solid #7a4aaa;
        }
        QPushButton#aiMetaBtn:hover {
            background-color: #5c2870;
            border: 1px solid #9966cc;
            color: #eeccff;
        }
        QPushButton#aiMetaBtn:pressed {
            background-color: #3a0f4a;
        }
        QPushButton#aiMetaBtn:disabled {
            background-color: #2a1a2e;
            color: #775577;
            border: 1px solid #3a2a3d;
        }
        QPushButton#copyPromptBtn {
            background-color: #1a3a5c;
            color: #aaddff;
            font-weight: bold;
            font-size: 10px;
            padding: 6px 12px;
            border: 1px solid #3a6a9a;
        }
        QPushButton#copyPromptBtn:hover {
            background-color: #284a70;
            border: 1px solid #4da6ff;
            color: #cceeff;
        }
        QPushButton#copyPromptBtn:pressed {
            background-color: #0f2a44;
        }
        QPushButton#negPromptBtn {
            background-color: #5c1a1a;
            color: #ffaaaa;
            font-weight: bold;
            font-size: 10px;
            padding: 6px 12px;
            border: 1px solid #9a3a3a;
        }
        QPushButton#negPromptBtn:hover {
            background-color: #702828;
            border: 1px solid #cc4444;
            color: #ffcccc;
        }
        QPushButton#negPromptBtn:pressed {
            background-color: #440f0f;
        }
        QPushButton#tagFreqBtn {
            background-color: #3a3a1a;
            color: #ffffaa;
            font-weight: bold;
            font-size: 10px;
            padding: 6px 12px;
            border: 1px solid #7a7a3a;
        }
        QPushButton#tagFreqBtn:hover {
            background-color: #5a5a28;
            border: 1px solid #cccc44;
            color: #ffffcc;
        }
        QPushButton#tagFreqBtn:pressed {
            background-color: #2a2a0f;
        }
        QLabel#alertLabel {
            color: #ff9933;
            font-weight: bold;
        }
        QLabel {
            color: #e0e0e0;
        }
        QTableWidget {
            background-color: #0d0d0d;
            alternate-background-color: #1a1a1a;
            gridline-color: #333;
            border: 1px solid #444;
            border-radius: 5px;
            selection-background-color: #0059b3;
        }
        QTableWidget::item {
            padding: 4px;
            border-bottom: 1px solid #222;
        }
        QTableWidget::item:selected {
            background-color: #0059b3;
            color: white;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #4da6ff;
            padding: 6px;
            border: 1px solid #444;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 5px;
            background-color: #1a1a1a;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #1a1a1a;
            color: #4da6ff;
            font-weight: bold;
            border-bottom: 2px solid #4da6ff;
        }
        QTabBar::tab:hover {
            background-color: #3b3b3b;
        }
        QProgressBar {
            border: 1px solid #444;
            border-radius: 3px;
            background-color: #2b2b2b;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background-color: #0059b3;
            border-radius: 2px;
        }
        QStatusBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-top: 1px solid #444;
        }
        QListWidget {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 5px;
        }
        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #222;
        }
        QListWidget::item:selected {
            background-color: #0059b3;
            color: white;
            border-radius: 3px;
        }
        QPlainTextEdit {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 5px;
            color: #e0e0e0;
        }
        QLineEdit {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 5px 28px 5px 8px;
            color: #e0e0e0;
            font-size: 11px;
        }
        QLineEdit:focus {
            border: 1px solid #4da6ff;
        }
        QLineEdit:placeholder {
            color: #666;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px 22px 4px 6px;
            color: #e0e0e0;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4da6ff;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 20px;
            height: 11px;
            border-left: 1px solid #444;
            border-bottom: 1px solid #444;
            border-top-right-radius: 3px;
            background-color: #1a1a1a;
            margin: 0;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
            background-color: #2b2b2b;
        }
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
            background-color: #0d0d0d;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 20px;
            height: 11px;
            border-left: 1px solid #444;
            border-bottom-right-radius: 3px;
            background-color: #1a1a1a;
            margin: 0;
        }
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #2b2b2b;
        }
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
            background-color: #0d0d0d;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url(ARROW_UP_SVG);
            width: 10px;
            height: 6px;
        }
        QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
            image: url(ARROW_UP_HOVER_SVG);
        }
        QSpinBox::up-arrow:pressed, QDoubleSpinBox::up-arrow:pressed {
            image: url(ARROW_UP_HOVER_SVG);
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url(ARROW_DOWN_SVG);
            width: 10px;
            height: 6px;
        }
        QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
            image: url(ARROW_DOWN_HOVER_SVG);
        }
        QSpinBox::down-arrow:pressed, QDoubleSpinBox::down-arrow:pressed {
            image: url(ARROW_DOWN_HOVER_SVG);
        }
        /* --- Per-spinbox coloured chevrons ------------------------------ */
        /* General threshold (green #66ff66) */
        QDoubleSpinBox#generalThreshold::up-arrow {
            image: url(GREEN_UP);
        }
        QDoubleSpinBox#generalThreshold::up-arrow:hover,
        QDoubleSpinBox#generalThreshold::up-arrow:pressed {
            image: url(GREEN_UP_HOVER);
        }
        QDoubleSpinBox#generalThreshold::down-arrow {
            image: url(GREEN_DOWN);
        }
        QDoubleSpinBox#generalThreshold::down-arrow:hover,
        QDoubleSpinBox#generalThreshold::down-arrow:pressed {
            image: url(GREEN_DOWN_HOVER);
        }
        /* Character threshold (pink #ff66a3) */
        QDoubleSpinBox#characterThreshold::up-arrow {
            image: url(PINK_UP);
        }
        QDoubleSpinBox#characterThreshold::up-arrow:hover,
        QDoubleSpinBox#characterThreshold::up-arrow:pressed {
            image: url(PINK_UP_HOVER);
        }
        QDoubleSpinBox#characterThreshold::down-arrow {
            image: url(PINK_DOWN);
        }
        QDoubleSpinBox#characterThreshold::down-arrow:hover,
        QDoubleSpinBox#characterThreshold::down-arrow:pressed {
            image: url(PINK_DOWN_HOVER);
        }
        /* Max tags (yellow #ffcc66) */
        QSpinBox#maxTags::up-arrow {
            image: url(YELLOW_UP);
        }
        QSpinBox#maxTags::up-arrow:hover,
        QSpinBox#maxTags::up-arrow:pressed {
            image: url(YELLOW_UP_HOVER);
        }
        QSpinBox#maxTags::down-arrow {
            image: url(YELLOW_DOWN);
        }
        QSpinBox#maxTags::down-arrow:hover,
        QSpinBox#maxTags::down-arrow:pressed {
            image: url(YELLOW_DOWN_HOVER);
        }
        QComboBox {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
            color: #e0e0e0;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 8px solid #4da6ff;
            width: 0px;
            height: 0px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #444;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
        }
        QCheckBox {
            color: #e0e0e0;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #666;
            border-radius: 4px;
            background-color: #0d0d0d;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #4da6ff;
            background-color: #1a1a1a;
        }
        QCheckBox::indicator:checked {
            background-color: #0059b3;
            border: 2px solid #4da6ff;
            image: url(CHECKMARK_PATH);
        }
        QCheckBox::indicator:checked:hover {
            background-color: #0073e6;
            border: 2px solid #66b3ff;
            image: url(CHECKMARK_PATH);
        }
        QCheckBox::indicator:unchecked:hover {
            background-color: #1a1a1a;
        }
    """.replace("CHECKMARK_PATH", CHECKMARK_SVG_PATH)\
      .replace("ARROW_UP_HOVER_SVG", ARROW_UP_HOVER_SVG)\
      .replace("ARROW_UP_SVG", ARROW_UP_SVG)\
      .replace("ARROW_DOWN_HOVER_SVG", ARROW_DOWN_HOVER_SVG)\
      .replace("ARROW_DOWN_SVG", ARROW_DOWN_SVG)\
      .replace("GREEN_UP_HOVER", _GREEN_UP_H)\
      .replace("GREEN_UP", _GREEN_UP)\
      .replace("GREEN_DOWN_HOVER", _GREEN_DOWN_H)\
      .replace("GREEN_DOWN", _GREEN_DOWN)\
      .replace("PINK_UP_HOVER", _PINK_UP_H)\
      .replace("PINK_UP", _PINK_UP)\
      .replace("PINK_DOWN_HOVER", _PINK_DOWN_H)\
      .replace("PINK_DOWN", _PINK_DOWN)\
      .replace("YELLOW_UP_HOVER", _YELLOW_UP_H)\
      .replace("YELLOW_UP", _YELLOW_UP)\
      .replace("YELLOW_DOWN_HOVER", _YELLOW_DOWN_H)\
      .replace("YELLOW_DOWN", _YELLOW_DOWN)
