from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TagTrie - prefix tree for O(k) autocomplete lookups (not O(n) scan)
# ---------------------------------------------------------------------------

class TagTrie:
    """Prefix tree mapping *normalised* tag prefixes to original tag names.

    Building the trie is O(n × len) but done once at load time.
    Each keystroke lookup is O(len(token)) - independent of tag count.
    """

    __slots__ = ("_roots", )

    def __init__(self) -> None:
        # _roots: normalised first-char → node dict
        self._roots: dict[str, dict] = {}

    def insert(self, original: str, normalised: str) -> None:
        """Insert a single (*original*, *normalised*) tag pair into the trie."""
        if not normalised:
            return
        char = normalised[0]
        if char not in self._roots:
            self._roots[char] = {}
        node = self._roots[char]
        for c in normalised[1:]:
            if c not in node:
                node[c] = {}
            node = node[c]
        # leaf: store original tags (multiple may share normalised form)
        node.setdefault("__tags__", []).append(original)

    @staticmethod
    def _normalise(s: str) -> str:
        """Strip underscores, dashes and spaces for fuzzy comparison."""
        return s.replace('_', '').replace('-', '').replace(' ', '').lower()

    def build(self, tags: list[str]) -> None:
        """Populate the trie from a list of original tag names."""
        self._roots.clear()
        for tag in tags:
            self.insert(tag, self._normalise(tag))

    def prefix_matches(self, token: str, max_results: int = 20) -> list[str]:
        """Return up to *max_results* tags whose normalised form starts with *token*."""
        token_norm = self._normalise(token)
        if not token_norm:
            return []

        char = token_norm[0]
        node = self._roots.get(char)
        if node is None:
            return []

        # Walk the trie down to the node corresponding to token_norm
        for c in token_norm[1:]:
            if c not in node:
                return []
            node = node[c]

        # Collect all leaf tags under this node via DFS
        results: list[str] = []
        self._collect(node, results, max_results)
        return results

    def _collect(self, node: dict, results: list[str], limit: int) -> None:
        """DFS traversal to gather leaf tags."""
        if len(results) >= limit:
            return
        tags = node.get("__tags__")
        if tags:
            for tag in tags:
                if len(results) >= limit:
                    return
                results.append(tag)
        for key, child in sorted(node.items()):
            if key == "__tags__":
                continue
            self._collect(child, results, limit)


# ---------------------------------------------------------------------------
# CSV loader thread - avoids blocking the Qt main thread
# ---------------------------------------------------------------------------

class TagLoadWorker(QtCore.QThread):
    """Load Danbooru tags from CSV in a background thread."""

    finished_signal = QtCore.Signal(list)
    error_signal = QtCore.Signal(str)

    def __init__(self, csv_path: str) -> None:
        super().__init__()
        self._csv_path = csv_path

    def run(self) -> None:
        tags: list[str] = []
        try:
            with open(self._csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_name = row.get("name", "").strip().lower()
                    if tag_name:
                        tags.append(tag_name)
            self.finished_signal.emit(tags)
        except Exception as e:
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# CaptionCompleterMixin
# ---------------------------------------------------------------------------

class CaptionCompleterMixin:
    """Mixin that adds Danbooru-tag autocomplete to a caption QPlainTextEdit.

    Requires the host class to define:
    - ``self.caption_edit``  (:class:`QtWidgets.QPlainTextEdit`)
    - ``self.danbooru_tags`` (:class:`list[str]`) - set before calling
      :meth:`_setup_caption_completer`.
    """

    # ---------- tag loading ---------------------------------------------------

    @staticmethod
    def _find_data_file(filename: str) -> Path | None:
        """Locate a bundled data file (works in dev and PyInstaller builds)."""
        # PyInstaller onefile extraction directory
        base = getattr(sys, '_MEIPASS', None)
        if base:
            candidate = Path(base) / filename
            if candidate.exists():
                return candidate
        # Development: relative to the project root (two levels above this file)
        candidate = Path(__file__).resolve().parents[2] / filename
        if candidate.exists():
            return candidate
        # Fallback: current working directory
        candidate = Path.cwd() / filename
        if candidate.exists():
            return candidate
        return None

    def _load_danbooru_tags(self) -> list[str]:
        """Load Danbooru tags from CSV file for autocomplete.

        Returns an empty list if the file cannot be found; the actual
        loading is dispatched to a background thread via
        :meth:`_load_danbooru_tags_async`.
        """
        csv_path = self._find_data_file("danbooru_tags_post_count.csv")
        if csv_path is None:
            logger.warning("Danbooru tags file not found (danbooru_tags_post_count.csv)")
            return []

        self._tag_load_worker = TagLoadWorker(str(csv_path))
        self._tag_load_worker.finished_signal.connect(self._on_tags_loaded)
        self._tag_load_worker.error_signal.connect(self._on_tags_load_error)
        self._tag_load_worker.start()
        return []  # Will be populated async via _on_tags_loaded

    def _on_tags_loaded(self, tags: list[str]) -> None:
        """Slot: called when CSV loading completes in the background."""
        self.danbooru_tags = sorted(tags)
        logger.info("Loaded %d Danbooru tags for autocomplete", len(tags))
        # Build the prefix trie for O(k) lookups
        self._tag_trie = TagTrie()
        self._tag_trie.build(self.danbooru_tags)
        # Now that tags are available, wire up the completer if needed
        if hasattr(self, 'caption_edit') and self.caption_edit is not None:
            self._setup_caption_completer()

    def _on_tags_load_error(self, error: str) -> None:
        """Slot: called on CSV loading error."""
        logger.warning("Could not load Danbooru tags: %s", error)

    # ---------- setup ---------------------------------------------------------

    def _setup_caption_completer(self) -> None:
        """Set up autocomplete for caption_edit using Danbooru tags (ComfyUI-style)."""
        if not self.danbooru_tags:
            return

        # Install event filter so keyboard navigation (arrows, Enter,
        # Escape, Tab) reaches _completer_event_filter before Qt's
        # default handling.
        self.caption_edit.installEventFilter(self)  # type: ignore[attr-defined]

        # Popup list – ToolTip flag keeps it visible without stealing focus
        self._completer_popup = QtWidgets.QListWidget(self)  # type: ignore[attr-defined]
        self._completer_popup.setWindowFlags(
            QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint
        )
        self._completer_popup.setFocusPolicy(QtCore.Qt.NoFocus)
        self._completer_popup.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self._completer_popup.setMouseTracking(True)
        self._completer_popup.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._completer_popup.setStyleSheet(
            "QListWidget { border: 1px solid #555; background: #2b2b2b; color: #ddd; font-size: 12px; }"
            "QListWidget::item:selected { background: #4a90d9; color: #fff; }"
            "QListWidget::item:hover { background: #3a3a3a; }"
        )
        self._completer_popup.itemClicked.connect(self._on_completer_item_clicked)
        self._completer_popup.hide()

        self.caption_edit.textChanged.connect(self._on_caption_text_changed)  # type: ignore[attr-defined]
        self._current_completions: list[str] = []
        self._completer_active = False
        self._completing = False  # guard against recursive signal

    # ---------- token detection ------------------------------------------------

    def _get_current_token(self) -> tuple[str, int, int]:
        """Return (token_text, token_start, token_end) at the cursor."""
        cursor = self.caption_edit.textCursor()  # type: ignore[attr-defined]
        text = self.caption_edit.toPlainText()  # type: ignore[attr-defined]
        cursor_pos = cursor.position()
        text_before = text[:cursor_pos]

        last_comma = text_before.rfind(',')
        if last_comma == -1:
            token_start = 0
        else:
            token_start = last_comma + 1
            while token_start < len(text_before) and text_before[token_start] == ' ':
                token_start += 1

        # end of token: next comma after cursor or end of text
        rest = text[cursor_pos:]
        next_comma = rest.find(',')
        token_end = cursor_pos if next_comma == -1 else cursor_pos + next_comma

        token_text = text[token_start:token_end].strip()
        return token_text, token_start, token_end

    # ---------- popup management ----------------------------------------------

    def _on_caption_text_changed(self) -> None:
        """Show tag suggestions as the user types in caption_edit.

        Uses a prefix trie (:class:`TagTrie`) for O(k) lookups instead
        of the old O(n) linear scan through all tags.
        """
        if self._completing or not self.danbooru_tags or not hasattr(self, '_completer_popup'):
            return

        token, _start, _end = self._get_current_token()

        if len(token) < 2:
            self._hide_completer()
            return

        trie = getattr(self, '_tag_trie', None)
        if trie is not None:
            matches = trie.prefix_matches(token)
        else:
            # Fallback: simple linear scan (should not happen once trie is built)
            token_norm = TagTrie._normalise(token)
            matches = []
            for tag in self.danbooru_tags:
                if TagTrie._normalise(tag).startswith(token_norm):
                    matches.append(tag)
                    if len(matches) >= 20:
                        break

        if not matches:
            self._hide_completer()
            return

        self._show_completer_popup(matches)

    def _show_completer_popup(self, matches: list[str]) -> None:
        """Populate and position the suggestion popup."""
        self._current_completions = matches
        self._completer_popup.blockSignals(True)
        self._completer_popup.clear()
        for tag in matches:
            self._completer_popup.addItem(tag)
        self._completer_popup.setCurrentRow(0)
        self._completer_popup.blockSignals(False)

        # Size: fit content width (min 250) and at most 10 rows tall
        row_h = self._completer_popup.sizeHintForRow(0) + 2
        n_visible = min(len(matches), 10)
        popup_h = row_h * n_visible + 4

        # Measure widest tag
        fm = self._completer_popup.fontMetrics()
        max_w = max(fm.horizontalAdvance(t) for t in matches) + 24
        popup_w = max(250, min(max_w, 450))

        self._completer_popup.setFixedSize(popup_w, popup_h)

        # Position below the text cursor, flip upward if off-screen
        rect = self.caption_edit.cursorRect()  # type: ignore[attr-defined]
        pt = self.caption_edit.viewport().mapToGlobal(rect.bottomLeft())  # type: ignore[attr-defined]
        screen = QtWidgets.QApplication.screenAt(pt)
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        screen_bottom = screen.availableGeometry().bottom()
        if pt.y() + popup_h > screen_bottom:
            pt.setY(self.caption_edit.viewport().mapToGlobal(rect.topLeft()).y() - popup_h)  # type: ignore[attr-defined]

        self._completer_popup.move(pt)
        self._completer_popup.show()
        self._completer_popup.raise_()
        self._completer_active = True

    def _hide_completer(self) -> None:
        """Hide the suggestion popup."""
        if hasattr(self, '_completer_popup'):
            self._completer_popup.hide()
        self._completer_active = False

    def _on_completer_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Accept the clicked suggestion."""
        self._accept_completion(item.text())

    # ---------- completion ----------------------------------------------------

    def _accept_completion(self, tag: str) -> None:
        """Replace the current token with *tag* and hide the popup."""
        self._hide_completer()
        _token, token_start, token_end = self._get_current_token()

        text = self.caption_edit.toPlainText()  # type: ignore[attr-defined]
        before = text[:token_start]
        after = text[token_end:]

        # Normalise trailing separator
        after = after.lstrip()
        if after.startswith(','):
            after = after[1:].lstrip()

        new_text = before + tag + (", " + after if after else ", ")
        new_pos = len(before) + len(tag) + 2  # after the ", "

        self._completing = True
        self.caption_edit.setPlainText(new_text)  # type: ignore[attr-defined]
        cursor = self.caption_edit.textCursor()  # type: ignore[attr-defined]
        cursor.setPosition(min(new_pos, len(new_text)))
        self.caption_edit.setTextCursor(cursor)  # type: ignore[attr-defined]
        self._completing = False

    # ---------- public helpers ------------------------------------------------

    def _insert_tag_at_cursor(self, tag: str) -> None:
        """Public helper – delegates to _accept_completion."""
        self._accept_completion(tag)

    def _on_tag_selected(self, tag: str) -> None:
        """Insert selected tag into caption at cursor position."""
        cursor = self.caption_edit.textCursor()  # type: ignore[attr-defined]
        text = self.caption_edit.toPlainText()  # type: ignore[attr-defined]
        cursor_pos = cursor.position()

        # Find the start of the current tag (after last comma or start of text)
        text_before_cursor = text[:cursor_pos]
        last_comma = text_before_cursor.rfind(',')

        if last_comma == -1:
            tag_start = 0
        else:
            tag_start = last_comma + 1
            while tag_start < len(text_before_cursor) and text_before_cursor[tag_start] == ' ':
                tag_start += 1

        # Find the end of current tag (next comma or end of text)
        tag_text = text[tag_start:]
        next_comma = tag_text.find(',')
        if next_comma == -1:
            tag_end = len(text)
        else:
            tag_end = tag_start + next_comma

        # Build new text
        before = text[:tag_start]
        after = text[tag_end:]
        after = after.lstrip()
        if after.startswith(','):
            after = after.lstrip(', ')

        new_text = before + tag
        if after:
            new_text += ", " + after
        else:
            new_text += ", "

        # Update text (block signals to prevent recursive calls)
        self.caption_edit.blockSignals(True)  # type: ignore[attr-defined]
        self.caption_edit.setPlainText(new_text)  # type: ignore[attr-defined]
        self.caption_edit.blockSignals(False)  # type: ignore[attr-defined]

        # Position cursor after the inserted tag and ", "
        cursor = self.caption_edit.textCursor()  # type: ignore[attr-defined]
        new_pos = tag_start + len(tag) + 2  # +2 for ", "
        if new_pos > len(new_text):
            new_pos = len(new_text)
        cursor.setPosition(new_pos)
        self.caption_edit.setTextCursor(cursor)  # type: ignore[attr-defined]

    # ---------- event filter (call from host eventFilter) ---------------------

    def _completer_event_filter(self, obj, event) -> bool:
        """Route keyboard navigation to the completer popup.

        Call this from the host's ``eventFilter`` override **before**
        falling through to ``super().eventFilter()``.
        """
        if obj is not getattr(self, 'caption_edit', None):
            return False
        if not hasattr(self, '_completer_popup'):
            return False

        if event.type() == QtCore.QEvent.KeyPress:
            popup = self._completer_popup
            if self._completer_active and popup.isVisible():
                key = event.key()
                if key == QtCore.Qt.Key_Down:
                    row = min(popup.currentRow() + 1, popup.count() - 1)
                    popup.setCurrentRow(row)
                    return True
                elif key == QtCore.Qt.Key_Up:
                    row = max(popup.currentRow() - 1, 0)
                    popup.setCurrentRow(row)
                    return True
                elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter,
                             QtCore.Qt.Key_Tab):
                    item = popup.currentItem()
                    if item:
                        self._accept_completion(item.text())
                    else:
                        self._hide_completer()
                    return True
                elif key == QtCore.Qt.Key_Escape:
                    self._hide_completer()
                    return True
        elif event.type() == QtCore.QEvent.FocusOut:
            QtCore.QTimer.singleShot(100, self._hide_completer)

        return False
