"""Entry point for Img-Tagbooru desktop application.

Run with:
    python run.py
or (no console window):
    pythonw run.py
"""
from __future__ import annotations

import sys
import traceback
import logging
from pathlib import Path

# Ensure project root is on sys.path before any local imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Write uncaught exceptions to a log file so pythonw crashes are visible
_log_path = PROJECT_ROOT / "crash.log"

def _excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        _log_path.write_text(msg, encoding="utf-8")
    except Exception:
        pass
    # Also try to show a message box if Qt is available
    try:
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app:
            QtWidgets.QMessageBox.critical(None, "Crash", msg[:1000])
    except Exception:
        pass

sys.excepthook = _excepthook

from frontend.native.main_window import main

if __name__ == "__main__":
    main()
