"""Generate THIRD_PARTY_NOTICES.txt for the packaged executable.

The PyInstaller build bundles every runtime dependency into Img-Tagbooru.exe.
MIT, BSD, Apache, MPL and LGPL all require the copyright notice and licence
text to accompany a binary distribution, so this script walks the dependency
graph from requirements.txt, collects each package's licence file(s) from the
installed dist-info metadata and writes them into one notices file that is
added to the build and shown from the About dialog.

Run from the project root with the project's virtual environment:

    .venv\\Scripts\\python.exe scripts\\generate_third_party_notices.py

Re-run it whenever requirements.txt changes.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "THIRD_PARTY_NOTICES.txt"

# Build-time tools that never end up inside the executable.
EXCLUDE = {"pyinstaller", "pyinstaller-hooks-contrib", "pip", "setuptools", "wheel"}

HEADER = """\
Img-Tagbooru - Third-Party Notices
Generated {today}

Img-Tagbooru itself is (c) Szymon Ruszkiewicz and released under the MIT
License (see LICENSE). The Windows executable additionally bundles the
open-source components listed below. Each is the property of its respective
authors and is distributed under its own licence, reproduced here as required
by those licences.

QT / PYSIDE6 - LGPL NOTICE
--------------------------
This application uses the Qt framework through PySide6 and Shiboken6, which
are used under the GNU Lesser General Public License version 3 (LGPL-3.0).
Qt is (c) The Qt Company Ltd. and other contributors. The Qt libraries are
dynamically linked and are shipped as separate DLL files inside the
executable's extraction directory, so you may replace them with a compatible
version of your own. The complete LGPL-3.0 text is included below under the
PySide6 entry. Qt source code is available from https://www.qt.io/ and
https://code.qt.io/.

MODELS AND DATA (not bundled - downloaded or installed by the user)
------------------------------------------------------------------
* WD tagger models (wd-swinv2-tagger-v3 and siblings) by SmilingWolf,
  Apache License 2.0, downloaded from https://huggingface.co/SmilingWolf on
  request.
* Danbooru tag vocabulary (danbooru_tags_post_count.csv): tag names and post
  counts from https://danbooru.donmai.us/. Danbooru's Terms of Service state
  that tags are factual information and not copyrightable. No images are
  included.
* Language and vision models run through Ollama (JoyCaption, Qwen, Mistral
  and others named in the README) are installed by the user and governed by
  their own licences, including the Meta Llama 3.1 Community License for
  Llama-derived models.

BUNDLED PYTHON PACKAGES
=======================
"""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_name(spec: str) -> str | None:
    spec = spec.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return _norm(match.group(1)) if match else None


def _roots() -> list[str]:
    names: list[str] = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = _req_name(line)
        if name and name not in EXCLUDE:
            names.append(name)
    return names


def _closure(roots: list[str]) -> dict[str, md.Distribution]:
    found: dict[str, md.Distribution] = {}
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in found or name in EXCLUDE:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        found[name] = dist
        for req in dist.requires or []:
            # Skip optional extras; only unconditional runtime requirements.
            if "extra ==" in req:
                continue
            child = _req_name(req)
            if child and child not in found:
                pending.append(child)
    return found


def _license_name(dist: md.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return classifiers[0].split("::")[-1].strip()
    lic = (meta.get("License") or "").strip()
    return lic.splitlines()[0][:80] if lic else "see licence text below"


def _license_texts(dist: md.Distribution) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for f in dist.files or []:
        upper = str(f).upper()
        if any(k in upper for k in ("LICENSE", "LICENCE", "COPYING", "NOTICE")):
            try:
                content = dist.locate_file(f).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if content.strip():
                texts.append((str(f), content.strip()))
    if not texts:
        lic = (dist.metadata.get("License") or "").strip()
        if len(lic) > 120:
            texts.append(("METADATA License field", lic))
    return texts


def main() -> int:
    dists = _closure(_roots())
    lines = [HEADER.format(today=date.today().isoformat())]

    lines.append("Summary\n-------")
    for name in sorted(dists):
        d = dists[name]
        lines.append(f"{d.metadata['Name']} {d.version}: {_license_name(d)}")
    lines.append("")

    for name in sorted(dists):
        d = dists[name]
        title = f"{d.metadata['Name']} {d.version}"
        lines.append("=" * 78)
        lines.append(title)
        lines.append(f"Licence: {_license_name(d)}")
        home = d.metadata.get("Home-page") or ""
        if not home:
            for url in d.metadata.get_all("Project-URL") or []:
                if "github" in url.lower() or "homepage" in url.lower():
                    home = url.split(",", 1)[-1].strip()
                    break
        if home:
            lines.append(f"Source: {home}")
        lines.append("=" * 78)
        texts = _license_texts(d)
        if not texts:
            lines.append("(no licence file shipped in the wheel; see the licence named above)")
        for fname, content in texts:
            lines.append(f"--- {fname} ---")
            lines.append(content)
            lines.append("")
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(dists)} packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
