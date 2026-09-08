"""PDF ingestion: documents become pages of text with stable character offsets.

Page-level granularity is deliberate. It is small enough to fit comfortably in a
prompt, it gives every extracted claim a citable page number for free, and it
makes grounding a simple substring search against the exact text the model was
shown.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# Ligatures and typographic punctuation that survive PDF extraction and would
# otherwise cause a verbatim quote to "not match" its own source page.
_TRANSLATIONS = str.maketrans(
    {
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-", " ": " ",
        "‐": "-", "‑": "-", "﻿": "",
    }
)

_WS = re.compile(r"\s+")
# A page number printed on the page itself, alone on a line near the edges.
_PRINTED_PAGE = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)


def normalise(text: str) -> str:
    """Fold a string into the form used for grounding comparisons."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATIONS)
    return _WS.sub(" ", text).strip().lower()


@dataclass
class Page:
    page_no: int          # 1-based position in the PDF
    text: str             # raw extracted text, offsets refer to this
    printed_page: str | None = None
    _norm: str | None = field(default=None, repr=False)

    @property
    def norm_text(self) -> str:
        if self._norm is None:
            self._norm = normalise(self.text)
        return self._norm

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Document:
    doc_id: str
    filename: str
    path: str
    pages: list[Page]
    sha256: str
    title: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, page_no: int) -> Page | None:
        idx = page_no - 1
        return self.pages[idx] if 0 <= idx < len(self.pages) else None


def _printed_page_number(text: str) -> str | None:
    """Best-effort read of the page number printed on the page.

    Our excerpts keep the original documents' numbering, so PDF position and
    printed page diverge. Citations should quote what a reader would see.
    """
    head, tail = text[:200], text[-200:]
    for chunk in (tail, head):
        matches = _PRINTED_PAGE.findall(chunk)
        if matches:
            return matches[-1]
    return None


def load_pdf(path: str | Path, doc_id: str | None = None) -> Document:
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    # Content-addressed id: re-uploading the same file is idempotent, which is
    # what makes incremental ingestion safe.
    doc_id = doc_id or sha[:12]

    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or None
        pages = [
            Page(
                page_no=i + 1,
                text=p.get_text(),
                printed_page=_printed_page_number(p.get_text()),
            )
            for i, p in enumerate(pdf)
        ]

    return Document(
        doc_id=doc_id,
        filename=path.name,
        path=str(path),
        pages=pages,
        sha256=sha,
        title=title.strip() if title else None,
    )
