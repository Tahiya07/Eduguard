"""
ingestion.py
==============================================================================

Document ingestion for the local academic assistant.

Public learning material can be chunked as raw text for RAG. Protected exam
material is tagged at ingestion time so retrieval and generation layers can
route it through teacher-only workflows and output screening.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:  # progress messages for the UI's waiting dialog (no-op outside a request)
    from operation_control import report_progress
except Exception:  # pragma: no cover - standalone use of this module
    def report_progress(*_args, **_kwargs) -> None:
        return None

# Chunking is deterministic; no RNG seeding is needed. (This module used to import PyTorch only to
# seed it, which added seconds to the first upload and to every module that imports DocumentChunk.)

try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    fitz = None
    _HAS_PYMUPDF = False

logger = logging.getLogger("ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TEXT_SUFFIXES = {".txt", ".md", ".csv"}


def resolve_tesseract_cmd() -> Optional[str]:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def ocr_backend_status() -> dict[str, object]:
    """Report whether image OCR can run (Tesseract or easyocr)."""
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        return {"available": False, "engine": "none", "reason": f"pillow_unavailable: {exc}"}

    tesseract_reason = "not_configured"
    cmd = resolve_tesseract_cmd()
    if cmd:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = cmd
            pytesseract.get_tesseract_version()
            return {"available": True, "engine": "pytesseract", "reason": "ok", "command": cmd}
        except Exception as exc:
            tesseract_reason = str(exc)

    try:
        import easyocr  # noqa: F401

        return {"available": True, "engine": "easyocr", "reason": "ok", "command": None}
    except Exception as exc:
        return {
            "available": False,
            "engine": "none",
            "reason": f"tesseract={tesseract_reason}; easyocr={exc}",
            "install_hint": (
                "Install Tesseract OCR and add it to PATH, or set TESSERACT_CMD to tesseract.exe. "
                "Windows: winget install UB-Mannheim.TesseractOCR. "
                "Alternatively pip install easyocr, or upload PDF/TXT and paste text instead of images."
            ),
        }


@dataclass
class DocumentChunk:
    chunk_id: int
    source: str
    text: str
    page: Optional[int] = None
    modality: str = "text"
    access_level: str = "public"
    content_type: str = "study_material"
    bloom_level: Optional[str] = None
    bloom_confidence: float = 0.0
    ordinal_risk: float = 0.0
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    safe_summary: Optional[str] = None


@dataclass
class RawDocumentChunk:
    chunk_id: int
    source: str
    text: str
    page: Optional[int]
    modality: str
    topic: Optional[str] = None


class DocumentIngestor:
    """Extract and chunk PDF, image, and text inputs for local RAG."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        normalize_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.normalize_whitespace = bool(normalize_whitespace)

    def _normalize(self, text: str) -> str:
        text = (text or "").replace("\x00", " ")
        if not self.normalize_whitespace:
            return text
        # Normalize spaces within each line but retain paragraph, heading, and
        # bullet boundaries. Flattening all whitespace makes CVs and structured
        # notes read like one run-on sentence before the summarizer sees them.
        lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    # A heading: a short line without sentence punctuation that is not a list item,
    # e.g. "Smart Contract Threats" or "3.2 Process Scheduling".
    _HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][^\n]{1,80}$")
    _LIST_ITEM = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25cf\u2013]|\d+[.)]|[a-z][.)])\s+")

    @classmethod
    def _is_heading(cls, line: str) -> bool:
        line = line.strip()
        if not line or len(line.split()) > 10 or line[-1] in ".,;!?:":
            return False
        if cls._LIST_ITEM.match(line) and not re.match(r"^\d+(?:\.\d+)+\s", line):
            return False
        return bool(cls._HEADING.match(line))

    @classmethod
    def _units(cls, normalized: str) -> List[Tuple[int, int, int, bool]]:
        """Sentence-sized spans (start, end, word_count, is_heading_line) of the normalized text.

        Lines are kept as boundaries (slides and lists are line-structured); long lines are split
        at sentence ends.
        """
        units: List[Tuple[int, int, int, bool]] = []
        offset = 0
        for line in normalized.split("\n"):
            start = offset
            offset += len(line) + 1
            if not line.strip():
                continue
            heading = cls._is_heading(line)
            cuts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line)] + [len(line)]
            for a, b in zip(cuts, cuts[1:]):
                raw = line[a:b]
                seg = raw.strip()
                if not seg:
                    continue
                s0 = start + a + (len(raw) - len(raw.lstrip()))
                units.append((s0, s0 + len(seg), len(seg.split()), heading))
        return units

    def _split_text_with_headings(
        self, text: str, heading: Optional[str] = None
    ) -> Tuple[List[Tuple[str, Optional[str]]], Optional[str]]:
        """Chunks of whole sentences/lines up to chunk_size words, overlapping by whole trailing
        sentences (at most chunk_overlap words). Returns [(chunk_text, section_heading)] and the
        heading in force at the end (carried over to the next page)."""
        normalized = self._normalize(text)
        units = self._units(normalized)
        if not units:
            return [], heading
        # A sentence longer than a whole chunk is cut into word windows.
        expanded: List[Tuple[int, int, int, bool]] = []
        for s0, e0, n, h in units:
            if n <= self.chunk_size:
                expanded.append((s0, e0, n, h))
                continue
            words = list(re.finditer(r"\S+", normalized[s0:e0]))
            for i in range(0, len(words), self.chunk_size):
                w = words[i:i + self.chunk_size]
                expanded.append((s0 + w[0].start(), s0 + w[-1].end(), len(w), False))
        units = expanded

        out: List[Tuple[str, Optional[str]]] = []
        current = heading
        i = 0
        while i < len(units):
            j, words = i, 0
            while j < len(units) and (j == i or words + units[j][2] <= self.chunk_size):
                words += units[j][2]
                j += 1
            # Do not end a chunk on a heading: it belongs with the text after it.
            while j - i > 1 and j < len(units) and units[j - 1][3]:
                j -= 1
            # A chunk of headings only takes the following sentence too, even past chunk_size.
            while j < len(units) and all(units[k][3] for k in range(i, j)):
                j += 1
            chunk_heading = current
            for k in range(i, j):
                if units[k][3]:
                    if k == i:
                        chunk_heading = normalized[units[k][0]:units[k][1]]
                    current = normalized[units[k][0]:units[k][1]]
            out.append((normalized[units[i][0]:units[j - 1][1]], chunk_heading))
            if j >= len(units):
                break
            # Overlap: repeat trailing whole sentences (never headings) worth <= chunk_overlap words.
            back, k = 0, j
            while k - 1 > i and not units[k - 1][3] and back + units[k - 1][2] <= self.chunk_overlap:
                back += units[k - 1][2]
                k -= 1
            i = k
        return out, current

    def _split_text(self, text: str) -> List[str]:
        return [chunk for chunk, _ in self._split_text_with_headings(text)[0]]

    def chunk_text(
        self,
        text: str,
        *,
        source: str = "<text>",
        modality: str = "text",
        access_level: str = "public",
        content_type: str = "study_material",
        page: Optional[int] = None,
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        pieces, _ = self._split_text_with_headings(text)
        for idx, (piece, heading) in enumerate(pieces):
            chunks.append(
                DocumentChunk(
                    chunk_id=idx,
                    source=source,
                    text=piece,
                    page=page,
                    modality=modality,
                    access_level=access_level,
                    content_type=content_type,
                    topic=heading,
                )
            )
        return chunks

    @staticmethod
    def _strip_running_lines(pages: List[str]) -> List[str]:
        """Remove page headers/footers (lines repeated on most pages) and bare page numbers."""
        def key(line: str) -> str:
            return re.sub(r"\d+", "#", line.strip().lower())

        def edges(page: str) -> List[str]:
            # Headers and footers are short lines at the very top or bottom of a page; body text
            # that happens to repeat across pages (a recurring definition) must stay.
            lines = [l for l in page.splitlines() if l.strip()]
            # Slide decks put the department/course banner under the title, so look at the top and
            # bottom three lines. Safe because only EXACT repeats (or "page N" lines) are removed.
            n = 3
            # A line starting in lowercase continues a wrapped sentence, so it is body text.
            return [l for l in lines[:n] + lines[-n:] if len(l.split()) <= 12 and not l.strip()[:1].islower()]

        page_number = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.I)
        # A line that differs between pages only by its number is a running line only when it says so
        # ("... page 7", "Slide 3 of 40"). Otherwise it is content: numbered slide titles such as
        # "Lecture section 3" / "Week 4" must survive (they were being removed as headers).
        page_word = re.compile(r"\b(?:page|pg\.?|p\.|slide)\s*\d", re.I)
        exact = lambda line: line.strip().lower()
        repeated: set = set()
        repeated_numbered: set = set()
        if len(pages) >= 4:
            need = max(3, 0.6 * len(pages))
            counts = Counter(k for page in pages for k in {exact(l) for l in edges(page)})
            repeated = {k for k, c in counts.items() if c >= need}
            counts_n = Counter(k for page in pages for k in {key(l) for l in edges(page) if page_word.search(l)})
            repeated_numbered = {k for k, c in counts_n.items() if c >= need}
        cleaned = []
        for page in pages:
            edge = {exact(l) for l in edges(page)}
            edge_n = {key(l) for l in edges(page)}
            lines = [l for l in page.splitlines()
                     if not page_number.match(l.strip())
                     and not (exact(l) in repeated and exact(l) in edge)
                     and not (key(l) in repeated_numbered and key(l) in edge_n)]
            cleaned.append("\n".join(lines))
        return cleaned

    # A PDF page with fewer words than this but with images is treated as scanned and OCR'd.
    OCR_MIN_WORDS = 15

    def _ocr_pdf_page(self, page) -> str:
        """Render one PDF page at 200 dpi and read it with Tesseract (no OCR available -> "")."""
        try:
            status = ocr_backend_status()
            if not status.get("available") or status.get("engine") != "pytesseract":
                return ""
            import io

            import pytesseract
            from PIL import Image

            pytesseract.pytesseract.tesseract_cmd = status.get("command") or resolve_tesseract_cmd()
            pix = page.get_pixmap(dpi=200)
            return str(pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png")))) or "")
        except Exception as exc:  # OCR is a best effort; a failed page is skipped as before
            logger.warning("OCR of PDF page %s failed: %s", getattr(page, "number", "?"), exc)
            return ""

    def _load_pdf_raw(self, path: Path) -> List[RawDocumentChunk]:
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF is required for PDF ingestion")
        doc = fitz.open(str(path))
        chunks: List[RawDocumentChunk] = []
        cid = 0
        try:
            report_progress(f"Reading the PDF's {len(doc)} page(s)...")
            # sort=True reads text boxes top-to-bottom, left-to-right (slides, two-column layouts).
            pages = [self._normalize(doc[i].get_text("text", sort=True) or "") for i in range(len(doc))]
            # Pages whose text is only a picture (scanned pages, screenshot slides) have no text layer
            # and used to be skipped silently. OCR those pages with the bundled Tesseract.
            for i, text in enumerate(pages):
                if len(text.split()) < self.OCR_MIN_WORDS and doc[i].get_images(full=False):
                    report_progress(f"Page {i + 1} of {len(pages)} is a scanned image; reading it with OCR...", i + 1, len(pages))
                    ocr = self._ocr_pdf_page(doc[i])
                    if len(ocr.split()) > len(text.split()):
                        pages[i] = self._normalize(ocr)
        finally:
            doc.close()
        heading: Optional[str] = None
        for page_i, text in enumerate(self._strip_running_lines(pages)):
            pieces, heading = self._split_text_with_headings(text, heading)
            for piece, chunk_heading in pieces:
                chunks.append(RawDocumentChunk(cid, str(path), piece, page_i + 1, "pdf", chunk_heading))
                cid += 1
        if not chunks:
            raise RuntimeError(
                f"No extractable text in {path.name}. Scanned/image-only PDFs need OCR — "
                "paste the text, or re-export the PDF with a text layer."
            )
        return chunks

    def _load_image_text_easyocr(self, path: Path) -> str:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        pieces = reader.readtext(str(path), detail=0)
        return " ".join(str(piece) for piece in pieces)

    def _load_image_text_tesseract(self, path: Path) -> str:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore

        cmd = resolve_tesseract_cmd()
        if not cmd:
            raise RuntimeError("tesseract executable not found")
        pytesseract.pytesseract.tesseract_cmd = cmd
        pytesseract.get_tesseract_version()
        return str(pytesseract.image_to_string(Image.open(path)) or "")

    def _load_image_text(self, path: Path) -> str:
        status = ocr_backend_status()
        if not status.get("available"):
            hint = status.get("install_hint") or status.get("reason")
            raise RuntimeError(f"Image OCR is unavailable. {hint}")

        engine = str(status.get("engine") or "")
        if engine == "pytesseract":
            try:
                return self._load_image_text_tesseract(path)
            except Exception as exc:
                logger.warning("Tesseract OCR failed for %s (%s); trying easyocr", path.name, exc)

        try:
            return self._load_image_text_easyocr(path)
        except Exception as exc:
            raise RuntimeError(
                f"Image OCR failed for {path.name}. Install Tesseract and add it to PATH "
                f"(or set TESSERACT_CMD), or pip install easyocr. Details: {exc}"
            ) from exc

    def process(
        self,
        path: Union[str, Path],
        *,
        access_level: str = "public",
        content_type: str = "study_material",
    ) -> List[DocumentChunk]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._load_pdf_raw(path)
            return [
                DocumentChunk(
                    chunk_id=raw.chunk_id,
                    source=raw.source,
                    text=raw.text,
                    page=raw.page,
                    modality=raw.modality,
                    access_level=access_level,
                    content_type=content_type,
                    topic=raw.topic,
                )
                for raw in raw_chunks
            ]
        if suffix in {".txt", ".md", ".csv"}:
            return self.chunk_text(
                path.read_text(encoding="utf-8", errors="ignore"),
                source=str(path),
                modality="text",
                access_level=access_level,
                content_type=content_type,
            )
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            text = self._load_image_text(path)
            if not self._normalize(text):
                raise RuntimeError(
                    f"No readable text was extracted from image '{path.name}'. "
                    "Try a clearer scan, paste the text directly, or upload a PDF/TXT file."
                )
            return self.chunk_text(
                text,
                source=str(path),
                modality="image",
                access_level=access_level,
                content_type=content_type,
            )
        raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")


def _self_test() -> None:
    ing = DocumentIngestor(chunk_size=5, chunk_overlap=1)
    chunks = ing.chunk_text("one two three four five six", source="demo")
    assert len(chunks) == 2
    assert chunks[0].text == "one two three four five"
    assert chunks[1].text == "six"
    print("[OK] ingestion self-test passed")


if __name__ == "__main__":
    _self_test()
