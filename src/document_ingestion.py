"""Generic document extraction and chunking for AdaptIQ.

Supported inputs: PDF, DOCX and PPTX. Existing S2-specific RAG scripts are
left untouched; this module is the reusable foundation for uploaded material.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}


def _read_bytes(source: str | Path | bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        # Streamlit UploadedFile and file-like objects both expose read().
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return data
    return Path(source).read_bytes()


def extract_document(source: str | Path | bytes | bytearray | BinaryIO,
                     filename: str | None = None) -> list[dict[str, str]]:
    """Extract non-empty text units, retaining page/slide/paragraph provenance.

    Returns dictionaries with `source_id`, `text`, and `location`.
    """
    if filename is None:
        filename = getattr(source, "name", None)
    if filename is None and isinstance(source, (str, Path)):
        filename = Path(source).name
    if not filename:
        raise ValueError("filename is required when source is raw bytes.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or 'unknown'}. "
            "Upload a PDF, DOCX or PPTX file."
        )

    raw = _read_bytes(source)
    if not raw:
        raise ValueError("The uploaded document is empty.")

    units: list[dict[str, str]] = []
    stream = BytesIO(raw)

    if extension == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(stream)
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                units.append({"source_id": f"page_{page_number}",
                              "location": f"Page {page_number}", "text": text})

    elif extension == ".docx":
        from docx import Document
        document = Document(stream)
        for index, paragraph in enumerate(document.paragraphs, start=1):
            text = paragraph.text.strip()
            if text:
                units.append({"source_id": f"paragraph_{index}",
                              "location": f"Paragraph {index}", "text": text})
        # Include table contents, which are common in lecture notes.
        for table_index, table in enumerate(document.tables, start=1):
            for row_index, row in enumerate(table.rows, start=1):
                text = " | ".join(cell.text.strip() for cell in row.cells).strip()
                if text.replace("|", "").strip():
                    units.append({"source_id": f"table_{table_index}_row_{row_index}",
                                  "location": f"Table {table_index}, row {row_index}",
                                  "text": text})

    else:  # .pptx
        from pptx import Presentation
        presentation = Presentation(stream)
        for slide_number, slide in enumerate(presentation.slides, start=1):
            pieces = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    text = shape.text.strip()
                    if text:
                        pieces.append(text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        pieces.append(" | ".join(cell.text.strip() for cell in row.cells))
            text = "\n".join(pieces).strip()
            if text:
                units.append({"source_id": f"slide_{slide_number}",
                              "location": f"Slide {slide_number}", "text": text})

    if not units:
        raise ValueError(
            "No extractable text was found. The file may be scanned or image-only; "
            "OCR support is not included in this first version."
        )
    return units


def chunk_document(units: list[dict[str, str]], max_chars: int = 1200,
                   overlap: int = 150) -> list[dict[str, str]]:
    """Create bounded text chunks while retaining original source locations."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars.")

    chunks: list[dict[str, str]] = []
    for unit in units:
        text = unit["text"].strip()
        start = 0
        part = 1
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Prefer a paragraph/sentence boundary near the end of the window.
            if end < len(text):
                boundary = max(text.rfind("\n", start, end),
                               text.rfind(". ", start, end))
                if boundary > start + max_chars // 2:
                    end = boundary + (1 if text[boundary] == "\n" else 2)
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": f"{unit['source_id']}_chunk_{part}",
                    "source_id": unit["source_id"],
                    "location": unit["location"],
                    "content": piece,
                })
                part += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
