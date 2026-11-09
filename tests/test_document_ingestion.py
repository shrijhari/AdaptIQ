from io import BytesIO

import pytest
from docx import Document
from pptx import Presentation

from document_ingestion import extract_document, chunk_document


def test_extract_docx_paragraph_and_table():
    doc = Document()
    doc.add_paragraph("Thermodynamics studies energy transfer.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Entropy"
    table.cell(0, 1).text = "A state property"
    stream = BytesIO()
    doc.save(stream)

    units = extract_document(stream.getvalue(), "notes.docx")
    assert any("Thermodynamics" in unit["text"] for unit in units)
    assert any("Entropy" in unit["text"] for unit in units)


def test_extract_pptx_slide_text():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    textbox = slide.shapes.add_textbox(0, 0, 4000000, 1000000)
    textbox.text = "First Law of Thermodynamics"
    stream = BytesIO()
    presentation.save(stream)

    units = extract_document(stream.getvalue(), "lecture.pptx")
    assert units[0]["location"] == "Slide 1"
    assert "First Law" in units[0]["text"]


def test_reject_unsupported_file():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_document(b"hello", "notes.txt")


def test_chunking_preserves_source_and_respects_size():
    units = [{"source_id": "page_1", "location": "Page 1",
              "text": ("A useful sentence about learning. " * 100)}]
    chunks = chunk_document(units, max_chars=300, overlap=40)
    assert len(chunks) > 1
    assert all(chunk["source_id"] == "page_1" for chunk in chunks)
    assert all(len(chunk["content"]) <= 300 for chunk in chunks)


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_document([], max_chars=100, overlap=100)
