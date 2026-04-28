from pathlib import Path
from typing import List, Dict

from PyPDF2 import PdfReader
from docx import Document

from interfaces import DocumentReader


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    return " ".join(text.split())


class PdfDocumentReader(DocumentReader):
    """Reads text content from PDF files, one entry per page."""

    def read(self, file_path: Path) -> List[Dict]:
        pages_data = []
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)

            if text.strip():
                pages_data.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_num,
                    "file_type": "pdf",
                })

        return pages_data


class DocxDocumentReader(DocumentReader):
    """Reads text content from .docx files as a single entry."""

    def read(self, file_path: Path) -> List[Dict]:
        doc = Document(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = clean_text("\n".join(paragraphs))

        if not full_text.strip():
            return []

        return [{
            "text": full_text,
            "source": file_path.name,
            "page": 1,
            "file_type": "docx",
        }]


class TxtDocumentReader(DocumentReader):
    """Reads text content from plain .txt files as a single entry."""

    def read(self, file_path: Path) -> List[Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = clean_text(f.read())

        if not text.strip():
            return []

        return [{
            "text": text,
            "source": file_path.name,
            "page": 1,
            "file_type": "txt",
        }]
