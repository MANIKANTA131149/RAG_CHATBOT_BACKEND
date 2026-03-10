import io
import re
from typing import Optional

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _extract_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return _extract_docx(file_bytes)
    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="replace")
    elif ext == "md":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        # Try UTF-8 as fallback
        return file_bytes.decode("utf-8", errors="replace")

def _extract_pdf(data: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")

def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"DOCX extraction failed: {e}")

def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    # Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
