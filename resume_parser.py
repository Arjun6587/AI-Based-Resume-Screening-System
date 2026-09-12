import pymupdf
from docx import Document
from pathlib import Path


def extract_pdf_text(file_path):
    """Extract text from a PDF resume."""
    text = ""

    with pymupdf.open(file_path) as document:
        for page in document:
            text += page.get_text(sort=True) + "\n"

    return text.strip()


def extract_docx_text(file_path):
    """Extract text from a DOCX resume."""
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text.strip())

    return "\n".join(paragraphs)


def extract_resume_text(file_path):
    """Extract text from either PDF or DOCX resume."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    elif extension == ".txt":
        return file_path.read_text(encoding="utf-8").strip()

    else:
    
        raise ValueError(
            "Unsupported file format. Please use PDF or DOCX."
        )


if __name__ == "__main__":
    print("Resume Parser is ready!")