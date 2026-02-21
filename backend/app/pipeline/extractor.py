"""
ClearPath RAG Chatbot — PDF Text Extraction

Uses PyMuPDF (fitz) to extract text from all PDFs in the docs/ directory.
Returns a list of Document objects with filename, page_number, and text.
"""

import os
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF


@dataclass
class Document:
    """Represents one page of extracted text from a PDF."""
    filename: str
    page_number: int
    text: str


def extract_all_pdfs(docs_dir: str = "docs") -> List[Document]:
    """
    Extract text from all PDF files in the given directory.

    Args:
        docs_dir: Path to directory containing PDF files.

    Returns:
        List of Document objects, one per page.
    """
    documents: List[Document] = []

    # Get sorted list of PDF files
    pdf_files = sorted(
        f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}/")

    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # skip empty pages
                    documents.append(Document(
                        filename=filename,
                        page_number=page_num + 1,  # 1-indexed
                        text=text.strip(),
                    ))
            doc.close()
        except Exception as e:
            print(f"[WARN] Failed to extract {filename}: {e}")

    return documents


if __name__ == "__main__":
    # Quick test: extract and show summary
    docs = extract_all_pdfs()
    print(f"Extracted {len(docs)} pages from {len(set(d.filename for d in docs))} PDFs")
    for d in docs[:3]:
        print(f"  {d.filename} p{d.page_number}: {d.text[:80]}...")
