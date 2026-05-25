from pypdf import PdfReader
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a given PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks for better semantic retrieval.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks