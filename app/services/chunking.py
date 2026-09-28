from typing import List

from app.config import get_settings


def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def paragraph_chunking(
    text: str,
    max_length: int = 500
) -> List[str]:
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0.")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

        if len(candidate) <= max_length:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(text: str, strategy: str) -> List[str]:
    strategy = strategy.lower()
    settings = get_settings()

    if strategy == "fixed":
        return fixed_size_chunking(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if strategy == "paragraph":
        return paragraph_chunking(text=text, max_length=settings.paragraph_max_length)

    raise ValueError("Invalid chunking strategy. Use 'fixed' or 'paragraph'.")