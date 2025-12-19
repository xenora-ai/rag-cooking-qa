# src/chunking/splitter.py
from typing import List


def split_into_chunks(text: str, max_len: int = 200, overlap: int = 50) -> List[str]:
    """
    Розбиває текст на чанки довжиною max_len слів з overlap (перекриттям)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_len
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_len - overlap  # наступний блок з перекриттям
    return chunks
