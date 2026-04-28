from typing import List

from interfaces import TextChunkerInterface


class TextChunker(TextChunkerInterface):
    """
    Splits a body of text into overlapping word-based chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of words per chunk.
    overlap : int
        Number of words to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks and return as a list of strings."""
        words = text.split()
        chunks = []

        if not words:
            return chunks

        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words).strip()
            if chunk:
                chunks.append(chunk)

        return chunks
