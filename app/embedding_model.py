from typing import List

from sentence_transformers import SentenceTransformer

from interfaces import EmbeddingModelInterface


class SentenceTransformerEmbedding(EmbeddingModelInterface):
    """
    Concrete implementation of EmbeddingModelInterface using a HuggingFace
    SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        The SentenceTransformer model to load.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def encode(self, texts) -> List:
        """Encode a string or list of strings into embedding vectors."""
        return self._model.encode(texts)
