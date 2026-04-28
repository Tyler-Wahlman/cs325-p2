from abc import ABC, abstractmethod
from typing import List, Dict


class DocumentReader(ABC):
    """Abstract base class for reading documents of different file types."""

    @abstractmethod
    def read(self, file_path) -> List[Dict]:
        """
        Read a document and return a list of page dictionaries.
        Each dict must contain: 'text', 'source', 'page', 'file_type'.
        """
        pass


class TextChunkerInterface(ABC):
    """Abstract base class for splitting text into chunks."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into a list of string chunks."""
        pass


class VectorStoreInterface(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict], embeddings: List) -> None:
        """Store documents with their embeddings and metadata."""
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> Dict:
        """Query the vector store and return the top_k matching results."""
        pass


class EmbeddingModelInterface(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts) -> List:
        """Convert text or list of texts into embedding vectors."""
        pass


class LLMServiceInterface(ABC):
    """Abstract base class for LLM backends (Ollama, OpenAI, mock, etc.)."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response string."""
        pass
