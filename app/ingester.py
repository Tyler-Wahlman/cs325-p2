from pathlib import Path
from typing import List, Dict

from interfaces import DocumentReader, TextChunkerInterface, VectorStoreInterface, EmbeddingModelInterface
from document_reader import PdfDocumentReader, DocxDocumentReader, TxtDocumentReader
from text_chunker import TextChunker
from vector_store import ChromaVectorStore
from embedding_model import SentenceTransformerEmbedding


class Ingester:
    """
    Orchestrates the full document ingestion pipeline.

    Responsibilities (SRP): loading documents, chunking them, embedding them,
    and storing them in the vector store. Each of those sub-tasks is delegated
    to a focused dependency injected via the constructor (DIP).

    Parameters
    ----------
    data_dir : Path
        Directory containing source documents.
    readers : dict
        Mapping of file extension (e.g. '.pdf') to a DocumentReader instance.
    chunker : TextChunkerInterface
        Responsible for splitting text into chunks.
    embedding_model : EmbeddingModelInterface
        Responsible for encoding text into vectors.
    vector_store : VectorStoreInterface
        Responsible for persisting chunks and embeddings.
    """

    def __init__(
        self,
        data_dir: Path,
        readers: Dict[str, DocumentReader],
        chunker: TextChunkerInterface,
        embedding_model: EmbeddingModelInterface,
        vector_store: VectorStoreInterface,
    ):
        self.data_dir = data_dir
        self.readers = readers
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def load_documents(self) -> List[Dict]:
        """Load all supported documents from the data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        documents = []
        for file_path in self.data_dir.iterdir():
            ext = file_path.suffix.lower()
            reader = self.readers.get(ext)
            if reader:
                documents.extend(reader.read(file_path))

        return documents

    def build_chunks(self, documents: List[Dict]) -> List[Dict]:
        """Convert loaded documents into smaller chunks with metadata."""
        chunked_docs = []

        for doc in documents:
            chunks = self.chunker.chunk(doc["text"])

            for idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "id": f"{doc['source']}_{doc['page']}_{idx}",
                    "text": chunk,
                    "metadata": {
                        "source": doc["source"],
                        "page": doc["page"],
                        "chunk_index": idx,
                        "file_type": doc["file_type"],
                    },
                })

        return chunked_docs

    def run(self) -> None:
        """Execute the full ingestion pipeline: load → chunk → embed → store."""
        documents = self.load_documents()
        chunked_docs = self.build_chunks(documents)

        ids = [doc["id"] for doc in chunked_docs]
        texts = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]
        embeddings = self.embedding_model.encode(texts).tolist()

        self.vector_store.add(ids, texts, metadatas, embeddings)


def build_default_ingester(data_dir: Path = Path("data")) -> Ingester:
    """
    Factory function that wires up the default Ingester with concrete implementations.
    Makes it easy to swap any component without touching Ingester itself.
    """
    readers = {
        ".pdf": PdfDocumentReader(),
        ".docx": DocxDocumentReader(),
        ".txt": TxtDocumentReader(),
    }
    chunker = TextChunker()
    embedding_model = SentenceTransformerEmbedding()
    vector_store = ChromaVectorStore()

    return Ingester(
        data_dir=data_dir,
        readers=readers,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
