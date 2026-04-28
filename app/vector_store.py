from typing import List, Dict

import chromadb

from interfaces import VectorStoreInterface


class ChromaVectorStore(VectorStoreInterface):
    """
    Concrete implementation of VectorStoreInterface using ChromaDB.

    Parameters
    ----------
    chroma_dir : str
        Path where ChromaDB persists its data on disk.
    collection_name : str
        Name of the ChromaDB collection to use.
    """

    def __init__(self, chroma_dir: str = "chroma_db", collection_name: str = "project_docs"):
        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict], embeddings: List) -> None:
        """Store document chunks with their embeddings and metadata in ChromaDB."""
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], top_k: int) -> Dict:
        """Query ChromaDB for the top_k most semantically similar chunks."""
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._collection.count()
