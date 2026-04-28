from typing import List, Dict

from interfaces import VectorStoreInterface, EmbeddingModelInterface, LLMServiceInterface
from vector_store import ChromaVectorStore
from embedding_model import SentenceTransformerEmbedding
from llm_service import OllamaService


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) query pipeline.

    Depends on abstractions (DIP), not concrete implementations. Any component
    can be swapped (e.g. MockLLMService for testing) without modifying this class.

    Parameters
    ----------
    vector_store : VectorStoreInterface
        Used to retrieve semantically similar document chunks.
    embedding_model : EmbeddingModelInterface
        Used to encode the user query into a vector.
    llm_service : LLMServiceInterface
        Used to generate a natural language answer from context.
    top_k : int
        Number of chunks to retrieve from the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedding_model: EmbeddingModelInterface,
        llm_service: LLMServiceInterface,
        top_k: int = 3,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_service = llm_service
        self.top_k = top_k

    def retrieve(self, query: str) -> Dict:
        """Encode the query and retrieve the top_k matching chunks."""
        query_embedding = self.embedding_model.encode(query).tolist()
        return self.vector_store.query(query_embedding, self.top_k)

    def build_context(self, results: Dict):
        """Combine retrieved chunks into a single context string with metadata."""
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        context = "\n\n".join(documents)
        return context, metadatas

    def build_prompt(self, query: str, context: str) -> str:
        """Construct the prompt to send to the LLM."""
        return f"""You are a helpful tabletop RPG assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{query}"""

    def format_citations(self, metadatas: List[Dict]) -> str:
        """Format retrieved metadata into deduplicated source citations."""
        seen = set()
        citations = []

        for meta in metadatas:
            citation = f"{meta.get('source')} (page {meta.get('page')})"
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)

        return "\n".join(citations)

    def ask(self, query: str) -> str:
        """
        Run the full RAG pipeline for a given query.
        Retrieve → build context → generate answer → attach citations.
        """
        results = self.retrieve(query)
        context, metadatas = self.build_context(results)
        prompt = self.build_prompt(query, context)
        answer = self.llm_service.generate(prompt)
        citations = self.format_citations(metadatas)

        return f"{answer}\n\nSources:\n{citations}"


def build_default_pipeline() -> RAGPipeline:
    """
    Factory function that wires up the default RAGPipeline with concrete implementations.
    Makes it easy to swap any component without touching RAGPipeline itself.
    """
    return RAGPipeline(
        vector_store=ChromaVectorStore(),
        embedding_model=SentenceTransformerEmbedding(),
        llm_service=OllamaService(),
    )
