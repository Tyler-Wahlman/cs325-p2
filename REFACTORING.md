REFACTORING.md
Overview
This document describes the refactoring changes made to the Project 1 codebase to meet the software quality standards required for Project 2. The refactoring focused on applying two SOLID principles: Single Responsibility Principle (S) and Dependency Inversion Principle (D).

Principle 1: Single Responsibility Principle (SRP)
Which Principle
S — Single Responsibility Principle: A class should have only one reason to change.

The Problem in Project 1
In Project 1, ingest.py was a single file responsible for everything in the ingestion pipeline:

Reading PDF files
Reading DOCX files
Reading TXT files
Cleaning and normalizing text
Chunking text into overlapping segments
Encoding text into embedding vectors
Writing chunks and embeddings to ChromaDB

If any single part of the pipeline changed — for example, swapping ChromaDB for a different vector database, or adding a new file type — the entire file had to be modified. This violates SRP because there were many different reasons for one file to change.
Similarly, rag_pipeline.py handled retrieval, context building, prompt construction, LLM generation, and citation formatting all in one place.

The Fix
Each responsibility was extracted into its own focused class in its own file:
FileSingle Responsibilitydocument_reader.pyReading files (PDF, DOCX, TXT)text_chunker.pySplitting text into overlapping chunksembedding_model.pyEncoding text into vectorsvector_store.pyStoring and querying ChromaDBllm_service.pyCommunicating with the Ollama LLMingester.pyOrchestrating the ingestion pipelinerag_pipeline.pyOrchestrating the RAG query pipeline
Now each file has exactly one reason to change. For example, if the chunking strategy changes, only text_chunker.py is touched. If ChromaDB is swapped, only vector_store.py changes.

Before and After
Before — ingest.py handled reading, chunking, embedding, and storing all in one file:
pythondef read_pdf(file_path: Path) -> List[Dict]:
    pages_data = []
    reader = PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text.strip():
            pages_data.append({
                "text": text,
                "source": file_path.name,
                "page": page_num,
                "file_type": "pdf",
            })
    return pages_data

def chunk_text(text: str, chunk_size=100, overlap=20) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def create_vector_store(chunked_docs: List[Dict]) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embeddings = embedding_model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
After — each responsibility lives in its own focused class:
python# document_reader.py
class PdfDocumentReader(DocumentReader):
    def read(self, file_path: Path) -> List[Dict]:
        ...  # only responsible for reading PDFs

# text_chunker.py
class TextChunker(TextChunkerInterface):
    def chunk(self, text: str) -> List[str]:
        ...  # only responsible for chunking text

# vector_store.py
class ChromaVectorStore(VectorStoreInterface):
    def add(self, ids, texts, metadatas, embeddings) -> None:
        ...  # only responsible for storing to ChromaDB

# embedding_model.py
class SentenceTransformerEmbedding(EmbeddingModelInterface):
    def encode(self, texts) -> List:
        ...  # only responsible for encoding text to vectors

Principle 2: Dependency Inversion Principle (DIP)
Which Principle
D — Dependency Inversion Principle: High-level modules should not depend on low-level modules. Both should depend on abstractions.

The Problem in Project 1
In Project 1, rag_pipeline.py directly instantiated concrete implementations at the module level:
python# Hardcoded at the top of rag_pipeline.py
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
ollama_client = ollama.Client(host="http://host.docker.internal:11434")
And the functions used them directly:
pythondef retrieve_relevant_chunks(query, top_k=TOP_K):
    collection = get_collection()
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

def generate_answer(query, context):
    response = ollama_client.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
This meant:

The pipeline was impossible to test without a real Ollama server running
The pipeline was impossible to test without a real ChromaDB instance
Swapping the LLM or vector database required rewriting core pipeline logic
There was no way to inject a mock for automated testing


The Fix
An interfaces.py file was introduced containing abstract base classes for every external dependency:
python# interfaces.py
class LLMServiceInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class VectorStoreInterface(ABC):
    @abstractmethod
    def add(self, ids, texts, metadatas, embeddings) -> None:
        pass

    @abstractmethod
    def query(self, query_embedding, top_k) -> Dict:
        pass

class EmbeddingModelInterface(ABC):
    @abstractmethod
    def encode(self, texts) -> List:
        pass
The RAGPipeline class now depends only on these abstractions, never on concrete implementations:
python# rag_pipeline.py
class RAGPipeline:
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
Concrete implementations are wired together in a factory function, keeping them separate from pipeline logic:
pythondef build_default_pipeline() -> RAGPipeline:
    return RAGPipeline(
        vector_store=ChromaVectorStore(),
        embedding_model=SentenceTransformerEmbedding(),
        llm_service=OllamaService(),
    )

How DIP Enables Testing
Because RAGPipeline depends on abstractions, mock implementations can be injected in tests without any real services running:
python# tests/test_rag.py
class MockLLMService(LLMServiceInterface):
    def generate(self, prompt: str) -> str:
        return "Goblins use ambush tactics and travel in groups."

class MockVectorStore(VectorStoreInterface):
    def query(self, query_embedding, top_k):
        return {
            "documents": [["Goblins are small green creatures."]],
            "metadatas": [[{"source": "monster_bestiary.docx", "page": 1}]],
        }

def test_ask_uses_mock_llm_not_real_ollama():
    pipeline = RAGPipeline(
        vector_store=MockVectorStore(),
        embedding_model=MockEmbeddingModel(),
        llm_service=MockLLMService(),
    )
    result = pipeline.ask("What are goblins?")
    assert "Goblins use ambush tactics" in result
This test passes with no Ollama server and no ChromaDB instance — proving that the high-level pipeline is fully decoupled from its low-level dependencies.

Summary of Changes
AreaBeforeAfterFile count3 files9 filesIngestionAll logic in ingest.pySplit across document_reader.py, text_chunker.py, embedding_model.py, vector_store.py, ingester.pyRAG pipelineHardcoded concrete dependenciesDepends on abstract interfacesLLMollama.Client called directlyInjected via LLMServiceInterfaceVector DBchromadb called directlyInjected via VectorStoreInterfaceTestabilityRequired real Ollama + ChromaDBFully testable with mocks

Author
CS325 - Project 2
Tyler Wahlman
Retrieval-Augmented Generation Assistant