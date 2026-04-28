import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from unittest.mock import MagicMock, patch, mock_open
 
from interfaces import (
    DocumentReader,
    TextChunkerInterface,
    VectorStoreInterface,
    EmbeddingModelInterface,
    LLMServiceInterface,
)
from text_chunker import TextChunker
from document_reader import PdfDocumentReader, DocxDocumentReader, TxtDocumentReader, clean_text
from ingester import Ingester
from rag_pipeline import RAGPipeline
 
 
# =============================================================================
# Mock Implementations (demonstrate DIP — swap real services for fakes in tests)
# =============================================================================
 
class MockEmbeddingModel(EmbeddingModelInterface):
    """Returns a fixed dummy vector instead of running a real model."""
 
    def encode(self, texts):
        mock = MagicMock()
        mock.tolist.return_value = [0.1, 0.2, 0.3]
        return mock
 
 
class MockVectorStore(VectorStoreInterface):
    """In-memory vector store that records what was added and returns fake results."""
 
    def __init__(self):
        self.added = []
 
    def add(self, ids, texts, metadatas, embeddings):
        for i in range(len(ids)):
            self.added.append({"id": ids[i], "text": texts[i], "metadata": metadatas[i]})
 
    def query(self, query_embedding, top_k):
        return {
            "documents": [["Goblins are small green creatures that ambush travelers."]],
            "metadatas": [[{"source": "monster_bestiary.docx", "page": 1}]],
        }
 
 
class MockLLMService(LLMServiceInterface):
    """Returns a hardcoded answer without making any real API call."""
 
    def generate(self, prompt: str) -> str:
        return "Goblins use ambush tactics and travel in groups."
 
 
# =============================================================================
# clean_text
# =============================================================================
 
class TestCleanText:
 
    def test_removes_extra_spaces(self):
        assert clean_text("hello   world") == "hello world"
 
    def test_removes_newlines(self):
        assert clean_text("hello\nworld") == "hello world"
 
    def test_removes_tabs(self):
        assert clean_text("hello\tworld") == "hello world"
 
    def test_empty_string(self):
        assert clean_text("") == ""
 
    def test_already_clean(self):
        assert clean_text("hello world") == "hello world"
 
 
# =============================================================================
# TextChunker
# =============================================================================
 
class TestTextChunker:
 
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=5, overlap=1)
 
    def test_returns_list(self):
        result = self.chunker.chunk("one two three four five")
        assert isinstance(result, list)
 
    def test_empty_text_returns_empty_list(self):
        assert self.chunker.chunk("") == []
 
    def test_short_text_single_chunk(self):
        result = self.chunker.chunk("one two three")
        assert len(result) == 1
        assert result[0] == "one two three"
 
    def test_chunk_size_respected(self):
        # 10 words, chunk_size=5, overlap=1 → step=4 → chunks at 0, 4, 8
        text = "a b c d e f g h i j"
        result = self.chunker.chunk(text)
        for chunk in result:
            assert len(chunk.split()) <= 5
 
    def test_overlap_creates_multiple_chunks(self):
        text = "a b c d e f g h i j"
        result = self.chunker.chunk(text)
        assert len(result) > 1
 
    def test_implements_interface(self):
        assert isinstance(self.chunker, TextChunkerInterface)
 
 
# =============================================================================
# TxtDocumentReader
# =============================================================================
 
class TestTxtDocumentReader:
 
    def setup_method(self):
        self.reader = TxtDocumentReader()
 
    def test_implements_interface(self):
        assert isinstance(self.reader, DocumentReader)
 
    def test_reads_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello world from a text file.")
        result = self.reader.read(f)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world from a text file."
        assert result[0]["source"] == "sample.txt"
        assert result[0]["file_type"] == "txt"
        assert result[0]["page"] == 1
 
    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   ")
        result = self.reader.read(f)
        assert result == []
 
    def test_normalizes_whitespace(self, tmp_path):
        f = tmp_path / "messy.txt"
        f.write_text("Hello   \n\n   world")
        result = self.reader.read(f)
        assert result[0]["text"] == "Hello world"
 
 
# =============================================================================
# MockVectorStore
# =============================================================================
 
class TestMockVectorStore:
 
    def setup_method(self):
        self.store = MockVectorStore()
 
    def test_implements_interface(self):
        assert isinstance(self.store, VectorStoreInterface)
 
    def test_add_stores_documents(self):
        self.store.add(
            ids=["doc1"],
            texts=["Some text"],
            metadatas=[{"source": "test.txt", "page": 1}],
            embeddings=[[0.1, 0.2]],
        )
        assert len(self.store.added) == 1
        assert self.store.added[0]["id"] == "doc1"
 
    def test_query_returns_expected_structure(self):
        result = self.store.query([0.1, 0.2, 0.3], top_k=1)
        assert "documents" in result
        assert "metadatas" in result
        assert len(result["documents"][0]) == 1
 
 
# =============================================================================
# MockLLMService
# =============================================================================
 
class TestMockLLMService:
 
    def setup_method(self):
        self.llm = MockLLMService()
 
    def test_implements_interface(self):
        assert isinstance(self.llm, LLMServiceInterface)
 
    def test_generate_returns_string(self):
        result = self.llm.generate("What are goblins?")
        assert isinstance(result, str)
        assert len(result) > 0
 
    def test_generate_does_not_call_external_api(self):
        # If this runs without network/Ollama, the mock is working correctly
        result = self.llm.generate("Any question")
        assert result == "Goblins use ambush tactics and travel in groups."
 
 
# =============================================================================
# Ingester
# =============================================================================
 
class TestIngester:
 
    def setup_method(self):
        self.mock_store = MockVectorStore()
        self.mock_embedding = MockEmbeddingModel()
        self.mock_chunker = TextChunker(chunk_size=10, overlap=2)
 
    def _make_ingester(self, data_dir):
        return Ingester(
            data_dir=data_dir,
            readers={".txt": TxtDocumentReader()},
            chunker=self.mock_chunker,
            embedding_model=self.mock_embedding,
            vector_store=self.mock_store,
        )
 
    def test_load_documents_reads_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world this is a test document.")
        ingester = self._make_ingester(tmp_path)
        docs = ingester.load_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "test.txt"
 
    def test_load_documents_skips_unsupported_files(self, tmp_path):
        (tmp_path / "ignore.csv").write_text("a,b,c")
        (tmp_path / "keep.txt").write_text("Valid content here.")
        ingester = self._make_ingester(tmp_path)
        docs = ingester.load_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "keep.txt"
 
    def test_load_documents_raises_if_dir_missing(self):
        ingester = self._make_ingester(Path("/nonexistent/path"))
        with pytest.raises(FileNotFoundError):
            ingester.load_documents()
 
    def test_build_chunks_produces_correct_ids(self, tmp_path):
        docs = [{
            "text": "word " * 20,
            "source": "sample.txt",
            "page": 1,
            "file_type": "txt",
        }]
        ingester = self._make_ingester(tmp_path)
        chunks = ingester.build_chunks(docs)
        assert all("sample.txt" in c["id"] for c in chunks)
 
    def test_run_stores_chunks_in_vector_store(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("The goblin is a small creature. " * 5)
        ingester = self._make_ingester(tmp_path)
        ingester.run()
        assert len(self.mock_store.added) > 0
 
 
# =============================================================================
# RAGPipeline
# =============================================================================
 
class TestRAGPipeline:
 
    def setup_method(self):
        self.pipeline = RAGPipeline(
            vector_store=MockVectorStore(),
            embedding_model=MockEmbeddingModel(),
            llm_service=MockLLMService(),
            top_k=1,
        )
 
    def test_retrieve_returns_documents(self):
        results = self.pipeline.retrieve("What are goblins?")
        assert "documents" in results
        assert "metadatas" in results
 
    def test_build_context_joins_chunks(self):
        results = {
            "documents": [["Chunk one.", "Chunk two."]],
            "metadatas": [[{"source": "a.txt", "page": 1}, {"source": "b.txt", "page": 2}]],
        }
        context, metadatas = self.pipeline.build_context(results)
        assert "Chunk one." in context
        assert "Chunk two." in context
 
    def test_build_prompt_contains_query(self):
        prompt = self.pipeline.build_prompt("What are goblins?", "Some context.")
        assert "What are goblins?" in prompt
        assert "Some context." in prompt
 
    def test_format_citations_deduplicates(self):
        metadatas = [
            {"source": "bestiary.docx", "page": 1},
            {"source": "bestiary.docx", "page": 1},
            {"source": "core_rules.pdf", "page": 2},
        ]
        citations = self.pipeline.format_citations(metadatas)
        lines = citations.strip().split("\n")
        assert len(lines) == 2
 
    def test_ask_returns_answer_and_sources(self):
        result = self.pipeline.ask("What are goblins?")
        assert "Goblins use ambush tactics" in result
        assert "Sources:" in result
        assert "monster_bestiary.docx" in result
 
    def test_ask_uses_mock_llm_not_real_ollama(self):
        # This test passes without Ollama running — proves DIP is working
        result = self.pipeline.ask("Anything")
        assert isinstance(result, str)