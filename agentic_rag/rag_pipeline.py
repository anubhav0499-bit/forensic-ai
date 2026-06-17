"""
LlamaIndex RAG Pipeline
========================

References
----------
Liu, J. (2022). "LlamaIndex: A data framework for your LLM applications."
    github.com/run-llama/llama_index. VectorStoreIndex + ChromaVectorStore
    integration used for financial document retrieval.

Ma, X., et al. (2023). "Query Rewriting for Retrieval-Augmented Large
    Language Models." arXiv:2305.14283.
    HyDE (Hypothetical Document Embedding): generate a hypothetical
    answer first, then embed it as the query — improves sparse-query recall.

Gao, L., et al. (2022). "Precise Zero-Shot Dense Retrieval without
    Relevance Labels." arXiv:2212.10496.
    Formal HyDE paper: embedding a hypothetical passage outperforms
    embedding the question directly on most BEIR benchmarks.

Provided Services
------------------
Context-aware financial document retrieval backed by the existing ChromaDB
vector store.  Provides:

  • VectorStoreIndex via ChromaVectorStore (same DB the rest of the platform uses)
  • HyDE (Hypothetical Document Embedding) query transform for better recall
  • Optional cross-encoder reranking via FlagEmbedding or SentenceTransformers
  • Falls back gracefully when LlamaIndex packages are not installed — returns
    an empty response object so callers don't need try/except everywhere.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

from config import DATA_DIR, EMBEDDING_CONFIG


class _FallbackResponse:
    """Minimal duck-type of a LlamaIndex Response when the library is absent."""
    response: str = ""
    source_nodes: list = []


class LlamaIndexRAGPipeline:
    """
    LlamaIndex-backed RAG pipeline.

    Usage::

        pipeline = LlamaIndexRAGPipeline(company_name="Infosys")
        result   = pipeline.query("Infosys", "revenue recognition policy FY2023")
        print(result.response)          # formatted context string
        print(result.source_nodes)      # list of NodeWithScore
    """

    def __init__(
        self,
        company_name: str = "",
        persist_dir: Path | None = None,
        similarity_top_k: int = 8,
        similarity_cutoff: float = 0.35,
    ):
        self.company_name   = company_name
        self.persist_dir    = persist_dir or (DATA_DIR / "chroma_db")
        self.top_k          = similarity_top_k
        self.cutoff         = similarity_cutoff
        self._index         = None
        self._query_engine  = None

    # ── Public interface ────────────────────────────────────────────────

    def query(self, company_name: str, query: str) -> _FallbackResponse:
        """
        Query the LlamaIndex pipeline for *company_name* using *query*.
        Returns a LlamaIndex Response (or _FallbackResponse on error).
        """
        if company_name != self.company_name:
            self.company_name = company_name
            self._index       = None
            self._query_engine = None

        try:
            if self._query_engine is None:
                self._build_query_engine()
            hyde_query = self._apply_hyde(query)
            return self._query_engine.query(hyde_query)
        except Exception as exc:
            logger.warning(f"[LlamaIndex] query failed for '{company_name}': {exc}")
            return _FallbackResponse()

    def index_documents(self, documents: list[dict], company_name: str) -> None:
        """
        Insert new documents (list of {text, metadata} dicts) into the index.
        Re-uses the existing ChromaDB collection so nothing is duplicated.
        """
        try:
            from llama_index.core import Document
        except ImportError:
            logger.warning("[LlamaIndex] llama-index not installed; skipping index_documents")
            return

        self.company_name  = company_name
        self._index        = None
        self._build_index()

        docs = [
            Document(text=d.get("text", ""), metadata=d.get("metadata", {}))
            for d in documents
            if d.get("text", "").strip()
        ]
        for doc in docs:
            self._index.insert(doc)
        logger.info(f"[LlamaIndex] inserted {len(docs)} documents for '{company_name}'")

    # ── Internal builders ───────────────────────────────────────────────

    def _build_index(self) -> None:
        try:
            from llama_index.core import Settings, VectorStoreIndex
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core.storage.storage_context import StorageContext
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "LlamaIndex packages not installed. "
                "Run: pip install llama-index llama-index-embeddings-huggingface "
                "llama-index-vector-stores-chroma"
            ) from exc

        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_CONFIG.model_name,
            cache_folder=str(DATA_DIR / ".cache" / "hf"),
        )
        Settings.embed_model  = embed_model
        Settings.chunk_size   = 512
        Settings.chunk_overlap = 64
        Settings.llm          = None   # We manage LLM separately via LangChain

        chroma_client   = chromadb.PersistentClient(path=str(self.persist_dir))
        collection_name = self._collection_name()

        try:
            collection = chroma_client.get_collection(collection_name)
            logger.debug(f"[LlamaIndex] using existing collection '{collection_name}'")
        except Exception:
            collection = chroma_client.get_or_create_collection(collection_name)
            logger.debug(f"[LlamaIndex] created collection '{collection_name}'")

        vector_store    = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index     = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    def _build_query_engine(self) -> None:
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.postprocessor import SimilarityPostprocessor

        if self._index is None:
            self._build_index()

        retriever    = VectorIndexRetriever(index=self._index, similarity_top_k=self.top_k)
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=self.cutoff)]

        # Cross-encoder reranker — try two implementations, skip if neither is installed
        for _load_reranker in [self._load_flag_reranker, self._load_sentence_reranker]:
            reranker = _load_reranker()
            if reranker:
                postprocessors.append(reranker)
                break

        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors,
        )

    @staticmethod
    def _load_flag_reranker():
        try:
            from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
            return FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=5)
        except Exception:
            return None

    @staticmethod
    def _load_sentence_reranker():
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            return SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5
            )
        except Exception:
            return None

    def _apply_hyde(self, query: str) -> str:
        """
        Hypothetical Document Embedding (HyDE): ask the LLM to generate a
        synthetic 2-sentence excerpt that *would* answer the query, then use
        that excerpt as the retrieval query for better semantic alignment.
        """
        try:
            from llm.langchain_client import lc_invoke
            hyde_prompt = (
                f"Write a 2-sentence hypothetical financial statement excerpt that would "
                f"directly answer this query about {self.company_name}:\n{query}\n\n"
                "Return ONLY the excerpt, no preamble."
            )
            return lc_invoke(hyde_prompt, fast=True)
        except Exception:
            return query

    def _collection_name(self) -> str:
        slug = self.company_name.lower().replace(" ", "_")[:40] if self.company_name else "global"
        return f"forensic_ai_{slug}"
