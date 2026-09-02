from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from langchain_core.documents import Document

from diplomski.console import configure_console_output
from diplomski.data_loader import load_all_elements
from diplomski.embedding_pipeline import EmbeddingPipeline
from diplomski.settings import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
)


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store for the RAG demo.

    The embedding pipeline prepares Unstructured PDF elements and creates
    embeddings. This class only handles storing, loading and searching vectors.
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_characters: int = 1000,
        new_after_n_chars: int = 800,
        overlap: int = 100,
        table_context_elements: int = 2,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        store_batch_size: int = 16,
        device: str = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.store_batch_size = store_batch_size
        self.pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            overlap=overlap,
            table_context_elements=table_context_elements,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            device=device,
        )
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._get_or_create_collection()

        print(
            f"[INFO] ChromaDB collection '{self.collection_name}' "
            f"loaded from '{self.persist_dir}'."
        )

    def build_from_elements(
        self,
        elements: list[Any],
        *,
        reset_collection: bool = True,
    ) -> list[Document]:
        """
        Build the vector store from raw Unstructured elements.

        This is the usual path:
        load_all_elements() -> EmbeddingPipeline -> ChromaDB.
        """

        print(f"[INFO] Building vector store from {len(elements)} raw elements...")
        documents = self.pipeline.documents_from_elements(elements)
        self.build_from_documents(documents, reset_collection=reset_collection)
        return documents

    def build_from_documents(
        self,
        documents: list[Document],
        *,
        reset_collection: bool = True,
    ) -> None:
        """Embed LangChain Documents incrementally and store them in ChromaDB."""

        print(f"[INFO] Building vector store from {len(documents)} documents...")

        if reset_collection:
            self.reset()

        if not documents:
            print("[WARN] No documents to store.")
            return

        total_documents = len(documents)
        for start_index in range(0, total_documents, self.store_batch_size):
            end_index = min(start_index + self.store_batch_size, total_documents)
            document_batch = documents[start_index:end_index]

            print(
                "[INFO] Embedding and storing documents "
                f"{start_index + 1}-{end_index}/{total_documents}..."
            )
            embeddings = self.pipeline.embed_documents(document_batch)
            self.add_documents(
                document_batch,
                embeddings,
                id_offset=start_index,
            )

        print(f"[INFO] Vector store built with {self.count()} documents.")

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray | None = None,
        *,
        reset_collection: bool = False,
        id_offset: int = 0,
    ) -> None:
        """Add LangChain Documents and their embeddings to ChromaDB."""

        if reset_collection:
            self.reset()

        final_embeddings = embeddings
        if final_embeddings is None:
            final_embeddings = self.pipeline.embed_documents(documents)

        if len(documents) != len(final_embeddings):
            raise ValueError(
                "Number of documents and embeddings must match: "
                f"{len(documents)} documents, {len(final_embeddings)} embeddings."
            )

        if not documents:
            print("[WARN] No documents to store.")
            return

        self.collection.upsert(
            ids=[
                _document_id(document, id_offset + index)
                for index, document in enumerate(documents)
            ],
            documents=[document.page_content for document in documents],
            metadatas=[_chroma_metadata(document.metadata) for document in documents],
            embeddings=[embedding.tolist() for embedding in final_embeddings],
        )

        print(f"[INFO] Added {len(documents)} documents to ChromaDB.")

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return search results in a simple dictionary format."""

        print(f"[INFO] Querying vector store for: {query_text!r}")
        document_count = self.count()
        if document_count == 0:
            print("[WARN] ChromaDB collection is empty.")
            return []

        query_embedding = self.pipeline.embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, document_count),
            include=["documents", "metadatas", "distances"],
        )

        return _format_query_results(results)

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Return top-k LangChain Documents with cosine similarity scores."""

        results = self.query(query_text, top_k=top_k)
        matches: list[tuple[Document, float]] = []

        for result in results:
            metadata = result["metadata"] or {}
            document = Document(
                page_content=result["text"],
                metadata=metadata,
            )
            matches.append((document, result["score"]))

        return matches

    def reset(self) -> None:
        """Delete and recreate the configured ChromaDB collection."""

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self._get_or_create_collection()
        print(f"[INFO] Reset ChromaDB collection '{self.collection_name}'.")

    def count(self) -> int:
        """Return the number of stored vectors."""

        return self.collection.count()

    def _get_or_create_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _format_query_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted: list[dict[str, Any]] = []
    for item_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        distance_value = float(distance)
        formatted.append(
            {
                "id": item_id,
                "score": 1.0 - distance_value,
                "distance": distance_value,
                "text": text,
                "metadata": metadata or {},
            }
        )

    return formatted


def _document_id(document: Document, index: int) -> str:
    metadata = document.metadata
    raw_id = "|".join(
        str(part)
        for part in (
            metadata.get("source"),
            metadata.get("page_number"),
            metadata.get("element_index"),
            metadata.get("chunk_index"),
            metadata.get("element_id"),
            document.page_content[:200],
            index,
        )
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:24]

    return f"doc_{digest}"


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    chroma_metadata: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            chroma_metadata[key] = value
            continue

        chroma_metadata[key] = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return chroma_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector store from PDF elements.",
    )
    parser.add_argument(
        "--data-dir",
        default="Literatura/Lekovi",
        help="Directory with PDF files.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_CHROMA_DIR,
        help="ChromaDB persistent directory.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="SentenceTransformer encode batch size.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        help="Maximum token length for the embedding model.",
    )
    parser.add_argument(
        "--store-batch-size",
        type=int,
        default=16,
        help="Number of documents to embed and store per outer batch.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_EMBEDDING_DEVICE,
        help="Embedding device: auto, cpu, cuda, cuda:0...",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append/upsert into the existing collection instead of resetting it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_console_output()

    args = _parse_args()
    elements = load_all_elements(args.data_dir)

    store = ChromaVectorStore(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        store_batch_size=args.store_batch_size,
        device=args.device,
    )
    documents = store.build_from_elements(
        elements,
        reset_collection=not args.no_reset,
    )

    print("[INFO] Stored documents:", store.count())
    print("[INFO] Example document:", documents[0].page_content[:500] if documents else None)

    results = store.query("Koje su najvaznije informacije o amlodipinu?", top_k=3)
    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result #{index}")
        print(f"score: {result['score']:.4f}")
        print(f"source: {result['metadata'].get('source')}")
        print(result["text"][:500])
