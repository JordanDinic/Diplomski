from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diplomski.console import configure_console_output
from diplomski.settings import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)
from diplomski.retriever import ChromaRetriever


def search_documents(
    query: str,
    k: int = DEFAULT_TOP_K,
    persist_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    device: str = DEFAULT_EMBEDDING_DEVICE,
    candidate_pool_size: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Find top-k most relevant documents from the existing ChromaDB store."""

    retriever = ChromaRetriever(
        persist_dir=persist_dir,
        collection_name=collection_name,
        device=device,
        candidate_pool_size=candidate_pool_size,
    )
    documents = retriever.retrieve(query, k=k)

    return [
        {
            "id": document.document_id,
            "score": document.score,
            "vector_score": document.vector_score,
            "lexical_score": document.lexical_score,
            "distance": document.distance,
            "text": document.text,
            "metadata": document.metadata,
        }
        for document in documents
    ]


def print_results(results: list[dict[str, Any]], preview_chars: int = 1000) -> None:
    """Print retrieval results in a readable terminal format."""

    if not results:
        print("[INFO] No results found.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print("=" * 80)
        print(f"Result #{index}")
        print(f"score: {result['score']:.4f}")
        print(f"vector_score: {result.get('vector_score')}")
        print(f"lexical_score: {result.get('lexical_score')}")
        print(f"distance: {result['distance']:.4f}")
        print(f"source: {_metadata_display_value(metadata, 'source', 'sources')}")
        print(f"file_name: {_metadata_display_value(metadata, 'file_name', 'file_names')}")
        print(f"page: {_metadata_display_value(metadata, 'page_number', 'page_numbers')}")
        print(f"content_type: {metadata.get('content_type')}")
        print(f"category: {metadata.get('category')}")
        print("-" * 80)
        print(result["text"][:preview_chars])


def _metadata_display_value(
    metadata: dict[str, Any],
    primary_key: str,
    fallback_key: str,
) -> Any:
    value = metadata.get(primary_key)
    if value is not None:
        return value

    fallback = metadata.get(fallback_key)
    if isinstance(fallback, str):
        try:
            return json.loads(fallback)
        except json.JSONDecodeError:
            return fallback

    return fallback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search top-k documents from the ChromaDB RAG vector store.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query. Quotes are optional; words are joined with spaces.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_CHROMA_DIR,
        help="Path to the ChromaDB persistent directory.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1000,
        help="Number of characters to print from each result.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
        help="Number of vector candidates to rerank before selecting top-k.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_EMBEDDING_DEVICE,
        help="Embedding device: auto, cpu, cuda, cuda:0...",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_console_output()

    args = _parse_args()
    query_text = " ".join(args.query)

    search_results = search_documents(
        query=query_text,
        k=args.top_k,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        device=args.device,
        candidate_pool_size=args.candidate_pool_size,
    )
    print_results(search_results, preview_chars=args.preview_chars)
