from __future__ import annotations

import argparse
import json
from typing import Any

from diplomski.console import configure_console_output
from diplomski.retriever import ChromaRetriever
from diplomski.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)


def search_documents(
    query: str,
    k: int = DEFAULT_TOP_K,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    cloud_host: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    candidate_pool_size: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    use_hybrid_search: bool = True,
    group_by_source: bool = True,
    group_by_document_limit: int = DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
) -> list[dict[str, Any]]:
    """Find top-k most relevant documents from Chroma Cloud."""

    retriever = ChromaRetriever(
        collection_name=collection_name,
        cloud_host=cloud_host,
        tenant=tenant,
        database=database,
        candidate_pool_size=candidate_pool_size,
        use_hybrid_search=use_hybrid_search,
        group_by_source=group_by_source,
        group_by_document_limit=group_by_document_limit,
    )
    documents = retriever.retrieve(query, k=k)

    return [
        {
            "id": document.document_id,
            "score": document.score,
            "chroma_score": document.chroma_score,
            "distance": document.distance,
            "search_type": document.search_type,
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
        print(f"score: {_format_optional_float(result.get('score'))}")
        print(f"chroma_score: {_format_optional_float(result.get('chroma_score'))}")
        print(f"distance: {_format_optional_float(result.get('distance'))}")
        print(f"search_type: {result.get('search_type')}")
        print(f"source: {_metadata_display_value(metadata, 'source', 'sources')}")
        print(f"file_name: {_metadata_display_value(metadata, 'file_name', 'file_names')}")
        print(f"medicine_name: {metadata.get('medicine_name')}")
        print(f"active_substance: {metadata.get('active_substance')}")
        print(f"page: {_metadata_display_value(metadata, 'page_number', 'page_numbers')}")
        print(f"section: {metadata.get('section_title')}")
        print(f"content_type: {metadata.get('content_type')}")
        print(f"category: {metadata.get('category')}")
        print("-" * 80)
        print(result["text"][:preview_chars])


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "None"

    return f"{float(value):.4f}"


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
        description="Search top-k documents from the Chroma Cloud RAG store.",
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
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma Cloud collection name.",
    )
    parser.add_argument(
        "--cloud-host",
        default=None,
        help="Chroma Cloud host. Defaults to CHROMA_HOST or api.trychroma.com.",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Chroma Cloud tenant. Defaults to CHROMA_TENANT.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Chroma Cloud database. Defaults to CHROMA_DATABASE.",
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
        help="Number of dense/sparse candidates before RRF and GroupBy.",
    )
    parser.add_argument(
        "--group-by-document-limit",
        type=int,
        default=DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
        help="Maximum chunks kept per source document when GroupBy is enabled.",
    )
    parser.add_argument(
        "--no-hybrid-search",
        action="store_true",
        help="Use dense query fallback instead of Cloud hybrid RRF search.",
    )
    parser.add_argument(
        "--no-group-by-source",
        action="store_true",
        help="Do not group/deduplicate results by source document.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_console_output()

    args = _parse_args()
    query_text = " ".join(args.query)

    search_results = search_documents(
        query=query_text,
        k=args.top_k,
        collection_name=args.collection_name,
        cloud_host=args.cloud_host,
        tenant=args.tenant,
        database=args.database,
        candidate_pool_size=args.candidate_pool_size,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        group_by_document_limit=args.group_by_document_limit,
    )
    print_results(search_results, preview_chars=args.preview_chars)
