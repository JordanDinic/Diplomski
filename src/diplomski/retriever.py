from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diplomski.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)
from diplomski.vector_store import ChromaVectorStore


@dataclass
class RetrievedDocument:
    """One retrieved document with text, metadata and Chroma search score."""

    text: str
    metadata: dict[str, Any]
    score: float
    document_id: str | None = None
    chroma_score: float | None = None
    distance: float | None = None
    search_type: str | None = None


class ChromaRetriever:
    """Retrieve top-k child chunks from Chroma Cloud without parent expansion."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        cloud_host: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        candidate_pool_size: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
        use_hybrid_search: bool = True,
        group_by_source: bool = True,
        group_by_document_limit: int = DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.candidate_pool_size = candidate_pool_size
        self.use_hybrid_search = use_hybrid_search
        self.group_by_source = group_by_source

        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=collection_name,
            cloud_host=cloud_host,
            tenant=tenant,
            database=database,
            use_hybrid_search=use_hybrid_search,
            group_by_source=group_by_source,
            group_by_document_limit=group_by_document_limit,
        )

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedDocument]:
        """Return up to k child chunks, keeping their text, metadata and scores."""

        results = self.vector_store.query(
            query,
            top_k=k,
            candidate_count=self.candidate_pool_size,
            use_hybrid_search=self.use_hybrid_search,
            group_by_source=self.group_by_source,
            expand_to_parent=False,
        )

        return [
            RetrievedDocument(
                text=result["text"],
                metadata=result["metadata"],
                score=result["score"],
                document_id=result.get("id"),
                chroma_score=result.get("chroma_score"),
                distance=result.get("distance"),
                search_type=result.get("search_type"),
            )
            for result in results
        ]
