from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diplomski.settings import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)
from diplomski.vector_store import ChromaVectorStore


QUERY_STOPWORDS = {
    "a",
    "ako",
    "da",
    "i",
    "ili",
    "je",
    "koja",
    "koje",
    "koji",
    "na",
    "od",
    "o",
    "se",
    "su",
    "u",
    "za",
}
COMMON_SECTION_TERMS = {
    "dejstva",
    "dejstvo",
    "nezeljena",
    "nezeljeno",
    "lek",
    "leka",
    "leku",
    "informacije",
}


@dataclass
class RetrievedDocument:
    """One retrieved document with text, metadata and similarity score."""

    text: str
    metadata: dict[str, Any]
    score: float
    distance: float
    document_id: str | None = None
    vector_score: float | None = None
    lexical_score: float = 0.0


class ChromaRetriever:
    """Retrieve top-k documents from the local ChromaDB vector store."""

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        candidate_pool_size: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    ) -> None:
        self.candidate_pool_size = candidate_pool_size
        self.vector_store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            device=device,
        )

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedDocument]:
        """Return the k most relevant documents for a query."""

        document_count = self.vector_store.count()
        if document_count == 0:
            print("[WARN] ChromaDB collection is empty. Build the vector store first.")
            return []

        candidate_count = min(
            max(k, self.candidate_pool_size),
            document_count,
        )
        results = self.vector_store.query(query, top_k=candidate_count)
        results = _rerank_results(query, results)[:k]

        return [
            RetrievedDocument(
                text=result["text"],
                metadata=result["metadata"],
                score=result["score"],
                distance=result["distance"],
                document_id=result.get("id"),
                vector_score=result.get("vector_score"),
                lexical_score=result.get("lexical_score", 0.0),
            )
            for result in results
        ]


def _rerank_results(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = _query_terms(query)
    if not query_terms:
        return results

    reranked: list[dict[str, Any]] = []

    for result in results:
        reranked_result = dict(result)
        vector_score = float(reranked_result.get("score", 0.0))
        lexical_score = _lexical_score(query_terms, reranked_result)

        reranked_result["vector_score"] = vector_score
        reranked_result["lexical_score"] = lexical_score
        reranked_result["score"] = vector_score + lexical_score
        reranked.append(reranked_result)

    return sorted(reranked, key=lambda item: item["score"], reverse=True)


def _lexical_score(query_terms: list[str], result: dict[str, Any]) -> float:
    metadata = result.get("metadata") or {}
    text = _normalize_text(result.get("text") or "")
    metadata_text = _normalize_text(_metadata_search_text(metadata))

    score = 0.0
    for term in query_terms:
        weight = 0.04 if term in COMMON_SECTION_TERMS else 0.14
        variants = _term_variants(term)

        if any(variant in text for variant in variants):
            score += weight

        if any(variant in metadata_text for variant in variants):
            score += weight * 1.5

    entity_terms = [term for term in query_terms if term not in COMMON_SECTION_TERMS]
    section_terms = [term for term in query_terms if term in COMMON_SECTION_TERMS]

    if (
        entity_terms
        and section_terms
        and _matches_any_term(entity_terms, text, metadata_text)
        and _matches_any_term(section_terms, text, metadata_text)
    ):
        score += 0.12

    return min(score, 0.65)


def _query_terms(query: str) -> list[str]:
    normalized_query = _normalize_text(query)
    terms: list[str] = []

    for token in re.findall(r"[a-z0-9]+", normalized_query):
        if len(token) < 3:
            continue
        if token in QUERY_STOPWORDS:
            continue
        if token in terms:
            continue

        terms.append(token)

    return terms


def _term_variants(term: str) -> set[str]:
    variants = {term}

    for suffix in ("ima", "ama", "om", "em", "og", "oj", "a", "e", "i", "u"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 5:
            variants.add(term[: -len(suffix)])

    for variant in list(variants):
        if variant.endswith("in") and len(variant) >= 7:
            variants.add(variant[:-2])

    return variants


def _matches_any_term(terms: list[str], text: str, metadata_text: str) -> bool:
    searchable = f"{text} {metadata_text}"
    return any(
        variant in searchable
        for term in terms
        for variant in _term_variants(term)
    )


def _metadata_search_text(metadata: dict[str, Any]) -> str:
    values: list[str] = []

    for key in ("source", "sources", "file_name", "file_names", "folder", "folders"):
        value = metadata.get(key)
        values.extend(_flatten_metadata_value(value))

    return " ".join(values)


def _flatten_metadata_value(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _flatten_metadata_value(parsed)

    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_metadata_value(item))
        return values

    return [str(value)]


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))
