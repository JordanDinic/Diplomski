from __future__ import annotations

import argparse
import os
from typing import Any, Iterable

import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig, VectorIndexConfig
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
    ChromaCloudQwenEmbeddingModel,
    ChromaCloudQwenEmbeddingTarget,
)
from chromadb.utils.embedding_functions.chroma_cloud_splade_embedding_function import (
    ChromaCloudSpladeEmbeddingModel,
)
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document

from diplomski.chunking import build_parent_child_documents
from diplomski.chroma_documents import (
    SEARCH_METADATA_KEYS,
    clean_metadata,
    document_id,
    format_parent_records,
    format_query_results,
    format_search_results,
    prepare_documents_for_chroma,
)
from diplomski.console import configure_console_output
from diplomski.data_loader import load_all_documents
from diplomski.settings import (
    DEFAULT_CHROMA_DATABASE,
    DEFAULT_CHROMA_HOST,
    DEFAULT_CHROMA_QWEN_TASK,
    DEFAULT_CHROMA_SPARSE_KEY,
    DEFAULT_CHROMA_STORE_BATCH_SIZE,
    DEFAULT_CHROMA_TENANT,
    DEFAULT_CHUNK_MAX_CHARACTERS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_HYBRID_DENSE_WEIGHT,
    DEFAULT_HYBRID_SPARSE_WEIGHT,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)

try:
    from chromadb import GroupBy, MinK
except ImportError:
    from chromadb.execution.expression.operator import GroupBy, MinK


CHILD_FILTER = {"record_type": "child"}
PARENT_FILTER = {"record_type": "parent"}

QWEN_RETRIEVAL_INSTRUCTIONS = {
    DEFAULT_CHROMA_QWEN_TASK: {
        ChromaCloudQwenEmbeddingTarget.DOCUMENTS: (
            "Represent this Serbian pharmacy document chunk for retrieval."
        ),
        ChromaCloudQwenEmbeddingTarget.QUERY: (
            "Represent this Serbian pharmacy question for retrieving relevant "
            "drug leaflet, interaction, or pharmacy practice passages."
        ),
    }
}


class ChromaVectorStore:
    """
    Chroma Cloud adapter for prepared RAG documents.

    This class does not parse PDFs and does not chunk text. It only manages the
    Cloud collection, uploads prepared LangChain Documents, and searches them.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        cloud_host: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        api_key: str | None = None,
        store_batch_size: int = DEFAULT_CHROMA_STORE_BATCH_SIZE,
        sparse_key: str = DEFAULT_CHROMA_SPARSE_KEY,
        use_hybrid_search: bool = True,
        group_by_source: bool = True,
        group_by_document_limit: int = DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
        dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_HYBRID_SPARSE_WEIGHT,
        reset_collection: bool = False,
    ) -> None:
        load_environment()

        self.collection_name = collection_name
        self.cloud_host = cloud_host_from_env(cloud_host)
        self.tenant = tenant or os.getenv("CHROMA_TENANT", DEFAULT_CHROMA_TENANT)
        self.database = database or os.getenv("CHROMA_DATABASE", DEFAULT_CHROMA_DATABASE)
        self.api_key = chroma_api_key(api_key)
        self.store_batch_size = store_batch_size
        self.sparse_key = sparse_key
        self.use_hybrid_search = use_hybrid_search
        self.group_by_source = group_by_source
        self.group_by_document_limit = group_by_document_limit
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        os.environ["CHROMA_API_KEY"] = self.api_key
        self.dense_embedding_function = qwen_embedding_function()
        self.sparse_embedding_function = splade_embedding_function()
        self.client = chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key,
            cloud_host=self.cloud_host,
        )

        if reset_collection:
            self.delete_collection_if_exists()

        self.collection = self.get_or_create_collection()
        print(
            f"[INFO] Chroma Cloud collection '{self.collection_name}' "
            f"loaded from database '{self.database}' on tenant '{self.tenant}'."
        )

    def get_or_create_collection(self) -> Any:
        """Create the Cloud collection with dense and sparse indexes if needed."""

        self.client.get_or_create_collection(
            name=self.collection_name,
            schema=collection_schema(
                dense_embedding_function=self.dense_embedding_function,
                sparse_embedding_function=self.sparse_embedding_function,
                sparse_key=self.sparse_key,
            ),
            embedding_function=None,
            metadata={
                "embedding_backend": "chroma_cloud",
                "dense_model": ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B.value,
                "sparse_model": ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1.value,
                "sparse_key": self.sparse_key,
            },
        )
        return self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.dense_embedding_function,
        )

    def reset(self) -> None:
        """Delete and recreate the configured collection."""

        self.delete_collection_if_exists()
        self.collection = self.get_or_create_collection()
        print(f"[INFO] Recreated Chroma Cloud collection '{self.collection_name}'.")

    def delete_collection_if_exists(self) -> None:
        """Delete the configured collection, ignoring the not-found case."""

        try:
            self.client.delete_collection(self.collection_name)
            print(f"[INFO] Deleted Chroma Cloud collection '{self.collection_name}'.")
        except Exception as exc:
            print(f"[INFO] Collection reset skipped before create: {exc}")

    def count(self) -> int:
        """Return the number of stored Chroma records."""

        return self.collection.count()

    def upload_documents(
        self,
        documents: Iterable[Document],
        *,
        reset_collection: bool = False,
    ) -> None:
        """Upload prepared LangChain Documents to Chroma Cloud."""

        if reset_collection:
            self.reset()

        prepared_documents = prepare_documents_for_chroma(documents)
        if not prepared_documents:
            print("[WARN] No documents to upload.")
            return

        total = len(prepared_documents)
        print(f"[INFO] Uploading {total} documents to Chroma Cloud...")
        for start in range(0, total, self.store_batch_size):
            end = min(start + self.store_batch_size, total)
            batch = prepared_documents[start:end]
            print(f"[INFO] Uploading documents {start + 1}-{end}/{total}...")
            self.collection.upsert(
                ids=[document_id(document, start + index) for index, document in enumerate(batch)],
                documents=[document.page_content for document in batch],
                metadatas=[clean_metadata(document.metadata) for document in batch],
            )

        print(f"[INFO] Chroma Cloud store now contains {self.count()} records.")

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        reset_collection: bool = False,
    ) -> None:
        """LangChain-style alias for upload_documents()."""

        self.upload_documents(documents, reset_collection=reset_collection)

    def search(
        self,
        query_text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        candidate_count: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
        use_hybrid_search: bool | None = None,
        group_by_source: bool | None = None,
        expand_to_parent: bool = True,
    ) -> list[dict[str, Any]]:
        """Search child chunks and optionally return their parent sections."""

        print(f"[INFO] Querying Chroma Cloud for: {query_text!r}")
        child_count = self.record_count(CHILD_FILTER)
        if child_count == 0:
            print("[WARN] Chroma Cloud collection has no child chunks.")
            return []

        limit = min(max(top_k, candidate_count), child_count)
        should_hybrid = self.use_hybrid_search if use_hybrid_search is None else use_hybrid_search
        should_group = self.group_by_source if group_by_source is None else group_by_source

        if should_hybrid:
            try:
                results = self.hybrid_search(query_text, limit, should_group)
            except Exception as exc:
                print(f"[WARN] Hybrid search failed; falling back to dense query: {exc}")
                results = self.dense_search(query_text, limit)
        else:
            results = self.dense_search(query_text, limit)

        if expand_to_parent:
            results = self.expand_results_to_parents(results)

        return results[:top_k]

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        candidate_count: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
        use_hybrid_search: bool | None = None,
        group_by_source: bool | None = None,
        expand_to_parent: bool = True,
    ) -> list[dict[str, Any]]:
        """Backward-compatible alias for search()."""

        return self.search(
            query_text,
            top_k=top_k,
            candidate_count=candidate_count,
            use_hybrid_search=use_hybrid_search,
            group_by_source=group_by_source,
            expand_to_parent=expand_to_parent,
        )

    def similarity_search(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Document, float]]:
        """Return LangChain Documents and scores."""

        return [
            (Document(page_content=result["text"], metadata=result["metadata"]), result["score"])
            for result in self.search(query_text, top_k=top_k)
        ]

    def hybrid_search(
        self,
        query_text: str,
        limit: int,
        group_by_source: bool,
    ) -> list[dict[str, Any]]:
        """Run Chroma Cloud dense + sparse RRF search over child chunks."""

        rank = Rrf(
            ranks=[
                Knn(query=query_text, limit=limit, return_rank=True),
                Knn(query=query_text, key=self.sparse_key, limit=limit, return_rank=True),
            ],
            weights=[self.dense_weight, self.sparse_weight],
            k=60,
        )
        search = Search().where(CHILD_FILTER).rank(rank)
        if group_by_source:
            search = search.group_by(
                GroupBy(
                    keys=K("source_document_id"),
                    aggregate=MinK(keys=K.SCORE, k=self.group_by_document_limit),
                )
            )

        result = self.collection.search(
            search.limit(limit).select(
                K.ID,
                K.DOCUMENT,
                K.SCORE,
                K.METADATA,
                *SEARCH_METADATA_KEYS,
            )
        )
        return format_search_results(result, search_type="hybrid_rrf_child")

    def dense_search(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """Run a dense Chroma query over child chunks."""

        result = self.collection.query(
            query_texts=[query_text],
            n_results=min(limit, self.record_count(CHILD_FILTER)),
            where=CHILD_FILTER,
            include=["documents", "metadatas", "distances"],
        )
        return format_query_results(result, search_type="dense_child")

    def record_count(self, where: dict[str, Any]) -> int:
        """Count records matching a Chroma metadata filter."""

        records = self.collection.get(where=where, include=["metadatas"])
        return len(records.get("ids", []))

    def expand_results_to_parents(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace child hit text with the linked parent section text."""

        parent_ids = [
            result["metadata"].get("parent_id")
            for result in results
            if result["metadata"].get("parent_id")
        ]
        parents = self.parents_by_id(parent_ids)
        expanded: list[dict[str, Any]] = []
        seen_parent_ids: set[str] = set()

        for result in results:
            child_metadata = result["metadata"]
            parent_id = child_metadata.get("parent_id")
            parent = parents.get(parent_id)

            if not parent_id or parent is None:
                expanded.append(result)
                continue
            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)
            metadata = dict(parent["metadata"])
            metadata.update(
                {
                    "retrieved_child_id": result.get("id"),
                    "retrieved_child_score": result.get("score"),
                    "retrieved_child_chunk_index": child_metadata.get("child_chunk_index"),
                    "retrieval_record_type": "child",
                }
            )
            expanded.append(
                {
                    **result,
                    "text": parent["text"],
                    "metadata": metadata,
                    "parent_id": parent_id,
                    "child_text": result["text"],
                    "search_type": f"{result['search_type']}_parent",
                }
            )

        return expanded

    def parents_by_id(self, parent_ids: Iterable[Any]) -> dict[str, dict[str, Any]]:
        """Fetch parent section records and reassemble oversized parent parts."""

        unique_ids = [str(parent_id) for parent_id in dict.fromkeys(parent_ids) if parent_id]
        if not unique_ids:
            return {}

        try:
            records = self.collection.get(
                where={"$and": [PARENT_FILTER, {"parent_id": {"$in": unique_ids}}]},
                include=["documents", "metadatas"],
            )
            return format_parent_records(records)
        except Exception as exc:
            print(f"[WARN] Batched parent fetch failed; trying one by one: {exc}")

        parents: dict[str, dict[str, Any]] = {}
        for parent_id in unique_ids:
            records = self.collection.get(
                where={"$and": [PARENT_FILTER, {"parent_id": parent_id}]},
                include=["documents", "metadatas"],
            )
            parents.update(format_parent_records(records))

        return parents


def collection_schema(
    *,
    dense_embedding_function: ChromaCloudQwenEmbeddingFunction,
    sparse_embedding_function: ChromaCloudSpladeEmbeddingFunction,
    sparse_key: str,
) -> Schema:
    """Build a Chroma Cloud schema with dense and sparse indexes."""

    schema = Schema()
    schema.create_index(
        config=VectorIndexConfig(
            space="cosine",
            source_key=K.DOCUMENT,
            embedding_function=dense_embedding_function,
        )
    )
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=sparse_embedding_function,
        ),
        key=sparse_key,
    )
    return schema


def qwen_embedding_function() -> ChromaCloudQwenEmbeddingFunction:
    """Dense embedding function executed by Chroma Cloud."""

    return ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        task=DEFAULT_CHROMA_QWEN_TASK,
        instructions=QWEN_RETRIEVAL_INSTRUCTIONS,
    )


def splade_embedding_function() -> ChromaCloudSpladeEmbeddingFunction:
    """Sparse embedding function executed by Chroma Cloud."""

    return ChromaCloudSpladeEmbeddingFunction(
        model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1,
    )


def cloud_host_from_env(cloud_host: str | None) -> str:
    host = cloud_host or os.getenv("CHROMA_HOST", DEFAULT_CHROMA_HOST)
    return host.removeprefix("https://").removeprefix("http://").strip().rstrip("/")


def chroma_api_key(api_key: str | None) -> str:
    final_api_key = api_key or os.getenv("CHROMA_API_KEY")
    if not final_api_key or final_api_key.startswith(("your_chroma_cloud_api_key", "<")):
        raise ValueError("Chroma Cloud API key is missing. Set CHROMA_API_KEY in .env.")
    return final_api_key


def load_environment() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Chroma Cloud vector store from PDF documents.",
    )
    parser.add_argument("--data-dir", default="Literatura", help="Directory with PDF files.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--cloud-host", default=None)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--store-batch-size", type=int, default=DEFAULT_CHROMA_STORE_BATCH_SIZE)
    parser.add_argument("--max-characters", type=int, default=DEFAULT_CHUNK_MAX_CHARACTERS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE)
    parser.add_argument("--test-query", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-hybrid-search", action="store_true")
    parser.add_argument("--no-group-by-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()

    pages = load_all_documents(args.data_dir)
    prepared = build_parent_child_documents(
        pages,
        chunk_size=args.max_characters,
        chunk_overlap=args.overlap,
    )
    store = ChromaVectorStore(
        collection_name=args.collection_name,
        cloud_host=args.cloud_host,
        tenant=args.tenant,
        database=args.database,
        store_batch_size=args.store_batch_size,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        reset_collection=args.reset,
    )
    store.upload_documents(prepared.parent_documents + prepared.child_documents)

    print("[INFO] Source pages:", len(pages))
    print("[INFO] Parent sections:", len(prepared.parent_documents))
    print("[INFO] Child chunks:", len(prepared.child_documents))
    print("[INFO] Stored Chroma records:", store.count())

    if prepared.child_documents:
        print("[INFO] Example child:", prepared.child_documents[0].page_content[:500])

    if args.test_query:
        for index, result in enumerate(
            store.search(args.test_query, top_k=3, candidate_count=args.candidate_pool_size),
            start=1,
        ):
            print("=" * 80)
            print(f"Result #{index}")
            print(f"score: {result['score']:.4f}")
            print(f"search_type: {result.get('search_type')}")
            print(f"source: {result['metadata'].get('source')}")
            print(f"section: {result['metadata'].get('section_title')}")
            print(result["text"][:500])


if __name__ == "__main__":
    main()
