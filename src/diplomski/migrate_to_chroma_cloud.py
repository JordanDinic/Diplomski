from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from diplomski.chunking import build_parent_child_documents
from diplomski.console import configure_console_output
from diplomski.data_loader import load_all_documents
from diplomski.settings import (
    DEFAULT_CHROMA_STORE_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
)
from diplomski.vector_store import ChromaVectorStore


def migrate_pdf_folder_to_cloud(
    data_dir: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    cloud_host: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    reset_collection: bool = False,
    store_batch_size: int = DEFAULT_CHROMA_STORE_BATCH_SIZE,
) -> list[Document]:
    """Parse PDF files, prepare parent-child chunks and upload them to Chroma Cloud."""

    page_documents = load_all_documents(data_dir)
    prepared = build_parent_child_documents(page_documents)
    store = _cloud_store(
        collection_name=collection_name,
        cloud_host=cloud_host,
        tenant=tenant,
        database=database,
        store_batch_size=store_batch_size,
        reset_collection=reset_collection,
    )
    store.upload_documents(prepared.all_documents)

    print(
        "[INFO] Migrated "
        f"{len(prepared.parent_documents)} parent sections and "
        f"{len(prepared.child_documents)} child chunks from PDFs."
    )
    return prepared.child_documents


def migrate_local_chroma_to_cloud(
    *,
    local_persist_dir: str | Path = "chroma_db",
    local_collection_name: str = DEFAULT_COLLECTION_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    cloud_host: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    reset_collection: bool = False,
    store_batch_size: int = DEFAULT_CHROMA_STORE_BATCH_SIZE,
) -> list[Document]:
    """
    Copy documents from a local Chroma collection into Chroma Cloud.

    Existing local embeddings are not reused. Chroma Cloud regenerates dense
    Qwen and sparse Splade embeddings from the copied document text.
    """

    documents = _load_local_chroma_documents(
        persist_dir=local_persist_dir,
        collection_name=local_collection_name,
    )
    store = _cloud_store(
        collection_name=collection_name,
        cloud_host=cloud_host,
        tenant=tenant,
        database=database,
        store_batch_size=store_batch_size,
        reset_collection=reset_collection,
    )
    store.upload_documents(documents)

    print(f"[INFO] Migrated {len(documents)} documents from local Chroma.")
    return documents


def _load_local_chroma_documents(
    *,
    persist_dir: str | Path,
    collection_name: str,
) -> list[Document]:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    records = collection.get(include=["documents", "metadatas"])

    ids = records.get("ids", [])
    texts = records.get("documents", [])
    metadatas = records.get("metadatas", [])
    documents: list[Document] = []

    for index, text in enumerate(texts):
        clean_text = (text or "").strip()
        if not clean_text:
            continue

        metadata = _metadata_at(metadatas, index)
        metadata["local_chroma_id"] = _value_at(ids, index)
        metadata.setdefault("migration_source", "local_chroma")

        documents.append(Document(page_content=clean_text, metadata=metadata))

    return documents


def _cloud_store(
    *,
    collection_name: str,
    cloud_host: str | None,
    tenant: str | None,
    database: str | None,
    store_batch_size: int,
    reset_collection: bool,
) -> ChromaVectorStore:
    return ChromaVectorStore(
        collection_name=collection_name,
        cloud_host=cloud_host,
        tenant=tenant,
        database=database,
        store_batch_size=store_batch_size,
        reset_collection=reset_collection,
    )


def _metadata_at(metadatas: list[Any], index: int) -> dict[str, Any]:
    if index >= len(metadatas) or metadatas[index] is None:
        return {}

    return dict(metadatas[index])


def _value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate local PDF or Chroma data into Chroma Cloud.",
    )
    parser.add_argument(
        "--source",
        choices=["pdfs", "local-chroma"],
        default="pdfs",
        help="Migration source.",
    )
    parser.add_argument(
        "--data-dir",
        default="Literatura/Lekovi",
        help="PDF directory used when --source pdfs.",
    )
    parser.add_argument(
        "--local-persist-dir",
        default="chroma_db",
        help="Local Chroma directory used when --source local-chroma.",
    )
    parser.add_argument(
        "--local-collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Local Chroma collection used when --source local-chroma.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Target Chroma Cloud collection.",
    )
    parser.add_argument(
        "--cloud-host",
        default=None,
        help="Chroma Cloud host. Defaults to CHROMA_HOST or project settings.",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Chroma Cloud tenant. Defaults to CHROMA_TENANT or project settings.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Chroma Cloud database. Defaults to CHROMA_DATABASE or project settings.",
    )
    parser.add_argument(
        "--store-batch-size",
        type=int,
        default=DEFAULT_CHROMA_STORE_BATCH_SIZE,
        help="Number of documents to upload per batch.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the target Cloud collection before uploading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_console_output()

    args = _parse_args()

    if args.source == "local-chroma":
        migrate_local_chroma_to_cloud(
            local_persist_dir=args.local_persist_dir,
            local_collection_name=args.local_collection_name,
            collection_name=args.collection_name,
            cloud_host=args.cloud_host,
            tenant=args.tenant,
            database=args.database,
            reset_collection=args.reset,
            store_batch_size=args.store_batch_size,
        )
    else:
        migrate_pdf_folder_to_cloud(
            data_dir=args.data_dir,
            collection_name=args.collection_name,
            cloud_host=args.cloud_host,
            tenant=args.tenant,
            database=args.database,
            reset_collection=args.reset,
            store_batch_size=args.store_batch_size,
        )
