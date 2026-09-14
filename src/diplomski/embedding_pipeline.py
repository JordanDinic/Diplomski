from __future__ import annotations

from langchain_core.documents import Document

from diplomski.chunking import ParentChildDocuments, build_parent_child_documents
from diplomski.data_loader import load_all_documents
from diplomski.settings import DEFAULT_CHUNK_MAX_CHARACTERS, DEFAULT_CHUNK_OVERLAP


class EmbeddingPipeline:
    """
    Prepare LangChain Documents for ChromaDB ingestion.

    The name is kept for compatibility with the rest of the project, but this
    class no longer computes embeddings locally. ChromaDB/Chroma Cloud handles
    embeddings when documents are inserted into the vector store.
    """

    def __init__(
        self,
        max_characters: int = DEFAULT_CHUNK_MAX_CHARACTERS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        **_legacy_options: object,
    ) -> None:
        self.max_characters = max_characters
        self.overlap = overlap

    def prepare_documents(self, page_documents: list[Document]) -> ParentChildDocuments:
        """Create parent sections and child chunks from page-level Documents."""

        prepared = build_parent_child_documents(
            page_documents,
            chunk_size=self.max_characters,
            chunk_overlap=self.overlap,
        )
        print(
            "[INFO] Prepared "
            f"{len(prepared.parent_documents)} parent sections and "
            f"{len(prepared.child_documents)} child chunks."
        )
        return prepared

    def documents_from_pages(self, page_documents: list[Document]) -> list[Document]:
        """Return child chunks for code that only needs searchable documents."""

        return self.prepare_documents(page_documents).child_documents

    def documents_from_elements(self, page_documents: list[Document]) -> list[Document]:
        """Backward-compatible alias for the older Unstructured-based pipeline."""

        return self.documents_from_pages(page_documents)

    def chunk_documents(self, page_documents: list[Document]) -> list[Document]:
        """Backward-compatible alias for older examples."""

        return self.documents_from_pages(page_documents)


if __name__ == "__main__":
    pages = load_all_documents("Literatura")
    pipeline = EmbeddingPipeline()
    prepared_documents = pipeline.prepare_documents(pages)

    print("[INFO] Number of pages:", len(pages))
    print("[INFO] Parent sections:", len(prepared_documents.parent_documents))
    print("[INFO] Child chunks:", len(prepared_documents.child_documents))
    print(
        "[INFO] Example child:",
        prepared_documents.child_documents[0].page_content[:500]
        if prepared_documents.child_documents
        else None,
    )
