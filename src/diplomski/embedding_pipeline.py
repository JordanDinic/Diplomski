from __future__ import annotations

import copy
from typing import Any

import numpy as np
from langchain_core.documents import Document
from markdownify import markdownify as markdownify_html
from sentence_transformers import SentenceTransformer
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Footer, Header, Image, PageBreak, Table

from diplomski.data_loader import load_all_elements
from diplomski.settings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
)


NON_CONTEXT_CATEGORIES = {
    "Header",
    "Footer",
    "PageBreak",
    "Image",
    "Picture",
    "Figure",
    "FigureCaption",
    "Table",
    "TableChunk",
}
TABLE_CATEGORIES = {"Table", "TableChunk"}
VERBAL_CONTEXT_CATEGORIES = {
    "NarrativeText",
    "Title",
    "ListItem",
    "Text",
    "UncategorizedText",
}


class EmbeddingPipeline:
    """
    Prepare Unstructured PDF elements for RAG embeddings.

    Flow:
    Unstructured elements -> table context enrichment -> chunk_by_title
    -> LangChain Documents -> SentenceTransformer embeddings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_characters: int = 1000,
        new_after_n_chars: int = 800,
        overlap: int = 100,
        table_context_elements: int = 2,
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        device: str = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.overlap = overlap
        self.table_context_elements = table_context_elements
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = _resolve_device(device)
        self._model: SentenceTransformer | None = None

    def prepare_elements(self, elements: list[Any]) -> list[Any]:
        """
        Add previous valid textual context directly into every table element.

        Header, Footer, PageBreak, Image, Figure and table-like elements are
        skipped while collecting context. Calling this method multiple times
        will not duplicate table context.
        """

        prepared: list[Any] = []
        enriched_tables = 0

        for index, element in enumerate(elements):
            if not _is_table(element):
                prepared.append(element)
                continue

            if _has_table_context(element):
                prepared.append(element)
                continue

            context_elements = self._previous_relevant_elements(elements, index)
            prepared.append(_table_with_context(element, context_elements))
            enriched_tables += 1

        print(
            f"[INFO] Prepared {len(prepared)} elements "
            f"and enriched {enriched_tables} tables."
        )
        return prepared

    def chunk_elements(self, elements: list[Any]) -> list[Any]:
        """
        Chunk all prepared elements with Unstructured's chunk_by_title.

        Tables are not split because skip_table_chunking=True is used.
        isolate_table=True keeps each table as its own chunk.
        """

        prepared_elements = self.prepare_elements(elements)
        chunks = chunk_by_title(
            prepared_elements,
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            overlap=self.overlap,
            include_orig_elements=True,
            skip_table_chunking=True,
            isolate_table=True,
        )

        print(
            f"[INFO] Chunked {len(prepared_elements)} prepared elements "
            f"into {len(chunks)} chunks."
        )
        return chunks

    def chunks_to_documents(self, chunks: list[Any]) -> list[Document]:
        """Convert Unstructured chunks into LangChain Document objects."""

        documents: list[Document] = []

        for chunk_index, chunk in enumerate(chunks):
            text = str(chunk).strip()
            if not text:
                continue

            metadata = _metadata_dict(chunk)
            category = _category(chunk)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "category": category,
                    "content_type": "table" if _is_table(chunk) else "text",
                    "element_id": getattr(chunk, "id", None),
                }
            )

            documents.append(Document(page_content=text, metadata=metadata))

        print(f"[INFO] Converted {len(chunks)} chunks into {len(documents)} documents.")
        return documents

    def documents_from_elements(self, elements: list[Any]) -> list[Document]:
        """Run preparation, chunking and LangChain Document conversion."""

        chunks = self.chunk_elements(elements)
        return self.chunks_to_documents(chunks)

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """Create embeddings for LangChain Document contents."""

        texts = [document.page_content for document in documents]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create embeddings for raw text strings."""

        if not texts:
            print("[WARN] No texts to embed.")
            return np.empty((0, 0), dtype=np.float32)

        print(f"[INFO] Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        print(f"[INFO] Embeddings shape: {embeddings_array.shape}")

        return embeddings_array

    @property
    def model(self) -> SentenceTransformer:
        """Load the embedding model lazily."""

        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._model.max_seq_length = self.max_seq_length
            print(f"[INFO] Loaded embedding model: {self.model_name}")
            print(f"[INFO] device: {self.device}")
            print(f"[INFO] max_seq_length: {self._model.max_seq_length}")
            print(f"[INFO] batch_size: {self.batch_size}")

        return self._model

    def _previous_relevant_elements(self, elements: list[Any], current_index: int) -> list[Any]:
        current_element = elements[current_index]
        relevant: list[Any] = []

        for candidate in reversed(elements[:current_index]):
            if not _same_source(candidate, current_element):
                continue
            if not _is_context_candidate(candidate):
                continue

            relevant.append(candidate)
            if len(relevant) == self.table_context_elements:
                break

        return list(reversed(relevant))


def _table_with_context(table: Any, context_elements: list[Any]) -> Table:
    table_text = _table_text(table)
    context_text = "\n\n".join(str(element).strip() for element in context_elements)

    if context_text:
        enriched_text = f"Context:\n{context_text}\n\nTable:\n{table_text}"
    else:
        enriched_text = f"Table:\n{table_text}"

    metadata = copy.deepcopy(getattr(table, "metadata", None))
    if metadata is not None:
        text_as_html = getattr(metadata, "text_as_html", None)
        if text_as_html:
            metadata.text_as_html = text_as_html

        metadata.content_type = "table"
        metadata.table_context_prepared = True
        metadata.table_content_format = "markdown" if text_as_html else "text"
        metadata.table_context_element_ids = [
            getattr(element, "id", None) for element in context_elements
        ]
        metadata.table_context_element_indexes = [
            _metadata_value(element, "element_index") for element in context_elements
        ]
        metadata.table_context_categories = [_category(element) for element in context_elements]

    return Table(
        text=enriched_text,
        element_id=getattr(table, "id", None),
        metadata=metadata,
        embeddings=getattr(table, "embeddings", None),
    )


def _table_text(table: Any) -> str:
    metadata = getattr(table, "metadata", None)
    text_as_html = getattr(metadata, "text_as_html", None)

    if text_as_html:
        return markdownify_html(text_as_html).strip()

    return str(table).strip()


def _has_table_context(table: Any) -> bool:
    metadata = getattr(table, "metadata", None)
    if getattr(metadata, "table_context_prepared", False):
        return True

    text = str(table).lstrip()
    return text.startswith("Context:\n") or text.startswith("Table:\n")


def _is_context_candidate(element: Any) -> bool:
    text = str(element).strip()
    if not text:
        return False
    if isinstance(element, (Header, Footer, Image, PageBreak)):
        return False

    category = _category(element)
    if category in NON_CONTEXT_CATEGORIES:
        return False

    return category in VERBAL_CONTEXT_CATEGORIES


def _is_table(element: Any) -> bool:
    return isinstance(element, Table) or _category(element) in TABLE_CATEGORIES


def _same_source(left: Any, right: Any) -> bool:
    left_source = _source_key(left)
    right_source = _source_key(right)

    if left_source is None and right_source is None:
        return True
    if left_source is None or right_source is None:
        return False

    return left_source == right_source


def _source_key(element: Any) -> Any:
    return (
        _metadata_value(element, "source")
        or _metadata_value(element, "file_name")
        or _metadata_value(element, "filename")
    )


def _category(element: Any) -> str:
    return str(getattr(element, "category", type(element).__name__))


def _metadata_value(element: Any, name: str) -> Any:
    metadata = getattr(element, "metadata", None)
    return getattr(metadata, name, None) if metadata is not None else None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _metadata_dict(element: Any) -> dict[str, Any]:
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return {}

    data = metadata.to_dict() if hasattr(metadata, "to_dict") else {}
    data.update(_custom_metadata_fields(metadata))

    orig_elements = getattr(metadata, "orig_elements", None)
    if orig_elements:
        data.update(_trace_metadata_from_orig_elements(orig_elements, data))
        data["orig_elements"] = [_element_summary(orig_element) for orig_element in orig_elements]
    elif "orig_elements" in data:
        data["orig_elements_serialized"] = data.pop("orig_elements")

    return data


def _trace_metadata_from_orig_elements(
    orig_elements: list[Any],
    existing_metadata: dict[str, Any],
) -> dict[str, Any]:
    trace_metadata: dict[str, Any] = {}

    for field, plural_field in (
        ("source", "sources"),
        ("folder", "folders"),
        ("file_name", "file_names"),
        ("file_type", "file_types"),
    ):
        values = _unique_metadata_values(orig_elements, field)
        if not values:
            continue

        if existing_metadata.get(field) is None:
            trace_metadata[field] = values[0]

        if len(values) > 1:
            trace_metadata[plural_field] = values

    page_numbers = _unique_metadata_values(orig_elements, "page_number")
    if page_numbers:
        if existing_metadata.get("page_number") is None:
            trace_metadata["page_number"] = page_numbers[0]
        trace_metadata["page_numbers"] = page_numbers

    element_indexes = _unique_metadata_values(orig_elements, "element_index")
    if element_indexes:
        trace_metadata["element_indexes"] = element_indexes

    return trace_metadata


def _unique_metadata_values(elements: list[Any], field: str) -> list[Any]:
    values: list[Any] = []

    for element in elements:
        value = _metadata_value(element, field)
        if value is None or value == "":
            continue
        if value in values:
            continue

        values.append(value)

    return values


def _custom_metadata_fields(metadata: Any) -> dict[str, Any]:
    fields = (
        "source",
        "folder",
        "file_name",
        "file_type",
        "element_index",
        "content_type",
        "table_context_element_ids",
        "table_context_element_indexes",
        "table_context_categories",
        "table_context_prepared",
        "table_content_format",
    )

    return {
        field: getattr(metadata, field)
        for field in fields
        if getattr(metadata, field, None) is not None
    }


def _element_summary(element: Any) -> dict[str, Any]:
    return {
        "element_id": getattr(element, "id", None),
        "category": _category(element),
        "source": _metadata_value(element, "source"),
        "file_name": _metadata_value(element, "file_name"),
        "page_number": _metadata_value(element, "page_number"),
        "element_index": _metadata_value(element, "element_index"),
        "text_preview": str(element).strip()[:200],
    }


if __name__ == "__main__":
    elements = load_all_elements("Literatura/Lekovi")

    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_elements(elements)
    documents = pipeline.chunks_to_documents(chunks)
    embeddings = pipeline.embed_documents(documents)

    print("[INFO] Number of elements:", len(elements))
    print("[INFO] Number of chunks:", len(chunks))
    print("[INFO] Number of documents:", len(documents))
    print("[INFO] Embeddings shape:", embeddings.shape)
    print("[INFO] Example document:", documents[0].page_content[:500] if documents else None)
