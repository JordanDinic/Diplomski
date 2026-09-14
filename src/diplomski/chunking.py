from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from diplomski.document_sections import build_parent_sections
from diplomski.settings import DEFAULT_CHUNK_MAX_CHARACTERS, DEFAULT_CHUNK_OVERLAP


@dataclass(frozen=True)
class ParentChildDocuments:
    """Parent sections and child chunks prepared for retrieval."""

    parent_documents: list[Document]
    child_documents: list[Document]

    @property
    def all_documents(self) -> list[Document]:
        return self.parent_documents + self.child_documents


def build_parent_child_documents(
    page_documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_MAX_CHARACTERS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ParentChildDocuments:
    """Create parent sections and child chunks from page-level PDF Documents."""

    parent_sections = build_parent_sections(page_documents)
    parent_documents = [_with_retrieval_context(parent) for parent in parent_sections]
    child_documents = chunk_parent_sections(
        parent_sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return ParentChildDocuments(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )


def chunk_parent_sections(
    parent_sections: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_MAX_CHARACTERS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split each parent section into child chunks without crossing sections."""

    child_documents: list[Document] = []

    for parent in parent_sections:
        section_chunks = split_text(parent.page_content, chunk_size, chunk_overlap)

        for child_index, chunk_text in enumerate(section_chunks):
            metadata = _child_metadata(parent.metadata, child_index, len(section_chunks))
            child_documents.append(
                Document(
                    page_content=_text_with_retrieval_context(chunk_text, metadata),
                    metadata=metadata,
                )
            )

    return child_documents


def split_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    """Split text into readable chunks using paragraphs, lines, then words."""

    clean_text = text.strip()
    if not clean_text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if len(clean_text) <= chunk_size:
        return [clean_text]

    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    units = _split_units(clean_text, chunk_size)
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = _join_text(current, unit)
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = _join_text(_overlap_tail(current, overlap), unit)
        else:
            chunks.append(unit.strip())
            current = ""

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = _join_text(_overlap_tail(current[:chunk_size], overlap), current[chunk_size:])

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _split_units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []

    for paragraph in re_split_keep_separator(text, "\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue

        units.extend(_split_long_paragraph(paragraph, chunk_size))

    return units


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    line_units = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(line_units) > 1:
        units: list[str] = []
        for line in line_units:
            if len(line) <= chunk_size:
                units.append(line)
            else:
                units.extend(_split_words(line, chunk_size))
        return units

    return _split_words(paragraph, chunk_size)


def _split_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    units: list[str] = []
    current = ""

    for word in words:
        candidate = _join_text(current, word, separator=" ")
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            units.append(current)

        if len(word) > chunk_size:
            units.extend(word[start:start + chunk_size] for start in range(0, len(word), chunk_size))
            current = ""
        else:
            current = word

    if current:
        units.append(current)

    return units


def re_split_keep_separator(text: str, separator: str) -> list[str]:
    """Split text while keeping paragraph grouping easy to read."""

    if separator not in text:
        return [text]

    return text.split(separator)


def _child_metadata(
    parent_metadata: dict[str, Any],
    child_index: int,
    child_count: int,
) -> dict[str, Any]:
    metadata = dict(parent_metadata)
    metadata.update(
        {
            "record_type": "child",
            "content_type": "child_chunk",
            "parent_id": parent_metadata["parent_id"],
            "section_id": parent_metadata["section_id"],
            "parent_section_index": parent_metadata.get("section_index"),
            "child_chunk_index": child_index,
            "child_chunk_count": child_count,
        }
    )

    return metadata


def _with_retrieval_context(document: Document) -> Document:
    metadata = dict(document.metadata)
    metadata["record_type"] = "parent"
    metadata["content_type"] = "parent_section"
    return Document(
        page_content=_text_with_retrieval_context(document.page_content, metadata),
        metadata=metadata,
    )


def _text_with_retrieval_context(text: str, metadata: dict[str, Any]) -> str:
    if _has_retrieval_context(text):
        return text

    header_lines: list[str] = []
    medicine_name = _metadata_text(metadata, "medicine_name")
    active_substance = _metadata_text(metadata, "active_substance")
    document_name = _document_display_name(metadata)
    section = metadata.get("section_path") or metadata.get("section_title")

    if medicine_name:
        header_lines.append(f"Lek: {medicine_name}")
    if active_substance:
        header_lines.append(f"Aktivna supstanca: {active_substance}")
    if document_name:
        header_lines.append(f"Document: {document_name}")
    if section:
        header_lines.append(f"Section: {section}")

    if not header_lines:
        return text.strip()

    return "\n".join(header_lines) + "\n\n" + text.strip()


def _has_retrieval_context(text: str) -> bool:
    clean_text = text.lstrip()
    return clean_text.startswith(("Lek:", "Document:"))


def _document_display_name(metadata: dict[str, Any]) -> str | None:
    file_name = metadata.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return file_name.strip()

    source = metadata.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]

    return None


def _metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _join_text(left: str, right: str, *, separator: str = "\n\n") -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()

    return f"{left.strip()}{separator}{right.strip()}"


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""

    tail = text[-overlap:]
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]

    return tail.strip()
