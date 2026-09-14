from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from langchain_core.documents import Document

from diplomski.settings import DEFAULT_CHROMA_DOCUMENT_MAX_BYTES


METADATA_VALUE_MAX_BYTES = 7_500

SEARCH_METADATA_KEYS = (
    "record_type",
    "source",
    "source_document_id",
    "folder",
    "file_name",
    "file_type",
    "document_type",
    "medicine_name",
    "active_substance",
    "content_type",
    "page",
    "page_number",
    "page_numbers",
    "page_index",
    "total_pages",
    "section_id",
    "section_index",
    "section_title",
    "section_path",
    "section_detection",
    "parent_id",
    "parent_section_index",
    "child_chunk_index",
    "child_chunk_count",
    "original_chunk_index",
    "cloud_chunk_index",
    "cloud_subchunk_index",
)

CHROMA_METADATA_KEYS = SEARCH_METADATA_KEYS + (
    "document_bytes",
    "split_from_oversized_document",
    "split_part_count",
)


def prepare_documents_for_chroma(
    documents: Iterable[Document],
    *,
    max_bytes: int = DEFAULT_CHROMA_DOCUMENT_MAX_BYTES,
) -> list[Document]:
    """Split oversized documents and add upload metadata."""

    prepared: list[Document] = []
    for document_index, document in enumerate(documents):
        parts = split_text_by_byte_limit(document.page_content, max_bytes=max_bytes)
        source_id = source_document_id(document)

        for part_index, part in enumerate(parts):
            metadata = dict(document.metadata)
            metadata.setdefault("record_type", "child")
            metadata.setdefault("content_type", "child_chunk")
            metadata.setdefault("source_document_id", source_id)
            metadata.setdefault(
                "original_chunk_index",
                metadata.get("child_chunk_index", metadata.get("chunk_index", document_index)),
            )
            metadata["cloud_chunk_index"] = len(prepared)
            metadata["cloud_subchunk_index"] = part_index
            metadata["document_bytes"] = len(part.encode("utf-8"))

            if len(parts) > 1:
                metadata["split_from_oversized_document"] = True
                metadata["split_part_count"] = len(parts)

            prepared.append(Document(page_content=part, metadata=metadata))

    return prepared


def split_text_by_byte_limit(text: str, *, max_bytes: int) -> list[str]:
    """Split text into UTF-8 byte-safe parts no larger than max_bytes."""

    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text.encode("utf-8")) <= max_bytes:
        return [clean_text]

    parts: list[str] = []
    current = ""

    for line in clean_text.splitlines(keepends=True):
        if len(line.encode("utf-8")) > max_bytes:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(split_long_line_by_byte_limit(line, max_bytes=max_bytes))
            continue

        candidate = current + line
        if current and len(candidate.encode("utf-8")) > max_bytes:
            parts.append(current.strip())
            current = line
        else:
            current = candidate

    if current:
        parts.append(current.strip())

    return [part for part in parts if part]


def split_long_line_by_byte_limit(text: str, *, max_bytes: int) -> list[str]:
    """Split one long line without cutting through UTF-8 characters."""

    parts: list[str] = []
    current_chars: list[str] = []
    current_size = 0

    for character in text:
        character_size = len(character.encode("utf-8"))
        if current_chars and current_size + character_size > max_bytes:
            parts.append("".join(current_chars).strip())
            current_chars = []
            current_size = 0

        current_chars.append(character)
        current_size += character_size

    if current_chars:
        parts.append("".join(current_chars).strip())

    return [part for part in parts if part]


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only Chroma-compatible trace metadata."""

    cleaned: dict[str, Any] = {}
    for key in CHROMA_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            cleaned[key] = safe_metadata_value(value)
    return cleaned


def safe_metadata_value(value: Any) -> str | int | float | bool | list[Any]:
    """Convert unsupported metadata values to bounded strings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return trim_metadata_string(value)
    if is_scalar_list(value):
        return trim_metadata_list(value)
    return trim_metadata_string(json.dumps(value, ensure_ascii=False, default=str))


def trim_metadata_string(value: str) -> str:
    """Limit one metadata string value by UTF-8 byte length."""

    encoded = value.encode("utf-8")
    if len(encoded) <= METADATA_VALUE_MAX_BYTES:
        return value

    marker = "...[truncated]"
    trimmed = encoded[: METADATA_VALUE_MAX_BYTES - len(marker.encode("utf-8"))]
    while trimmed:
        try:
            return trimmed.decode("utf-8") + marker
        except UnicodeDecodeError:
            trimmed = trimmed[:-1]
    return marker


def trim_metadata_list(value: list[Any]) -> list[Any] | str:
    """Keep scalar lists when small, otherwise shorten them safely."""

    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized.encode("utf-8")) <= METADATA_VALUE_MAX_BYTES:
        return value

    trimmed: list[Any] = []
    for item in value:
        candidate = trimmed + [item]
        if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > METADATA_VALUE_MAX_BYTES:
            break
        trimmed = candidate

    return trimmed if trimmed else trim_metadata_string(serialized)


def is_scalar_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, (str, int, float, bool)) for item in value)
    )


def document_id(document: Document, index: int) -> str:
    """Build a stable Chroma id from source, parent and chunk metadata."""

    metadata = document.metadata
    record_type = str(metadata.get("record_type", "child"))
    raw_id = "|".join(
        str(part)
        for part in (
            record_type,
            metadata.get("source_document_id"),
            metadata.get("source"),
            metadata.get("parent_id"),
            metadata.get("section_id"),
            metadata.get("child_chunk_index"),
            metadata.get("cloud_subchunk_index"),
            document.page_content[:200],
            index,
        )
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:24]
    return f"{record_type}_{digest}"


def source_document_id(document: Document) -> str:
    existing_id = document.metadata.get("source_document_id")
    if existing_id:
        return str(existing_id)

    source = document.metadata.get("source") or document.metadata.get("file_name")
    digest = hashlib.sha1(str(source or document.page_content[:500]).encode("utf-8")).hexdigest()
    return f"source_{digest[:20]}"


def format_search_results(results: Any, *, search_type: str) -> list[dict[str, Any]]:
    rows_by_query = results.rows() if hasattr(results, "rows") else []
    rows = rows_by_query[0] if rows_by_query else []
    formatted: list[dict[str, Any]] = []

    for row in rows:
        chroma_score = float(row_value(row, "score", "#score") or 0.0)
        formatted.append(
            {
                "id": row_value(row, "id", "#id"),
                "score": -chroma_score,
                "chroma_score": chroma_score,
                "distance": None,
                "text": row_value(row, "document", "#document") or "",
                "metadata": row_metadata(row),
                "search_type": search_type,
            }
        )

    return formatted


def format_query_results(results: dict[str, Any], *, search_type: str) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    rows = zip(
        results.get("ids", [[]])[0],
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    )

    for item_id, text, metadata, distance in rows:
        distance_value = float(distance)
        formatted.append(
            {
                "id": item_id,
                "score": 1.0 - distance_value,
                "chroma_score": distance_value,
                "distance": distance_value,
                "text": text or "",
                "metadata": metadata or {},
                "search_type": search_type,
            }
        )

    return formatted


def format_parent_records(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parts_by_parent: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}

    for text, metadata in zip(records.get("documents", []), records.get("metadatas", [])):
        metadata = metadata or {}
        parent_id = metadata.get("parent_id")
        if parent_id:
            part_index = int_or_default(metadata.get("cloud_subchunk_index"), 0)
            parts_by_parent.setdefault(str(parent_id), []).append((part_index, text or "", metadata))

    parents: dict[str, dict[str, Any]] = {}
    for parent_id, parts in parts_by_parent.items():
        ordered = sorted(parts, key=lambda item: item[0])
        parents[parent_id] = {
            "text": "\n".join(part[1] for part in ordered if part[1]).strip(),
            "metadata": dict(ordered[0][2]),
        }

    return parents


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row_value(row, "metadata", "#metadata") or {}
    metadata = dict(metadata) if isinstance(metadata, dict) else {}

    for key in SEARCH_METADATA_KEYS:
        value = row_value(row, key)
        if value is not None:
            metadata[key] = value

    return metadata


def row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
