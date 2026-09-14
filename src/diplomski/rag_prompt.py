from __future__ import annotations

import json
from typing import Any

from diplomski.retriever import RetrievedDocument
from diplomski.settings import DEFAULT_MAX_CONTEXT_CHARS


RAG_SYSTEM_INSTRUCTION = """
Ti si RAG asistent za pitanja na osnovu dokumenata o lekovima.
Odgovaraj na srpskom jeziku.
Koristi samo informacije iz prosledjenog konteksta.
Ako odgovor nije podrzan kontekstom, jasno reci da nema dovoljno informacija.
Ne izmisljaj doze, indikacije, kontraindikacije ili nezeljena dejstva.
Kada je korisno, navedi izvore na kraju odgovora.
Ovo nije zamena za savet lekara ili farmaceuta.
""".strip()


def build_rag_prompt(
    question: str,
    documents: list[RetrievedDocument],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Build the final RAG prompt from a question and retrieved documents."""

    context = format_context(documents, max_context_chars=max_context_chars)

    return f"""
Kontekst:
{context}

Pitanje:
{question}

Zadatak:
Odgovori direktno i jasno na osnovu konteksta. Ako relevantne informacije nisu
u kontekstu, reci da nemas dovoljno informacija iz dostupnih dokumenata.
""".strip()


def format_context(
    documents: list[RetrievedDocument],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Format retrieved documents into a bounded context block."""

    parts: list[str] = []
    used_chars = 0

    for index, document in enumerate(documents, start=1):
        source_label = format_source_label(document.metadata)
        text = document.text.strip()
        if not text:
            continue

        header_lines = [
            f"[Dokument {index}]",
            f"score: {document.score:.4f}",
            f"source: {source_label}",
        ]
        if document.metadata.get("medicine_name"):
            header_lines.append(f"lek: {document.metadata.get('medicine_name')}")
        if document.metadata.get("active_substance"):
            header_lines.append(
                f"aktivna_supstanca: {document.metadata.get('active_substance')}"
            )
        header_lines.extend(
            [
                f"section: {document.metadata.get('section_title')}",
                f"content_type: {document.metadata.get('content_type')}",
                f"tekst:\n{text}",
            ]
        )
        block = "\n".join(header_lines)

        remaining_chars = max_context_chars - used_chars
        if remaining_chars <= 0:
            break

        if len(block) > remaining_chars:
            block = block[:remaining_chars].rstrip()

        parts.append(block)
        used_chars += len(block)

    return "\n\n".join(parts) if parts else "[Nema pronadjenih dokumenata.]"


def extract_sources(documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
    """Return compact, de-duplicated source information for the final answer."""

    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()

    for document in documents:
        metadata = document.metadata
        source = _metadata_display_value(metadata, "source", "sources")
        file_name = _metadata_display_value(metadata, "file_name", "file_names")
        page = _metadata_display_value(metadata, "page_number", "page_numbers")
        key = (
            json.dumps(source, ensure_ascii=False, default=str),
            json.dumps(file_name, ensure_ascii=False, default=str),
            json.dumps(page, ensure_ascii=False, default=str),
        )

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "source": source,
                "file_name": file_name,
                "medicine_name": metadata.get("medicine_name"),
                "active_substance": metadata.get("active_substance"),
                "page": page,
                "section": metadata.get("section_title"),
                "content_type": metadata.get("content_type"),
                "score": document.score,
            }
        )

    return sources


def format_source_label(metadata: dict[str, Any]) -> str:
    """Create a readable source label from Chroma metadata."""

    file_name = _metadata_display_value(metadata, "file_name", "file_names")
    source = _metadata_display_value(metadata, "source", "sources")
    page = _metadata_display_value(metadata, "page_number", "page_numbers")
    section = metadata.get("section_title")
    medicine_name = metadata.get("medicine_name")
    active_substance = metadata.get("active_substance")

    label = medicine_name or file_name or source or "unknown source"
    if active_substance:
        label = f"{label} ({active_substance})"
    if page is not None:
        label = f"{label}, page {page}"
    if section:
        label = f"{label}, section {section}"

    return str(label)


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
