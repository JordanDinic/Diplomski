from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from pypdf import PdfReader

from diplomski.document_types import DocumentType


LEAFLET_SECTION_TITLES = {
    1: "1. Sta je lek i cemu je namenjen",
    2: "2. Sta treba da znate pre nego sto uzmete lek",
    3: "3. Kako se uzima lek",
    4: "4. Moguca nezeljena dejstva",
    5: "5. Kako cuvati lek",
    6: "6. Sadrzaj pakovanja i ostale informacije",
}

LEAFLET_HEADING_PATTERNS = {
    1: (
        r"\b1\s+sta\s+je\s+lek\s+.{0,120}?\s+i\s+cemu\s+je\s+namenjen\b",
        r"\b1\s+\u0448\u0442\u0430\s+\u0458\u0435\s+\u043b\u0435\u043a\s+.{0,120}?\s+\u0438\s+\u0447\u0435\u043c\u0443\s+\u0458\u0435\s+\u043d\u0430\u043c\u0435\u045a\u0435\u043d\b",
    ),
    2: (
        r"\b2\s+sta\s+treba\s+da\s+znate\s+pre\s+nego\s+sto\s+(?:uzmete|primenite)\s*lek\b",
        r"\b2\s+\u0448\u0442\u0430\s+\u0442\u0440\u0435\u0431\u0430\s+\u0434\u0430\s+\u0437\u043d\u0430\u0442\u0435\s+\u043f\u0440\u0435\s+\u043d\u0435\u0433\u043e\s+\u0448\u0442\u043e\s+(?:\u0443\u0437\u043c\u0435\u0442\u0435|\u043f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u0435)\s*\u043b\u0435\u043a\b",
    ),
    3: (
        r"\b3\s+kako\s+se\s+(?:uzima|primenjuje|upotrebljava)\s*lek\b",
        r"\b3\s+\u043a\u0430\u043a\u043e\s+\u0441\u0435\s+(?:\u0443\u0437\u0438\u043c\u0430|\u043f\u0440\u0438\u043c\u0435\u045a\u0443\u0458\u0435)\s*\u043b\u0435\u043a\b",
    ),
    4: (
        r"\b4\s+moguca\s+nezeljena\s+dejstva\b",
        r"\b4\s+\u043c\u043e\u0433\u0443\u045b\u0430\s+\u043d\u0435\u0436\u0435\u0459\u0435\u043d\u0430\s+\u0434\u0435\u0458\u0441\u0442\u0432\u0430\b",
    ),
    5: (
        r"\b5\s+kako\s+cuvati\s+lek\b",
        r"\b5\s+\u043a\u0430\u043a\u043e\s+\u0447\u0443\u0432\u0430\u0442\u0438\s+\u043b\u0435\u043a\b",
    ),
    6: (
        r"\b6\s+sadrzaj\s+pakovanja\s+i\s+ostale\s+informacije\b",
        r"\b6\s+\u0441\u0430\u0434\u0440\u0436\u0430\u0458\s+\u043f\u0430\u043a\u043e\u0432\u0430\u045a\u0430\s+\u0438\s+\u043e\u0441\u0442\u0430\u043b\u0435\s+\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0458\u0435\b",
    ),
}

INTERACTION_HEADINGS = (
    "INTERAKCIJE LEKOVA",
    "Vrste interakcija",
    "Hemijske interakcije",
    "Farmakodinamske interakcije",
    "Farmakodinamicke interakcije",
    "Farmakokineticke interakcije",
    "Resorpcija",
    "Distribucija",
    "Metabolizam",
    "Ekskrecija",
    "Interakcije lekova i biljnih preparata",
    "Interakcije lekova i hrane",
    "Interakcije lekova sa hranom",
    "Interakcije lekova i vitamina",
    "Literatura",
)


@dataclass(frozen=True)
class TextSpan:
    """A source-aware text span inside a combined PDF text."""

    start: int
    end: int
    page_number: int


@dataclass(frozen=True)
class SectionMarker:
    """A detected section boundary."""

    start: int
    title: str
    path: str
    method: str
    level: int = 0


def build_parent_sections(page_documents: list[Document]) -> list[Document]:
    """Build parent section Documents from page-level PDF Documents."""

    parent_sections: list[Document] = []

    for source_documents in _documents_by_source(page_documents):
        document_type = _document_type(source_documents)

        if document_type == DocumentType.MEDICINE_LEAFLET.value:
            sections = _medicine_leaflet_sections(source_documents)
        elif document_type == DocumentType.PRACTICE_GUIDE.value:
            sections = _practice_guide_sections(source_documents)
        elif document_type == DocumentType.INTERACTION_REFERENCE.value:
            sections = _interaction_reference_sections(source_documents)
        else:
            sections = _page_sections(source_documents, method="page_fallback")

        parent_sections.extend(sections)

    return parent_sections


def _medicine_leaflet_sections(page_documents: list[Document]) -> list[Document]:
    combined_text, spans = _combine_pages(page_documents)
    markers = _leaflet_markers(combined_text)
    if not markers:
        return _page_sections(page_documents, method="leaflet_page_fallback")

    return _sections_from_markers(
        page_documents=page_documents,
        combined_text=combined_text,
        spans=spans,
        markers=markers,
        method="medicine_leaflet_headings",
    )


def _practice_guide_sections(page_documents: list[Document]) -> list[Document]:
    source = _metadata_value(page_documents[0], "source")
    outline_markers = _outline_markers(page_documents, str(source)) if source else []
    if outline_markers:
        combined_text, spans = _combine_pages(page_documents)
        return _sections_from_markers(
            page_documents=page_documents,
            combined_text=combined_text,
            spans=spans,
            markers=outline_markers,
            method="pdf_outline",
        )

    return _page_sections(page_documents, method="practice_guide_page_fallback")


def _interaction_reference_sections(page_documents: list[Document]) -> list[Document]:
    combined_text, spans = _combine_pages(page_documents)
    markers = _line_heading_markers(
        combined_text,
        headings=INTERACTION_HEADINGS,
        method="interaction_heading_rules",
    )
    if not markers:
        return _page_sections(page_documents, method="interaction_page_fallback")

    return _sections_from_markers(
        page_documents=page_documents,
        combined_text=combined_text,
        spans=spans,
        markers=markers,
        method="interaction_heading_rules",
    )


def _page_sections(page_documents: list[Document], *, method: str) -> list[Document]:
    sections: list[Document] = []

    for section_index, document in enumerate(page_documents):
        page_number = _metadata_value(document, "page_number") or _metadata_value(document, "page")
        title = f"Page {page_number}" if page_number is not None else "Page"
        sections.append(
            _section_document(
                source_document=page_documents[0],
                section_index=section_index,
                title=title,
                path=title,
                text=document.page_content,
                page_numbers=[page_number] if page_number is not None else [],
                method=method,
            )
        )

    return sections


def _sections_from_markers(
    *,
    page_documents: list[Document],
    combined_text: str,
    spans: list[TextSpan],
    markers: list[SectionMarker],
    method: str,
) -> list[Document]:
    sorted_markers = sorted(markers, key=lambda marker: marker.start)
    sections: list[Document] = []

    first_marker = sorted_markers[0]
    preamble = combined_text[:first_marker.start].strip()
    if preamble:
        sections.append(
            _section_document(
                source_document=page_documents[0],
                section_index=0,
                title="Document introduction",
                path="Document introduction",
                text=preamble,
                page_numbers=_page_numbers_for_range(spans, 0, first_marker.start),
                method=f"{method}_preamble",
            )
        )

    for marker_index, marker in enumerate(sorted_markers):
        end = (
            sorted_markers[marker_index + 1].start
            if marker_index + 1 < len(sorted_markers)
            else len(combined_text)
        )
        text = combined_text[marker.start:end].strip()
        if not text:
            continue

        sections.append(
            _section_document(
                source_document=page_documents[0],
                section_index=len(sections),
                title=marker.title,
                path=marker.path,
                text=text,
                page_numbers=_page_numbers_for_range(spans, marker.start, end),
                method=marker.method or method,
            )
        )

    return sections


def _section_document(
    *,
    source_document: Document,
    section_index: int,
    title: str,
    path: str,
    text: str,
    page_numbers: list[int],
    method: str,
) -> Document:
    metadata = dict(source_document.metadata)
    section_id = _section_id(metadata, section_index, title, page_numbers)
    metadata.update(
        {
            "record_type": "parent",
            "content_type": "parent_section",
            "section_index": section_index,
            "section_id": section_id,
            "parent_id": section_id,
            "section_title": title,
            "section_path": path,
            "section_detection": method,
            "page_numbers": page_numbers,
            "page_number": page_numbers[0] if page_numbers else metadata.get("page_number"),
            "page": page_numbers[0] if page_numbers else metadata.get("page"),
        }
    )

    return Document(page_content=text.strip(), metadata=metadata)


def _leaflet_markers(text: str) -> list[SectionMarker]:
    indexed = _indexed_text(text)
    matches: list[tuple[int, int, int]] = []

    for number, patterns in LEAFLET_HEADING_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, indexed.normalized):
                start = indexed.raw_index(match.start())
                if start is None:
                    continue
                matches.append((start, match.start(), number))

    selected = _select_last_complete_leaflet_sequence(matches)
    if not selected:
        return []

    return [
        SectionMarker(
            start=start,
            title=LEAFLET_SECTION_TITLES[number],
            path=LEAFLET_SECTION_TITLES[number],
            method="medicine_leaflet_headings",
        )
        for start, number in selected
    ]


def _select_last_complete_leaflet_sequence(
    matches: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    if not matches:
        return []

    ordered = sorted(set(matches), key=lambda item: item[0])
    selected: list[tuple[int, int]] = []
    expected = 6

    for raw_start, _normalized_start, number in reversed(ordered):
        if number != expected:
            continue
        selected.append((raw_start, number))
        expected -= 1
        if expected == 0:
            break

    if expected != 0:
        return []

    return list(reversed(selected))


def _line_heading_markers(
    text: str,
    *,
    headings: Iterable[str],
    method: str,
) -> list[SectionMarker]:
    heading_by_key = {
        _indexed_text(heading, keep_spaces=False).normalized: heading
        for heading in headings
    }
    markers: list[SectionMarker] = []
    offset = 0

    for line in text.splitlines(keepends=True):
        line_text = line.strip()
        line_key = _indexed_text(line_text, keep_spaces=False).normalized

        if _is_standalone_heading_line(line_text, line_key, heading_by_key):
            title = _matched_heading_title(line_key, heading_by_key)
            if title is not None:
                markers.append(
                    SectionMarker(
                        start=offset + max(line.find(line_text), 0),
                        title=title,
                        path=title,
                        method=method,
                    )
                )

        offset += len(line)

    return _deduplicate_markers(markers)


def _is_standalone_heading_line(
    line_text: str,
    line_key: str,
    heading_by_key: dict[str, str],
) -> bool:
    if not line_text or not line_key:
        return False

    first_character = line_text[0]
    if not first_character.isalnum():
        return False

    for heading_key in sorted(heading_by_key, key=len, reverse=True):
        if heading_by_key[heading_key] == "INTERAKCIJE LEKOVA":
            if line_key == heading_key:
                return True
            continue
        if line_key == heading_key:
            return True
        if line_key.startswith(heading_key) and len(line_key) <= len(heading_key) + 8:
            return True

    return False


def _matched_heading_title(
    line_key: str,
    heading_by_key: dict[str, str],
) -> str | None:
    for heading_key in sorted(heading_by_key, key=len, reverse=True):
        title = heading_by_key[heading_key]
        if title == "INTERAKCIJE LEKOVA":
            if line_key == heading_key:
                return title
            continue
        if line_key == heading_key:
            return title
        if line_key.startswith(heading_key) and len(line_key) <= len(heading_key) + 8:
            return title

    return None


def _outline_markers(page_documents: list[Document], source: str) -> list[SectionMarker]:
    path = Path(source)
    if not path.exists():
        return []

    try:
        reader = PdfReader(str(path))
        outline_entries = _flatten_outline(reader.outline, reader)
    except Exception as exc:
        print(f"[WARN] Could not read PDF outline from {path}: {exc}")
        return []

    if not outline_entries:
        return []

    combined_text, _spans = _combine_pages(page_documents)
    page_offsets = _page_offsets(page_documents)
    page_texts = {
        int(document.metadata["page_index"]): document.page_content
        for document in page_documents
        if "page_index" in document.metadata
    }
    markers: list[SectionMarker] = []
    cursor_by_page: dict[int, int] = {}

    for entry in outline_entries:
        page_index = entry["page_index"]
        page_text = page_texts.get(page_index)
        page_start = page_offsets.get(page_index)
        if page_text is None or page_start is None:
            continue

        cursor = cursor_by_page.get(page_index, 0)
        local_position = _find_heading_position(page_text, entry["title"], start_at=cursor)
        if local_position is None:
            continue

        cursor_by_page[page_index] = local_position + 1
        global_position = page_start + local_position
        if global_position >= len(combined_text):
            continue

        markers.append(
            SectionMarker(
                start=global_position,
                title=entry["title"],
                path=" > ".join(entry["path"]),
                method="pdf_outline",
                level=entry["level"],
            )
        )

    return _deduplicate_markers(markers)


def _flatten_outline(outline: list[Any], reader: PdfReader) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def walk(items: list[Any], parent_path: list[str]) -> None:
        last_title: str | None = None

        for item in items:
            if isinstance(item, list):
                walk(item, parent_path + ([last_title] if last_title else []))
                continue

            title = _clean_inline_text(str(getattr(item, "title", "") or ""))
            if not title:
                last_title = None
                continue

            try:
                page_index = reader.get_destination_page_number(item)
            except Exception:
                last_title = title
                continue

            entries.append(
                {
                    "title": title,
                    "path": parent_path + [title],
                    "page_index": page_index,
                    "level": len(parent_path),
                }
            )
            last_title = title

    walk(outline, [])
    return entries


def _deduplicate_markers(markers: list[SectionMarker]) -> list[SectionMarker]:
    unique: list[SectionMarker] = []
    seen: set[tuple[int, str]] = set()

    for marker in sorted(markers, key=lambda item: item.start):
        key = (marker.start, marker.path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(marker)

    return unique


def _find_heading_position(text: str, heading: str, *, start_at: int = 0) -> int | None:
    for keep_spaces in (True, False):
        indexed_text = _indexed_text(text, keep_spaces=keep_spaces)
        normalized_heading = _indexed_text(heading, keep_spaces=keep_spaces).normalized
        if not normalized_heading:
            continue

        candidates = _heading_candidates(normalized_heading)
        for candidate in candidates:
            start = 0
            while True:
                index = indexed_text.normalized.find(candidate, start)
                if index < 0:
                    break

                raw_index = indexed_text.raw_index(index)
                if (
                    raw_index is not None
                    and raw_index >= start_at
                    and _is_match_boundary(indexed_text.normalized, index, len(candidate))
                ):
                    return raw_index
                start = index + 1

    return None


def _is_match_boundary(text: str, start: int, length: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after_index = start + length
    after = text[after_index] if after_index < len(text) else " "

    return not before.isalnum() and not after.isalnum()


def _heading_candidates(normalized_heading: str) -> list[str]:
    candidates = [normalized_heading]

    words = normalized_heading.split()
    if len(words) > 6:
        candidates.append(" ".join(words[: min(len(words), 12)]))
    if len(normalized_heading) > 120:
        candidates.append(normalized_heading[:120].rstrip())

    unique: list[str] = []
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        if candidate in unique:
            continue
        unique.append(candidate)

    return unique


@dataclass(frozen=True)
class IndexedText:
    normalized: str
    raw_indexes: tuple[int, ...]

    def raw_index(self, normalized_index: int) -> int | None:
        if normalized_index < 0 or normalized_index >= len(self.raw_indexes):
            return None
        return self.raw_indexes[normalized_index]


def _indexed_text(text: str, *, keep_spaces: bool = True) -> IndexedText:
    normalized_chars: list[str] = []
    raw_indexes: list[int] = []
    last_was_space = False

    for raw_index, character in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", character)
        normalized_character = "".join(
            char for char in decomposed if not unicodedata.combining(char)
        ).casefold()

        if not normalized_character:
            continue

        for char in normalized_character:
            if char.isalnum():
                normalized_chars.append(char)
                raw_indexes.append(raw_index)
                last_was_space = False
            elif keep_spaces and char.isspace() and normalized_chars and not last_was_space:
                normalized_chars.append(" ")
                raw_indexes.append(raw_index)
                last_was_space = True

    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        raw_indexes.pop()

    return IndexedText("".join(normalized_chars), tuple(raw_indexes))


def _combine_pages(page_documents: list[Document]) -> tuple[str, list[TextSpan]]:
    parts: list[str] = []
    spans: list[TextSpan] = []
    offset = 0

    for document in page_documents:
        text = document.page_content.strip()
        if not text:
            continue

        if parts:
            separator = "\n\n"
            parts.append(separator)
            offset += len(separator)

        start = offset
        parts.append(text)
        offset += len(text)
        page_number = int(document.metadata.get("page_number") or document.metadata.get("page"))
        spans.append(TextSpan(start=start, end=offset, page_number=page_number))

    return "".join(parts), spans


def _page_offsets(page_documents: list[Document]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    offset = 0

    for document in page_documents:
        text = document.page_content.strip()
        if not text:
            continue

        if offsets:
            offset += 2

        page_index = document.metadata.get("page_index")
        if page_index is not None:
            offsets[int(page_index)] = offset
        offset += len(text)

    return offsets


def _page_numbers_for_range(spans: list[TextSpan], start: int, end: int) -> list[int]:
    page_numbers: list[int] = []

    for span in spans:
        if span.end <= start or span.start >= end:
            continue
        if span.page_number not in page_numbers:
            page_numbers.append(span.page_number)

    return page_numbers


def _documents_by_source(documents: list[Document]) -> list[list[Document]]:
    groups: list[list[Document]] = []
    current_group: list[Document] = []
    current_source: Any = object()

    for document in sorted(
        documents,
        key=lambda item: (
            str(item.metadata.get("source", "")),
            int(item.metadata.get("page_index", item.metadata.get("page", 0))),
        ),
    ):
        source = document.metadata.get("source")
        if current_group and source != current_source:
            groups.append(current_group)
            current_group = []

        current_group.append(document)
        current_source = source

    if current_group:
        groups.append(current_group)

    return groups


def _document_type(documents: list[Document]) -> str:
    return str(documents[0].metadata.get("document_type") or DocumentType.UNKNOWN.value)


def _metadata_value(document: Document, name: str) -> Any:
    return document.metadata.get(name)


def _section_id(
    metadata: dict[str, Any],
    section_index: int,
    title: str,
    page_numbers: list[int],
) -> str:
    raw_id = "|".join(
        str(part)
        for part in (
            metadata.get("source_document_id"),
            metadata.get("source"),
            section_index,
            title,
            page_numbers[:3],
        )
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
    return f"section_{digest}"


def _clean_inline_text(text: str) -> str:
    return " ".join(text.split())
