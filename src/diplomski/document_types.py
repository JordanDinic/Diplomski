from __future__ import annotations

import unicodedata
from enum import Enum
from pathlib import Path


CYRILLIC_LEAFLET_MARKER = "\u0443\u043f\u0443\u0442\u0441\u0442\u0432\u043e \u0437\u0430 \u043b\u0435\u043a"


class DocumentType(str, Enum):
    """Known PDF document families in the pharmacy RAG corpus."""

    MEDICINE_LEAFLET = "medicine_leaflet"
    PRACTICE_GUIDE = "practice_guide"
    INTERACTION_REFERENCE = "interaction_reference"
    UNKNOWN = "unknown"


def detect_document_type(file_path: str | Path, text_sample: str = "") -> DocumentType:
    """Infer the document type from its path and a small text sample."""

    path = Path(file_path)
    normalized_path = _normalize_text(" ".join(path.parts))
    normalized_text = _normalize_text(text_sample)
    combined = f"{normalized_path}\n{normalized_text}"

    if "vodic" in normalized_path or "dobre apotekarske prakse" in combined:
        return DocumentType.PRACTICE_GUIDE

    if "interakcije lekova" in combined or "interakcije-lekova" in normalized_path:
        return DocumentType.INTERACTION_REFERENCE

    if "lekovi" in normalized_path:
        return DocumentType.MEDICINE_LEAFLET

    if "uputstvo za lek" in combined or CYRILLIC_LEAFLET_MARKER in combined:
        return DocumentType.MEDICINE_LEAFLET

    return DocumentType.UNKNOWN


def _normalize_text(text: str) -> str:
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().replace("_", " ").split())
