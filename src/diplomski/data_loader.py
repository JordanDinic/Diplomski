from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

from diplomski.console import configure_console_output
from diplomski.document_types import DocumentType, detect_document_type


PDF_EXTENSION = ".pdf"
DEFAULT_DATA_DIR = "Literatura"


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load one PDF and return one LangChain Document per non-empty page.

    This function is intentionally only a loading/extraction step. It does not
    split sections, create embeddings, write to ChromaDB, or call an LLM.
    """

    path = _validate_pdf_path(file_path)
    reader = _open_pdf(path)
    document_type = detect_document_type(path, _text_sample(reader))
    source_document_id = _source_document_id(path)
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = _extract_page_text(page)
        if not text:
            continue

        page_number = page_index + 1
        documents.append(
            Document(
                page_content=text,
                metadata=_page_metadata(
                    path=path,
                    page_number=page_number,
                    page_index=page_index,
                    total_pages=len(reader.pages),
                    document_type=document_type,
                    source_document_id=source_document_id,
                ),
            )
        )

    return documents


def load_all_documents(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Document]:
    """
    Load all PDFs from a directory tree as page-level LangChain Documents.

    Unreadable PDFs are skipped with a warning so one bad file does not stop
    ingestion for the whole corpus.
    """

    data_path = _validate_directory_path(data_dir)
    documents: list[Document] = []

    for pdf_path in sorted(data_path.rglob(f"*{PDF_EXTENSION}")):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:
            print(f"[WARN] Skipping {pdf_path}: {exc}")

    return documents


def _validate_pdf_path(file_path: str | Path) -> Path:
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path: {path}")
    if path.suffix.lower() != PDF_EXTENSION:
        raise ValueError(f"Expected a PDF file: {path}")

    return path


def _validate_directory_path(data_dir: str | Path) -> Path:
    data_path = Path(data_dir).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_path}")
    if not data_path.is_dir():
        raise ValueError(f"Expected a directory path: {data_path}")

    return data_path


def _open_pdf(path: Path) -> PdfReader:
    try:
        return PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {path}") from exc


def _text_sample(reader: PdfReader, max_pages: int = 2) -> str:
    samples: list[str] = []

    for page in reader.pages[:max_pages]:
        text = _extract_page_text(page)
        if text:
            samples.append(text)

    return "\n".join(samples)


def _extract_page_text(page: Any) -> str:
    try:
        return (page.extract_text() or "").strip()
    except Exception as exc:
        print(f"[WARN] Could not extract page text: {exc}")
        return ""


def _page_metadata(
    *,
    path: Path,
    page_number: int,
    page_index: int,
    total_pages: int,
    document_type: DocumentType,
    source_document_id: str,
) -> dict[str, Any]:
    metadata = {
        "source": str(path),
        "source_document_id": source_document_id,
        "folder": path.parent.name,
        "file_name": path.name,
        "file_type": PDF_EXTENSION,
        "document_type": document_type.value,
        "content_type": "text",
        "page": page_number,
        "page_number": page_number,
        "page_index": page_index,
        "total_pages": total_pages,
    }
    metadata.update(_medicine_metadata(path, document_type))
    return metadata


def _medicine_metadata(path: Path, document_type: DocumentType) -> dict[str, str]:
    """Return medicine-specific metadata for drug leaflet PDFs."""

    if document_type != DocumentType.MEDICINE_LEAFLET:
        return {}

    return {
        "medicine_name": _display_name(path.stem, capitalize_first=True),
        "active_substance": _display_name(path.parent.name),
    }


def _display_name(raw_name: str, *, capitalize_first: bool = False) -> str:
    """Convert file or folder names into a readable label."""

    normalized = (
        raw_name.replace("_", " ")
        .replace("-", " ")
        .replace("&", " & ")
    )
    display_name = " ".join(normalized.split())
    if capitalize_first and display_name:
        return display_name[:1].upper() + display_name[1:]

    return display_name


def _source_document_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:20]
    return f"source_{digest}"


def _print_document_preview(document: Document, index: int) -> None:
    metadata = document.metadata
    preview = document.page_content.replace("\n", " ")[:500]

    print("=" * 80)
    print(f"Document #{index}")
    print(f"document_type: {metadata.get('document_type')}")
    print(f"medicine_name: {metadata.get('medicine_name')}")
    print(f"active_substance: {metadata.get('active_substance')}")
    print(f"content_type: {metadata.get('content_type')}")
    print(f"page: {metadata.get('page')}")
    print(f"source: {metadata.get('source')}")
    print(f"content: {preview}")


if __name__ == "__main__":
    configure_console_output()

    loaded_documents = load_all_documents(DEFAULT_DATA_DIR)
    print(f"Loaded {len(loaded_documents)} PDF pages.")

    for document_number, loaded_document in enumerate(loaded_documents, start=1):
        _print_document_preview(loaded_document, document_number)
