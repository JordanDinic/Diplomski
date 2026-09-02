from __future__ import annotations

from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf


PDF_EXTENSION = ".pdf"
DEFAULT_DATA_DIR = "Literatura/Lekovi"


def load_pdf(file_path: str | Path) -> list[Any]:
    """
    Load one PDF with Unstructured and return raw extracted elements.

    This layer only parses PDFs and adds traceability metadata. Chunking,
    conversion to LangChain Documents, embeddings and vector-store writes
    happen later in the RAG pipeline.
    """

    path = _validate_pdf_path(file_path)
    elements = partition_pdf(
        filename=str(path),
        strategy="fast",
        languages=["srp_latn"],
        infer_table_structure=True,
    )

    for element_index, element in enumerate(elements):
        _add_file_metadata(element, path, element_index)

    return elements


def load_all_elements(data_dir: str | Path) -> list[Any]:
    """
    Load all PDF files from a directory and return Unstructured elements.

    Invalid or unreadable PDFs are skipped with a warning so one bad file does
    not stop the complete ingestion run.
    """

    data_path = _validate_directory_path(data_dir)
    elements: list[Any] = []

    for pdf_path in sorted(data_path.rglob(f"*{PDF_EXTENSION}")):
        try:
            elements.extend(load_pdf(pdf_path))
        except Exception as exc:
            print(f"[WARN] Skipping {pdf_path}: {exc}")

    return elements


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


def _add_file_metadata(element: Any, path: Path, element_index: int) -> None:
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return

    category = _category(element)
    metadata.source = str(path)
    metadata.folder = path.parent.name
    metadata.file_name = path.name
    metadata.file_type = PDF_EXTENSION
    metadata.element_index = element_index
    metadata.category = category
    metadata.content_type = "table" if category in {"Table", "TableChunk"} else "text"


def _category(element: Any) -> str:
    return str(getattr(element, "category", type(element).__name__))


def _print_element_preview(element: Any, index: int) -> None:
    metadata = getattr(element, "metadata", None)
    category = _category(element)
    preview = str(element).strip().replace("\n", " ")[:500]

    print("=" * 80)
    print(f"Element #{index}")
    print(f"category: {category}")
    print(f"content_type: {getattr(metadata, 'content_type', None)}")
    print(f"page: {getattr(metadata, 'page_number', None)}")
    print(f"element_index: {getattr(metadata, 'element_index', None)}")
    print(f"source: {getattr(metadata, 'source', None)}")
    print(f"content: {preview}")

    if category in {"Table", "TableChunk"}:
        print("\ntable_html:")
        print(getattr(metadata, "text_as_html", None) or "[no HTML available]")


if __name__ == "__main__":
    loaded_elements = load_all_elements(DEFAULT_DATA_DIR)
    print(f"Loaded {len(loaded_elements)} PDF elements.")

    for element_number, loaded_element in enumerate(loaded_elements, start=1):
        _print_element_preview(loaded_element, element_number)
