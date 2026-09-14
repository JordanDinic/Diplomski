import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from diplomski.data_loader import load_all_documents, load_pdf
from diplomski.document_types import DocumentType, detect_document_type


ROOT = Path(__file__).resolve().parents[1]
XANAX_PDF = ROOT / "Literatura/Lekovi/alprazolam/xanax.pdf"
GUIDE_PDF = ROOT / "Literatura/Vodic/Vodic_dobre_apotekarske_prakse.pdf"
INTERACTIONS_PDF = ROOT / "Literatura/Interakcije-lekova/interakcije-lekova.pdf"


class DocumentTypeTests(unittest.TestCase):
    def test_detects_document_type_from_known_paths(self) -> None:
        self.assertEqual(
            detect_document_type("Literatura/Lekovi/alprazolam/xanax.pdf"),
            DocumentType.MEDICINE_LEAFLET,
        )
        self.assertEqual(
            detect_document_type("Literatura/Vodic/Vodic_dobre_apotekarske_prakse.pdf"),
            DocumentType.PRACTICE_GUIDE,
        )
        self.assertEqual(
            detect_document_type("Literatura/Interakcije-lekova/interakcije-lekova.pdf"),
            DocumentType.INTERACTION_REFERENCE,
        )

    def test_detects_leaflet_from_text_when_path_is_unknown(self) -> None:
        self.assertEqual(
            detect_document_type("tmp/random.pdf", "UPUTSTVO ZA LEK\nXanax"),
            DocumentType.MEDICINE_LEAFLET,
        )
        self.assertEqual(
            detect_document_type(
                "tmp/random.pdf",
                "\u0423\u041f\u0423\u0422\u0421\u0422\u0412\u041e "
                "\u0417\u0410 \u041b\u0415\u041a\nLexilium",
            ),
            DocumentType.MEDICINE_LEAFLET,
        )


class DataLoaderTests(unittest.TestCase):
    def test_load_pdf_returns_page_documents_with_trace_metadata(self) -> None:
        first_page = Mock()
        first_page.extract_text.return_value = "UPUTSTVO ZA LEK\nSample medicine"
        second_page = Mock()
        second_page.extract_text.return_value = "Second page text"
        reader = Mock()
        reader.pages = [first_page, second_page]

        with patch(
            "diplomski.data_loader._validate_pdf_path",
            return_value=Path("Literatura/Lekovi/sample-substance/test.pdf"),
        ), \
             patch("diplomski.data_loader._open_pdf", return_value=reader):
            documents = load_pdf("ignored.pdf")

        self.assertEqual(len(documents), 2)
        self.assertTrue(all(isinstance(document, Document) for document in documents))
        self.assertEqual(documents[0].page_content, "UPUTSTVO ZA LEK\nSample medicine")
        self.assertEqual(documents[0].metadata["document_type"], "medicine_leaflet")
        self.assertEqual(
            documents[0].metadata["source"],
            "Literatura\\Lekovi\\sample-substance\\test.pdf",
        )
        self.assertEqual(documents[0].metadata["file_name"], "test.pdf")
        self.assertEqual(documents[0].metadata["folder"], "sample-substance")
        self.assertEqual(documents[0].metadata["medicine_name"], "Test")
        self.assertEqual(documents[0].metadata["active_substance"], "sample substance")
        self.assertEqual(documents[0].metadata["page"], 1)
        self.assertEqual(documents[0].metadata["page_number"], 1)
        self.assertEqual(documents[0].metadata["page_index"], 0)
        self.assertEqual(documents[0].metadata["total_pages"], 2)
        self.assertEqual(documents[1].metadata["page"], 2)

    @unittest.skipUnless(XANAX_PDF.exists(), "Local Xanax PDF is not available")
    def test_real_leaflet_loads_as_page_documents(self) -> None:
        documents = load_pdf(XANAX_PDF)

        self.assertEqual(len(documents), 9)
        self.assertEqual(
            {document.metadata["document_type"] for document in documents},
            {"medicine_leaflet"},
        )
        self.assertEqual(documents[0].metadata["file_name"], "xanax.pdf")
        self.assertEqual(documents[0].metadata["medicine_name"], "Xanax")
        self.assertEqual(documents[0].metadata["active_substance"], "alprazolam")
        self.assertIn("Xanax", documents[0].page_content)

    @unittest.skipUnless(GUIDE_PDF.exists(), "Local guide PDF is not available")
    def test_real_guide_loads_as_practice_guide(self) -> None:
        documents = load_pdf(GUIDE_PDF)

        self.assertEqual(len(documents), 93)
        self.assertEqual(
            {document.metadata["document_type"] for document in documents},
            {"practice_guide"},
        )
        self.assertNotIn("medicine_name", documents[0].metadata)
        self.assertNotIn("active_substance", documents[0].metadata)

    @unittest.skipUnless(INTERACTIONS_PDF.exists(), "Local interactions PDF is not available")
    def test_real_interactions_loads_as_reference_document(self) -> None:
        documents = load_pdf(INTERACTIONS_PDF)

        self.assertEqual(len(documents), 20)
        self.assertEqual(
            {document.metadata["document_type"] for document in documents},
            {"interaction_reference"},
        )

    def test_load_all_documents_skips_bad_pdf_and_continues(self) -> None:
        with patch("diplomski.data_loader._validate_directory_path", return_value=Path("data")), \
             patch.object(Path, "rglob", return_value=[Path("bad.pdf"), Path("good.pdf")]), \
             patch("diplomski.data_loader.load_pdf", side_effect=[ValueError("bad"), [Document(page_content="ok")]]):
            documents = load_all_documents("data")

        self.assertEqual([document.page_content for document in documents], ["ok"])


if __name__ == "__main__":
    unittest.main()
