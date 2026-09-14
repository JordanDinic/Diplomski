import unittest
from pathlib import Path

from langchain_core.documents import Document

from diplomski.chunking import build_parent_child_documents
from diplomski.data_loader import load_pdf
from diplomski.document_sections import build_parent_sections


ROOT = Path(__file__).resolve().parents[1]
XANAX_PDF = ROOT / "Literatura/Lekovi/alprazolam/xanax.pdf"
GUIDE_PDF = ROOT / "Literatura/Vodic/Vodic_dobre_apotekarske_prakse.pdf"


def page_document(
    text: str,
    *,
    source: str = "sample.pdf",
    document_type: str = "medicine_leaflet",
    page: int = 1,
    medicine_name: str | None = None,
    active_substance: str | None = None,
) -> Document:
    metadata = {
        "source": source,
        "source_document_id": "source_sample",
        "folder": "Lekovi",
        "file_name": Path(source).name,
        "file_type": ".pdf",
        "document_type": document_type,
        "content_type": "text",
        "page": page,
        "page_number": page,
        "page_index": page - 1,
        "total_pages": 1,
    }
    if medicine_name:
        metadata["medicine_name"] = medicine_name
    if active_substance:
        metadata["active_substance"] = active_substance

    return Document(
        page_content=text,
        metadata=metadata,
    )


class DocumentSectionTests(unittest.TestCase):
    def test_cyrillic_leaflet_sections_are_detected(self) -> None:
        page = page_document(
            """
            \u0423\u041f\u0423\u0422\u0421\u0422\u0412\u041e \u0417\u0410 \u041b\u0415\u041a
            1. \u0428\u0442\u0430 \u0458\u0435 \u043b\u0435\u043a \u041f\u0440\u0438\u043c\u0435\u0440 \u0438 \u0447\u0435\u043c\u0443 \u0458\u0435 \u043d\u0430\u043c\u0435\u045a\u0435\u043d
            Prva sekcija.
            2. \u0428\u0442\u0430 \u0442\u0440\u0435\u0431\u0430 \u0434\u0430 \u0437\u043d\u0430\u0442\u0435 \u043f\u0440\u0435 \u043d\u0435\u0433\u043e \u0448\u0442\u043e \u0443\u0437\u043c\u0435\u0442\u0435 \u043b\u0435\u043a \u041f\u0440\u0438\u043c\u0435\u0440
            Druga sekcija.
            3. \u041a\u0430\u043a\u043e \u0441\u0435 \u0443\u0437\u0438\u043c\u0430 \u043b\u0435\u043a \u041f\u0440\u0438\u043c\u0435\u0440
            Treca sekcija.
            4. \u041c\u043e\u0433\u0443\u045b\u0430 \u043d\u0435\u0436\u0435\u0459\u0435\u043d\u0430 \u0434\u0435\u0458\u0441\u0442\u0432\u0430
            Cetvrta sekcija.
            5. \u041a\u0430\u043a\u043e \u0447\u0443\u0432\u0430\u0442\u0438 \u043b\u0435\u043a \u041f\u0440\u0438\u043c\u0435\u0440
            Peta sekcija.
            6. \u0421\u0430\u0434\u0440\u0436\u0430\u0458 \u043f\u0430\u043a\u043e\u0432\u0430\u045a\u0430 \u0438 \u043e\u0441\u0442\u0430\u043b\u0435 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u0458\u0435
            Sesta sekcija.
            """,
            source="Literatura/Lekovi/primer/primer-cirilica.pdf",
        )

        sections = build_parent_sections([page])

        self.assertEqual(
            [section.metadata["section_title"] for section in sections if section.metadata["section_index"] > 0],
            [
                "1. Sta je lek i cemu je namenjen",
                "2. Sta treba da znate pre nego sto uzmete lek",
                "3. Kako se uzima lek",
                "4. Moguca nezeljena dejstva",
                "5. Kako cuvati lek",
                "6. Sadrzaj pakovanja i ostale informacije",
            ],
        )

    def test_medicine_leaflet_sections_do_not_cross_main_headings(self) -> None:
        page = page_document(
            """
            UPUTSTVO ZA LEK
            1. Sta je lek Primer i cemu je namenjen
            Section one body.
            2. Sta treba da znate pre nego sto uzmete lek Primer
            Section two body.
            3. Kako se uzima lek Primer
            Section three body.
            4. Moguca nezeljena dejstva
            Section four body.
            5. Kako cuvati lek Primer
            Section five body.
            6. Sadrzaj pakovanja i ostale informacije
            Section six body.
            """,
            source="Literatura/Lekovi/primer/primer.pdf",
        )

        sections = build_parent_sections([page])
        titled_sections = [section for section in sections if section.metadata["section_index"] > 0]

        self.assertEqual(len(titled_sections), 6)
        self.assertEqual(
            [section.metadata["section_title"] for section in titled_sections],
            [
                "1. Sta je lek i cemu je namenjen",
                "2. Sta treba da znate pre nego sto uzmete lek",
                "3. Kako se uzima lek",
                "4. Moguca nezeljena dejstva",
                "5. Kako cuvati lek",
                "6. Sadrzaj pakovanja i ostale informacije",
            ],
        )
        self.assertIn("Section four body.", titled_sections[3].page_content)
        self.assertNotIn("Section five body.", titled_sections[3].page_content)

    def test_child_chunk_does_not_mix_two_sections(self) -> None:
        page = page_document(
            """
            1. Sta je lek Primer i cemu je namenjen
            Section one body.
            2. Sta treba da znate pre nego sto uzmete lek Primer
            Section two body.
            3. Kako se uzima lek Primer
            Section three body.
            4. Moguca nezeljena dejstva
            Section four body.
            5. Kako cuvati lek Primer
            Section five body.
            6. Sadrzaj pakovanja i ostale informacije
            Section six body.
            """,
            source="Literatura/Lekovi/primer/primer.pdf",
        )

        prepared = build_parent_child_documents([page], chunk_size=500, chunk_overlap=50)

        for child in prepared.child_documents:
            section_title = child.metadata["section_title"]
            content = child.page_content

            if section_title == "1. Sta je lek i cemu je namenjen":
                self.assertIn("Section one body.", content)
                self.assertNotIn("Section two body.", content)
            if section_title == "2. Sta treba da znate pre nego sto uzmete lek":
                self.assertIn("Section two body.", content)
                self.assertNotIn("Section three body.", content)

    def test_parent_child_chunks_keep_parent_metadata(self) -> None:
        page = page_document(
            """
            1. Sta je lek Primer i cemu je namenjen
            alpha beta gamma delta epsilon zeta eta theta iota kappa lambda
            2. Sta treba da znate pre nego sto uzmete lek Primer
            warnings body
            3. Kako se uzima lek Primer
            dosing body
            4. Moguca nezeljena dejstva
            adverse body
            5. Kako cuvati lek Primer
            storage body
            6. Sadrzaj pakovanja i ostale informacije
            package body
            """,
            source="Literatura/Lekovi/primer/primer.pdf",
            medicine_name="primer",
            active_substance="primer supstanca",
        )

        prepared = build_parent_child_documents([page], chunk_size=45, chunk_overlap=10)
        first_parent = prepared.parent_documents[0]
        first_child = prepared.child_documents[0]

        self.assertEqual(first_parent.metadata["record_type"], "parent")
        self.assertEqual(first_child.metadata["record_type"], "child")
        self.assertEqual(first_child.metadata["parent_id"], first_parent.metadata["parent_id"])
        self.assertEqual(first_child.metadata["section_id"], first_parent.metadata["section_id"])
        self.assertTrue(
            first_child.page_content.startswith(
                "Lek: primer\n"
                "Aktivna supstanca: primer supstanca\n"
                "Document: primer.pdf\n"
                "Section:"
            )
        )
        self.assertEqual(first_child.metadata["source"], "Literatura/Lekovi/primer/primer.pdf")
        self.assertEqual(first_child.metadata["page"], 1)
        self.assertEqual(first_child.metadata["document_type"], "medicine_leaflet")
        self.assertEqual(first_child.metadata["medicine_name"], "primer")
        self.assertEqual(first_child.metadata["active_substance"], "primer supstanca")
        self.assertIn("section_title", first_child.metadata)
        self.assertIn("child_chunk_index", first_child.metadata)
        self.assertIn("child_chunk_count", first_child.metadata)

    def test_interaction_reference_uses_known_heading_rules(self) -> None:
        page = page_document(
            """
            INTERAKCIJE LEKOVA
            Intro text.
            Vrste interakcija
            Type text.
            Farmakokineticke interakcije
            PK text.
            Resorpcija
            Absorption text.
            """,
            source="Literatura/Interakcije-lekova/interakcije-lekova.pdf",
            document_type="interaction_reference",
        )

        sections = build_parent_sections([page])
        titles = [section.metadata["section_title"] for section in sections]

        self.assertIn("INTERAKCIJE LEKOVA", titles)
        self.assertIn("Vrste interakcija", titles)
        self.assertIn("Farmakokineticke interakcije", titles)
        self.assertIn("Resorpcija", titles)

    @unittest.skipUnless(XANAX_PDF.exists(), "Local Xanax PDF is not available")
    def test_real_xanax_leaflet_has_six_main_sections(self) -> None:
        sections = build_parent_sections(load_pdf(XANAX_PDF))
        titles = [section.metadata["section_title"] for section in sections]

        self.assertIn("4. Moguca nezeljena dejstva", titles)
        self.assertEqual(
            [title for title in titles if title.startswith(tuple("123456"))],
            [
                "1. Sta je lek i cemu je namenjen",
                "2. Sta treba da znate pre nego sto uzmete lek",
                "3. Kako se uzima lek",
                "4. Moguca nezeljena dejstva",
                "5. Kako cuvati lek",
                "6. Sadrzaj pakovanja i ostale informacije",
            ],
        )

    @unittest.skipUnless(GUIDE_PDF.exists(), "Local guide PDF is not available")
    def test_real_guide_uses_pdf_outline_when_available(self) -> None:
        sections = build_parent_sections(load_pdf(GUIDE_PDF))
        outline_sections = [
            section
            for section in sections
            if section.metadata["section_detection"] == "pdf_outline"
        ]

        self.assertGreater(len(outline_sections), 40)
        self.assertEqual(
            outline_sections[0].metadata["section_title"],
            "VODI\u010c DOBRE APOTEKARSKE PRAKSE",
        )


if __name__ == "__main__":
    unittest.main()
