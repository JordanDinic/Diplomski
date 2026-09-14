import unittest

from langchain_core.documents import Document

from diplomski.settings import DEFAULT_CHROMA_DOCUMENT_MAX_BYTES
from diplomski.chroma_documents import (
    clean_metadata,
    document_id,
    prepare_documents_for_chroma,
)
from diplomski.vector_store import ChromaVectorStore


class VectorStoreHelperTests(unittest.TestCase):
    def test_prepare_documents_splits_text_above_chroma_limit(self) -> None:
        document = Document(
            page_content=("line with content\n" * 2000).strip(),
            metadata={
                "record_type": "parent",
                "content_type": "parent_section",
                "source": "sample.pdf",
                "source_document_id": "source_sample",
                "parent_id": "parent_1",
                "section_id": "section_1",
                "section_title": "Long section",
            },
        )

        prepared = prepare_documents_for_chroma([document])

        self.assertGreater(len(prepared), 1)
        self.assertTrue(
            all(
                len(item.page_content.encode("utf-8")) <= DEFAULT_CHROMA_DOCUMENT_MAX_BYTES
                for item in prepared
            )
        )
        self.assertTrue(all(item.metadata["record_type"] == "parent" for item in prepared))
        self.assertTrue(all(item.metadata["parent_id"] == "parent_1" for item in prepared))
        self.assertEqual(
            [item.metadata["cloud_subchunk_index"] for item in prepared],
            list(range(len(prepared))),
        )
        self.assertTrue(all(item.metadata["split_from_oversized_document"] for item in prepared))

    def test_clean_metadata_keeps_trace_fields_and_removes_unlisted_large_fields(self) -> None:
        metadata = {
            "source": "sample.pdf",
            "page": 3,
            "document_type": "medicine_leaflet",
            "medicine_name": "Norvasc",
            "active_substance": "amlodipin",
            "section_title": "4. Moguca nezeljena dejstva",
            "orig_elements": ["x" * 10000],
        }

        cleaned = clean_metadata(metadata)

        self.assertEqual(cleaned["source"], "sample.pdf")
        self.assertEqual(cleaned["page"], 3)
        self.assertEqual(cleaned["document_type"], "medicine_leaflet")
        self.assertEqual(cleaned["medicine_name"], "Norvasc")
        self.assertEqual(cleaned["active_substance"], "amlodipin")
        self.assertEqual(cleaned["section_title"], "4. Moguca nezeljena dejstva")
        self.assertNotIn("orig_elements", cleaned)

    def test_document_id_is_stable_for_same_document(self) -> None:
        document = Document(
            page_content="Same text",
            metadata={
                "record_type": "child",
                "source": "sample.pdf",
                "source_document_id": "source_sample",
                "parent_id": "parent_1",
                "section_id": "section_1",
                "child_chunk_index": 0,
                "cloud_subchunk_index": 0,
            },
        )

        self.assertEqual(document_id(document, 0), document_id(document, 0))

    def test_parent_expansion_returns_parent_section_text(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.collection = FakeParentCollection()

        expanded = store.expand_results_to_parents(
            [
                {
                    "id": "child_1",
                    "score": 0.9,
                    "chroma_score": 0.1,
                    "distance": None,
                    "search_type": "dense_child",
                    "text": "Document: sample.pdf\nSection: S\n\nsmall child",
                    "metadata": {
                        "record_type": "child",
                        "parent_id": "parent_1",
                        "child_chunk_index": 0,
                    },
                }
            ]
        )

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["text"], "parent part 1\nparent part 2")
        self.assertEqual(expanded[0]["metadata"]["record_type"], "parent")
        self.assertEqual(expanded[0]["metadata"]["retrieved_child_id"], "child_1")
        self.assertEqual(expanded[0]["child_text"], "Document: sample.pdf\nSection: S\n\nsmall child")


class FakeParentCollection:
    def get(self, where, include):
        return {
            "documents": ["parent part 2", "parent part 1"],
            "metadatas": [
                {
                    "record_type": "parent",
                    "parent_id": "parent_1",
                    "cloud_subchunk_index": 1,
                    "section_title": "S",
                },
                {
                    "record_type": "parent",
                    "parent_id": "parent_1",
                    "cloud_subchunk_index": 0,
                    "section_title": "S",
                },
            ],
        }


if __name__ == "__main__":
    unittest.main()
