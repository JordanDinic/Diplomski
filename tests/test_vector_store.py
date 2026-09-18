import unittest
from unittest.mock import Mock

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


class VectorStoreSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.children = [
            {
                "id": f"child_{index}",
                "text": f"Child text {index}",
                "score": 0.9 - index / 10,
                "search_type": "dense_child",
                "metadata": {
                    "source": "sample.pdf",
                    "record_type": "child",
                    "parent_id": "parent_1",
                    "child_chunk_index": index,
                },
            }
            for index in range(3)
        ]
        self.store = ChromaVectorStore.__new__(ChromaVectorStore)
        self.store.use_hybrid_search = True
        self.store.group_by_source = False
        self.store.record_count = Mock(return_value=len(self.children))
        self.store.hybrid_search = Mock(return_value=self.children)
        self.store.dense_search = Mock(return_value=self.children)
        self.store.parents_by_id = Mock(
            side_effect=AssertionError("Child-only retrieval must not fetch parents.")
        )

    def test_child_only_search_keeps_hits_from_the_same_section(self) -> None:
        for hybrid in (True, False):
            with self.subTest(hybrid=hybrid):
                results = self.store.query(
                    "query", top_k=2, candidate_count=3,
                    use_hybrid_search=hybrid, expand_to_parent=False,
                )

                self.assertEqual(results, self.children[:2])
                self.store.parents_by_id.assert_not_called()

    def test_dense_fallback_also_skips_parent_expansion(self) -> None:
        self.store.hybrid_search.side_effect = RuntimeError("Hybrid unavailable")

        results = self.store.search("query", top_k=2, expand_to_parent=False)

        self.assertEqual(results, self.children[:2])
        self.store.dense_search.assert_called_once()
        self.store.parents_by_id.assert_not_called()

    def test_parent_expansion_remains_available_when_requested(self) -> None:
        self.store.parents_by_id.side_effect = None
        self.store.parents_by_id.return_value = {
            "parent_1": {
                "text": "Full parent section",
                "metadata": {"record_type": "parent", "parent_id": "parent_1"},
            }
        }

        results = self.store.query("query", top_k=2, expand_to_parent=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Full parent section")
        self.assertEqual(results[0]["child_text"], self.children[0]["text"])
        self.assertEqual(results[0]["score"], self.children[0]["score"])
        self.store.parents_by_id.assert_called_once()


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
