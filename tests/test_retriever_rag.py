import unittest

from diplomski.rag_pipeline import RAGPipeline
from diplomski.retriever import ChromaRetriever, RetrievedDocument


class RetrieverTests(unittest.TestCase):
    def test_retriever_delegates_query_to_vector_store(self) -> None:
        store = FakeVectorStore()
        retriever = ChromaRetriever(
            vector_store=store,
            candidate_pool_size=12,
            use_hybrid_search=False,
            group_by_source=False,
        )

        documents = retriever.retrieve("nezeljena dejstva", k=3)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "retrieved text")
        self.assertEqual(store.calls[0]["query"], "nezeljena dejstva")
        self.assertEqual(store.calls[0]["top_k"], 3)
        self.assertEqual(store.calls[0]["candidate_count"], 12)


class RAGPipelineTests(unittest.TestCase):
    def test_rag_pipeline_uses_retriever_context_and_llm(self) -> None:
        retriever = FakeRetriever(
            [
                RetrievedDocument(
                    text="Document: norvasc.pdf\nSection: 4. Moguca nezeljena dejstva\n\nKontext.",
                    metadata={
                        "source": "norvasc.pdf",
                        "file_name": "norvasc.pdf",
                        "medicine_name": "Norvasc",
                        "active_substance": "amlodipin",
                        "page_number": 4,
                        "section_title": "4. Moguca nezeljena dejstva",
                        "content_type": "parent_section",
                    },
                    score=0.91,
                    document_id="parent_1",
                )
            ]
        )
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=1)

        response = pipeline.answer("Koja su nezeljena dejstva amlodipina?")

        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(retriever.queries, ["Koja su nezeljena dejstva amlodipina?"])
        self.assertIn("Kontext.", llm.prompts[0])
        self.assertEqual(response.sources[0]["file_name"], "norvasc.pdf")
        self.assertEqual(response.sources[0]["medicine_name"], "Norvasc")
        self.assertEqual(response.sources[0]["active_substance"], "amlodipin")
        self.assertEqual(response.sources[0]["section"], "4. Moguca nezeljena dejstva")

    def test_rag_pipeline_does_not_call_llm_without_context(self) -> None:
        retriever = FakeRetriever([])
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever=retriever, llm=llm)

        response = pipeline.answer("Nepoznato pitanje?")

        self.assertEqual(llm.prompts, [])
        self.assertIn("Nemam dovoljno informacija", response.answer)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = []

    def query(
        self,
        query_text,
        top_k,
        candidate_count,
        use_hybrid_search,
        group_by_source,
    ):
        self.calls.append(
            {
                "query": query_text,
                "top_k": top_k,
                "candidate_count": candidate_count,
                "use_hybrid_search": use_hybrid_search,
                "group_by_source": group_by_source,
            }
        )
        return [
            {
                "id": "result_1",
                "text": "retrieved text",
                "metadata": {"source": "sample.pdf"},
                "score": 0.8,
                "chroma_score": 0.2,
                "distance": None,
                "search_type": "dense_child_parent",
            }
        ]


class FakeRetriever:
    def __init__(self, documents) -> None:
        self.documents = documents
        self.queries = []

    def retrieve(self, query, k=5):
        self.queries.append(query)
        return self.documents[:k]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts = []

    def generate(self, prompt, system_instruction=None):
        self.prompts.append(prompt)
        return "generated answer"


if __name__ == "__main__":
    unittest.main()
