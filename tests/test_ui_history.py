import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from diplomski.rag_pipeline import RAGResponse
from diplomski.retriever import RetrievedDocument


UI_PATH = Path(__file__).resolve().parents[1] / "src/diplomski/ui_app.py"


def make_response(question: str) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=f"Answer for {question}",
        sources=[{"file_name": f"{question}.pdf", "page": [2, 3], "score": 0.9}],
        retrieved_documents=[RetrievedDocument(
            text=f"Evidence for {question}",
            metadata={"file_name": f"{question}.pdf", "page_numbers": [2, 3]},
            score=0.9,
        )],
        prompt=f"Prompt for {question}",
    )


class UIHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.rag = Mock()
        self.rag.retriever.vector_store.count.return_value = 20
        self.rag.answer.side_effect = make_response
        factory = patch("diplomski.rag_pipeline.create_rag_pipeline", return_value=self.rag)
        factory.start()
        self.addCleanup(factory.stop)
        self.app = AppTest.from_file(str(UI_PATH), default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def ask(self, question: str) -> None:
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)

    def toggle_context(self, enabled: bool) -> None:
        control = next(t for t in self.app.toggle if t.label == "Prikazi kontekst")
        control.set_value(enabled).run()
        self.assertFalse(self.app.exception)

    def test_each_answer_keeps_its_evidence_after_reruns_and_new_questions(self) -> None:
        self.ask("first")
        self.app.run()
        self.assertEqual(self.app.dataframe[0].value["PDF"].tolist(), ["first.pdf"])
        self.ask("second")
        self.app.run()

        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_message), 4)
        self.assertEqual(len(self.app.dataframe), 2)
        self.assertEqual(self.app.dataframe[0].value["PDF"].tolist(), ["first.pdf"])
        self.assertEqual(self.app.dataframe[1].value["PDF"].tolist(), ["second.pdf"])
        self.assertEqual([t.value for t in self.app.text], ["Evidence for first", "Evidence for second"])
        self.assertEqual(self.rag.answer.call_count, 2)

    def test_hidden_context_is_saved_and_settings_do_not_regenerate_answers(self) -> None:
        self.toggle_context(False)
        self.ask("hidden")
        self.assertEqual(len(self.app.expander), 0)
        self.assertEqual(len(self.app.dataframe), 1)
        self.toggle_context(True)
        self.assertEqual([t.value for t in self.app.text], ["Evidence for hidden"])
        slider = next(s for s in self.app.slider if s.label == "Broj izvora")
        slider.set_value(7).run()
        self.assertEqual(len(self.app.dataframe), 1)
        self.assertEqual([t.value for t in self.app.text], ["Evidence for hidden"])
        self.toggle_context(False)
        self.toggle_context(True)
        self.rag.answer.assert_called_once_with("hidden")

    def test_example_answers_are_saved_and_clear_removes_all_history(self) -> None:
        example = "Koje su vazne interakcije ibuprofena?"
        next(b for b in self.app.button if b.label == example).click().run()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state.messages), 2)
        self.assertEqual([t.value for t in self.app.text], [f"Evidence for {example}"])
        self.assertEqual(len(self.app.dataframe), 1)
        next(b for b in self.app.button if b.label == "Ocisti razgovor").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state.messages, [])
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertEqual(len(self.app.dataframe), 0)
        self.assertEqual(len(self.app.expander), 0)
        self.rag.answer.assert_called_once_with(example)

    def test_history_is_a_snapshot_and_supports_older_text_only_messages(self) -> None:
        self.app.session_state.messages = [{"role": "assistant", "content": "Legacy answer"}]
        response = make_response("snapshot")
        self.rag.answer.side_effect = None
        self.rag.answer.return_value = response
        self.ask("snapshot")
        response.sources[0]["file_name"] = "changed.pdf"
        response.sources[0]["page"].append(100)
        response.retrieved_documents[0].text = "Changed evidence"
        response.retrieved_documents[0].metadata["page_numbers"].append(100)
        self.app.run()

        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_message), 3)
        self.assertEqual(self.app.dataframe[0].value["PDF"].tolist(), ["snapshot.pdf"])
        self.assertEqual([t.value for t in self.app.text], ["Evidence for snapshot"])
        message = self.app.session_state.messages[-1]
        self.assertEqual(message["sources"][0]["page"], [2, 3])
        self.assertEqual(message["context"][0]["metadata"]["page_numbers"], [2, 3])

    def test_backend_unavailable_does_not_hide_saved_evidence(self) -> None:
        self.ask("saved")
        self.rag.retriever.vector_store.count.side_effect = RuntimeError("Cloud unavailable")
        self.app.run()

        self.assertFalse(self.app.exception)
        self.assertIn("Cloud unavailable", [error.value for error in self.app.error])
        self.assertTrue(self.app.chat_input[0].disabled)
        self.assertEqual(len(self.app.dataframe), 1)
        self.assertEqual([t.value for t in self.app.text], ["Evidence for saved"])
        self.rag.answer.assert_called_once_with("saved")

    def test_failed_generation_does_not_create_an_answer_or_duplicate_sources(self) -> None:
        self.ask("saved")
        self.rag.answer.side_effect = RuntimeError("Generation failed")
        self.ask("failed")
        self.assertIn("Generation failed", [error.value for error in self.app.error])
        self.app.run()

        self.assertFalse(self.app.exception)
        self.assertEqual([m["role"] for m in self.app.session_state.messages],
                         ["user", "assistant", "user"])
        self.assertEqual(len(self.app.dataframe), 1)
        self.assertEqual([t.value for t in self.app.text], ["Evidence for saved"])
        self.assertEqual(self.rag.answer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
