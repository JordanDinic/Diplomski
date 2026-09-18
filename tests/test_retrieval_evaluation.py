import unittest
from unittest.mock import Mock, patch

from diplomski.evaluate_retrieval import (
    evaluate_case_at_k,
    parse_args,
    result_matches_expected_chunk,
    run_live_evaluation,
    value_contains,
    value_matches,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_cli_defaults_to_child_results_and_allows_parent_comparison(self) -> None:
        for arguments, expected in (
            ([], False),
            (["--child-only"], False),
            (["--expand-to-parent"], True),
        ):
            with self.subTest(arguments=arguments):
                with patch("sys.argv", ["evaluate_retrieval.py", *arguments]):
                    self.assertEqual(parse_args().expand_to_parent, expected)

    def test_live_evaluation_passes_and_records_context_mode(self) -> None:
        store = Mock()
        store.search.return_value = []
        cases = [{
            "id": "sample", "question": "query",
            "expected_chunks": [{"file_name": "sample.pdf"}],
        }]

        for expand in (False, True):
            with self.subTest(expand_to_parent=expand):
                report = run_live_evaluation(
                    eval_cases=cases, k_values=[1], store=store,
                    candidate_pool_size=60, use_hybrid_search=True,
                    group_by_source=True, expand_to_parent=expand,
                )

                self.assertEqual(store.search.call_args.kwargs["expand_to_parent"], expand)
                self.assertEqual(report["expand_to_parent"], expand)

    def test_result_matches_expected_chunk_by_metadata(self) -> None:
        result = {
            "text": "Lek: Norvasc\nAktivna supstanca: amlodipin",
            "metadata": {
                "file_name": "norvasc.pdf",
                "active_substance": "amlodipin",
                "section_title": "4. Moguca nezeljena dejstva",
            },
        }
        expected = {
            "file_name": "norvasc.pdf",
            "active_substance": "amlodipin",
            "section_title": "4. Moguca nezeljena dejstva",
        }

        self.assertTrue(result_matches_expected_chunk(result, expected))

    def test_contains_matching_is_accent_insensitive(self) -> None:
        self.assertTrue(
            value_contains(
                "Aktivnost A: Planiranje, nabavka, skladištenje i čuvanje lekova",
                "skladistenje i cuvanje",
            )
        )

    def test_list_values_can_match_expected_scalar(self) -> None:
        self.assertTrue(value_matches(["norvasc.pdf", "xanax.pdf"], "xanax.pdf"))

    def test_evaluate_case_calculates_precision_and_recall_at_k(self) -> None:
        eval_case = {
            "id": "sample",
            "question": "Koja su nezeljena dejstva amlodipina?",
            "expected_chunks": [
                {
                    "label": "Norvasc adverse effects",
                    "file_name": "norvasc.pdf",
                    "section_title": "4. Moguca nezeljena dejstva",
                },
                {
                    "label": "Amlopin Combo adverse effects",
                    "file_name": "amlopin-combo.pdf",
                    "section_title": "4. Moguca nezeljena dejstva",
                },
            ],
        }
        results = [
            {
                "text": "matched norvasc",
                "score": 0.9,
                "metadata": {
                    "file_name": "norvasc.pdf",
                    "section_title": "4. Moguca nezeljena dejstva",
                },
            },
            {
                "text": "wrong section",
                "score": 0.8,
                "metadata": {
                    "file_name": "norvasc.pdf",
                    "section_title": "3. Kako se uzima lek",
                },
            },
        ]

        metrics = evaluate_case_at_k(eval_case, results, k=2)

        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["hit_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 1.0)
        self.assertEqual(metrics["matched_expected"], 1)
        self.assertEqual(metrics["expected_total"], 2)


if __name__ == "__main__":
    unittest.main()
