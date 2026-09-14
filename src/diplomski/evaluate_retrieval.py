from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

from diplomski.console import configure_console_output
from diplomski.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
)
from diplomski.vector_store import ChromaVectorStore


DEFAULT_EVAL_FILE = Path("eval/retrieval_gold.json")
IGNORED_EXPECTED_KEYS = {"label", "notes", "must_contain"}
CONTAINS_SUFFIX = "_contains"


def load_eval_cases(eval_file: str | Path = DEFAULT_EVAL_FILE) -> list[dict[str, Any]]:
    """Load retrieval evaluation cases from a JSON file."""

    path = Path(eval_file)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Retrieval eval file must contain a JSON list.")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Eval case #{index + 1} must be an object.")
        if not item.get("question"):
            raise ValueError(f"Eval case #{index + 1} is missing 'question'.")
        if not item.get("expected_chunks"):
            raise ValueError(f"Eval case #{index + 1} is missing 'expected_chunks'.")

    return data


def evaluate_case_at_k(
    eval_case: dict[str, Any],
    results: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """
    Calculate retrieval metrics for one question at k.

    precision@k = relevant results in first k positions / k
    recall@k = expected chunks found in first k positions / expected chunks
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    expected_chunks = list(eval_case["expected_chunks"])
    top_results = results[:k]
    matched_expected_indexes: set[int] = set()
    relevant_result_count = 0
    first_relevant_rank: int | None = None

    ranked_results: list[dict[str, Any]] = []
    for rank, result in enumerate(top_results, start=1):
        matched_indexes = matching_expected_indexes(result, expected_chunks)
        is_relevant = bool(matched_indexes)

        if is_relevant:
            relevant_result_count += 1
            first_relevant_rank = first_relevant_rank or rank
            matched_expected_indexes.update(matched_indexes)

        ranked_results.append(
            {
                "rank": rank,
                "is_relevant": is_relevant,
                "matched_expected": [
                    expected_label(expected_chunks[index]) for index in sorted(matched_indexes)
                ],
                "score": result.get("score"),
                "file_name": result_metadata_value(result, "file_name"),
                "medicine_name": result_metadata_value(result, "medicine_name"),
                "active_substance": result_metadata_value(result, "active_substance"),
                "section_title": result_metadata_value(result, "section_title"),
                "search_type": result.get("search_type"),
            }
        )

    expected_count = len(expected_chunks)
    matched_count = len(matched_expected_indexes)

    return {
        "id": eval_case.get("id"),
        "question": eval_case["question"],
        "k": k,
        "precision_at_k": relevant_result_count / k,
        "recall_at_k": matched_count / expected_count if expected_count else 0.0,
        "hit_at_k": 1.0 if matched_count else 0.0,
        "mrr_at_k": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        "relevant_results": relevant_result_count,
        "returned_results": len(top_results),
        "matched_expected": matched_count,
        "expected_total": expected_count,
        "missing_expected": [
            expected_label(expected_chunks[index])
            for index in range(expected_count)
            if index not in matched_expected_indexes
        ],
        "ranked_results": ranked_results,
    }


def matching_expected_indexes(
    result: dict[str, Any],
    expected_chunks: list[dict[str, Any]],
) -> set[int]:
    """Return all expected chunk indexes matched by a retrieval result."""

    return {
        index
        for index, expected_chunk in enumerate(expected_chunks)
        if result_matches_expected_chunk(result, expected_chunk)
    }


def result_matches_expected_chunk(
    result: dict[str, Any],
    expected_chunk: dict[str, Any],
) -> bool:
    """
    Match a retrieval result against expected metadata/text constraints.

    Exact keys such as file_name and section_title use normalized equality.
    Keys ending with _contains use normalized substring matching.
    must_contain checks phrases inside the retrieved text.
    """

    for key, expected_value in expected_chunk.items():
        if key in IGNORED_EXPECTED_KEYS:
            continue

        if key.endswith(CONTAINS_SUFFIX):
            actual_key = key[: -len(CONTAINS_SUFFIX)]
            actual_value = result_value(result, actual_key)
            if not value_contains(actual_value, expected_value):
                return False
            continue

        actual_value = result_value(result, key)
        if not value_matches(actual_value, expected_value):
            return False

    for phrase in expected_chunk.get("must_contain", []):
        if normalize_text(str(phrase)) not in normalize_text(str(result.get("text", ""))):
            return False

    return True


def value_matches(actual_value: Any, expected_value: Any) -> bool:
    """Compare scalar or list values using accent-insensitive text matching."""

    if actual_value is None:
        return False

    if isinstance(expected_value, list):
        return any(value_matches(actual_value, item) for item in expected_value)

    if isinstance(actual_value, list):
        return any(value_matches(item, expected_value) for item in actual_value)

    return normalize_text(str(actual_value)) == normalize_text(str(expected_value))


def value_contains(actual_value: Any, expected_value: Any) -> bool:
    """Return True if the normalized actual value contains expected text."""

    if actual_value is None:
        return False

    if isinstance(expected_value, list):
        return any(value_contains(actual_value, item) for item in expected_value)

    if isinstance(actual_value, list):
        return any(value_contains(item, expected_value) for item in actual_value)

    return normalize_text(str(expected_value)) in normalize_text(str(actual_value))


def result_value(result: dict[str, Any], key: str) -> Any:
    """Read a value either from top-level result fields or result metadata."""

    if key in {"text", "document", "page_content"}:
        return result.get("text")

    if key in result:
        return result[key]

    return result_metadata_value(result, key)


def result_metadata_value(result: dict[str, Any], key: str) -> Any:
    """Read a metadata value, including common plural fallbacks."""

    metadata = result.get("metadata") or {}
    if key in metadata:
        return metadata[key]

    fallbacks = {
        "source": "sources",
        "file_name": "file_names",
        "page_number": "page_numbers",
        "page": "page_number",
    }
    fallback_key = fallbacks.get(key)
    if fallback_key and fallback_key in metadata:
        return metadata[fallback_key]

    return None


def expected_label(expected_chunk: dict[str, Any]) -> str:
    """Create a readable label for one expected chunk."""

    if expected_chunk.get("label"):
        return str(expected_chunk["label"])

    parts = []
    for key in ("file_name", "medicine_name", "active_substance", "section_title"):
        if expected_chunk.get(key):
            parts.append(f"{key}={expected_chunk[key]}")

    for key in ("file_name_contains", "section_title_contains", "text_contains"):
        if expected_chunk.get(key):
            parts.append(f"{key}={expected_chunk[key]}")

    return " | ".join(parts) if parts else str(expected_chunk)


def normalize_text(text: str) -> str:
    """Normalize text for robust Serbian Latin/Cyrillic-light matching."""

    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    normalized = (
        without_accents.casefold()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )
    return " ".join(normalized.split())


def aggregate_metrics(case_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate macro averages over all evaluated questions for one k."""

    if not case_metrics:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr_at_k": 0.0,
        }

    return {
        "precision_at_k": average(item["precision_at_k"] for item in case_metrics),
        "recall_at_k": average(item["recall_at_k"] for item in case_metrics),
        "hit_at_k": average(item["hit_at_k"] for item in case_metrics),
        "mrr_at_k": average(item["mrr_at_k"] for item in case_metrics),
    }


def average(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def run_live_evaluation(
    *,
    eval_cases: list[dict[str, Any]],
    k_values: list[int],
    store: ChromaVectorStore,
    candidate_pool_size: int,
    use_hybrid_search: bool,
    group_by_source: bool,
    expand_to_parent: bool,
) -> dict[str, Any]:
    """Run all eval questions against Chroma Cloud and calculate metrics."""

    max_k = max(k_values)
    evaluated_cases: list[dict[str, Any]] = []

    for case in eval_cases:
        results = store.search(
            case["question"],
            top_k=max_k,
            candidate_count=candidate_pool_size,
            use_hybrid_search=use_hybrid_search,
            group_by_source=group_by_source,
            expand_to_parent=expand_to_parent,
        )
        metrics_by_k = {
            str(k): evaluate_case_at_k(case, results, k)
            for k in k_values
        }
        evaluated_cases.append(
            {
                "id": case.get("id"),
                "question": case["question"],
                "metrics_by_k": metrics_by_k,
            }
        )

    summary = {
        str(k): aggregate_metrics(
            [case["metrics_by_k"][str(k)] for case in evaluated_cases]
        )
        for k in k_values
    }

    return {
        "summary": summary,
        "cases": evaluated_cases,
    }


def print_evaluation_report(report: dict[str, Any], *, show_results: bool = False) -> None:
    """Print a readable retrieval evaluation report."""

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)
    print("SUMMARY")
    for k, metrics in report["summary"].items():
        print(
            f"@{k}: "
            f"precision={metrics['precision_at_k']:.4f} | "
            f"recall={metrics['recall_at_k']:.4f} | "
            f"hit={metrics['hit_at_k']:.4f} | "
            f"mrr={metrics['mrr_at_k']:.4f}"
        )

    print("\nPER QUESTION")
    max_k = max(int(k) for k in report["summary"])
    for item in report["cases"]:
        metrics = item["metrics_by_k"][str(max_k)]
        print("-" * 80)
        print(f"id: {item.get('id')}")
        print(f"question: {item['question']}")
        print(
            f"precision@{max_k}: {metrics['precision_at_k']:.4f} | "
            f"recall@{max_k}: {metrics['recall_at_k']:.4f} | "
            f"matched: {metrics['matched_expected']}/{metrics['expected_total']}"
        )
        if metrics["missing_expected"]:
            print("missing:")
            for missing in metrics["missing_expected"]:
                print(f"  - {missing}")

        if show_results:
            print("top results:")
            for result in metrics["ranked_results"]:
                marker = "OK" if result["is_relevant"] else "--"
                print(
                    f"  {marker} #{result['rank']} "
                    f"score={result['score']} "
                    f"file={result['file_name']} "
                    f"active={result['active_substance']} "
                    f"section={result['section_title']}"
                )


def write_json_report(report: dict[str, Any], output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Chroma retrieval with precision@k and recall@k.",
    )
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("-k", "--k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--cloud-host", default=None)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    )
    parser.add_argument(
        "--group-by-document-limit",
        type=int,
        default=DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    )
    parser.add_argument("--no-hybrid-search", action="store_true")
    parser.add_argument("--no-group-by-source", action="store_true")
    parser.add_argument(
        "--child-only",
        action="store_true",
        help="Evaluate raw retrieved child chunks instead of expanded parent sections.",
    )
    parser.add_argument("--show-results", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()

    k_values = sorted(set(args.k_values))
    eval_cases = load_eval_cases(args.eval_file)
    store = ChromaVectorStore(
        collection_name=args.collection_name,
        cloud_host=args.cloud_host,
        tenant=args.tenant,
        database=args.database,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        group_by_document_limit=args.group_by_document_limit,
    )
    report = run_live_evaluation(
        eval_cases=eval_cases,
        k_values=k_values,
        store=store,
        candidate_pool_size=args.candidate_pool_size,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        expand_to_parent=not args.child_only,
    )
    print_evaluation_report(report, show_results=args.show_results)

    if args.json_output:
        write_json_report(report, args.json_output)
        print(f"[INFO] Wrote JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
