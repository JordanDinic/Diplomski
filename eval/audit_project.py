"""Collect thesis evaluation evidence without uploading or deleting Cloud records.

Run from the project root with the project's Python environment:
    python eval/audit_project.py --live --rag
Live checks use paid APIs when the configured providers charge for requests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

from diplomski import settings
from diplomski.chroma_documents import document_id, prepare_documents_for_chroma
from diplomski.chunking import build_parent_child_documents, split_text
from diplomski.data_loader import load_pdf
from diplomski.document_sections import LEAFLET_SECTION_TITLES, _flatten_outline
from diplomski.evaluate_retrieval import (
    aggregate_metrics, evaluate_case_at_k, load_eval_cases, result_matches_expected_chunk,
)
from diplomski.gemini_client import GeminiFlashClient
from diplomski.rag_pipeline import RAGPipeline
from diplomski.rag_prompt import extract_sources, format_context
from diplomski.retriever import ChromaRetriever, RetrievedDocument
from diplomski.vector_store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]


class ReadOnlyStore(ChromaVectorStore):
    """Require an existing collection even though the application creates one."""

    def get_or_create_collection(self) -> Any:
        return self.client.get_collection(
            name=self.collection_name, embedding_function=self.dense_embedding_function,
        )

    def reset(self) -> None:
        raise RuntimeError("Database mutations are disabled during evaluation.")

    def delete_collection_if_exists(self) -> None:
        raise RuntimeError("Database mutations are disabled during evaluation.")

    def upload_documents(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Database mutations are disabled during evaluation.")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0], "median": statistics.median(ordered),
        "mean": statistics.mean(ordered), "max": ordered[-1],
        "p95_nearest_rank": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
    }


def corpus_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    all_parents: list[Document] = []
    all_children: list[Document] = []
    for pdf in sorted((ROOT / "Literatura").rglob("*.pdf")):
        print(f"[AUDIT] PDF {pdf.name}", flush=True)
        pages = load_pdf(pdf)
        prepared = build_parent_child_documents(pages)
        reader = PdfReader(str(pdf))
        parents, children = prepared.parent_documents, prepared.child_documents
        outline = _flatten_outline(reader.outline, reader)
        file_report = {
            "file": str(pdf.relative_to(ROOT)), "sha256": sha256(pdf),
            "pdf_pages": len(reader.pages), "extracted_pages": len(pages),
            "document_type": pages[0].metadata["document_type"] if pages else "empty",
            "parent_count": len(parents), "child_count": len(children),
            "section_methods": dict(Counter(p.metadata["section_detection"] for p in parents)),
            "six_standard_titles_detected": set(LEAFLET_SECTION_TITLES.values()).issubset(
                {p.metadata["section_title"] for p in parents}
            ),
            "outline_entries": len(outline),
            "outline_markers_used": sum(p.metadata["section_detection"] == "pdf_outline" for p in parents),
            "sections": [{
                "title": p.metadata["section_title"], "path": p.metadata["section_path"],
                "pages": p.metadata["page_numbers"], "characters": len(p.page_content),
                "bytes": len(p.page_content.encode("utf-8")),
                "body_preview": p.page_content.split("\n\n", 1)[-1][:160],
            } for p in parents],
        }
        files.append(file_report)
        all_parents.extend(parents)
        all_children.extend(children)
    uploaded = prepare_documents_for_chroma(all_parents + all_children)
    parents_by_id = {p.metadata["parent_id"]: p for p in all_parents}
    reachable_gold = []
    for case in cases:
        for expected in case["expected_chunks"]:
            matches = [p for p in all_parents if result_matches_expected_chunk(
                {"text": p.page_content, "metadata": p.metadata}, expected,
            )]
            reachable_gold.append({
                "case_id": case["id"], "expected": expected, "matching_parents": len(matches),
            })
    return {
        "files": files, "pdf_count": len(files),
        "document_types": dict(Counter(f["document_type"] for f in files)),
        "pdf_pages": sum(f["pdf_pages"] for f in files),
        "extracted_pages": sum(f["extracted_pages"] for f in files),
        "parent_count": len(all_parents), "child_count": len(all_children),
        "prepared_cloud_records": len(uploaded),
        "prepared_record_types": dict(Counter(d.metadata["record_type"] for d in uploaded)),
        "child_characters_including_prefix": distribution([len(c.page_content) for c in all_children]),
        "parent_characters_including_prefix": distribution([len(p.page_content) for p in all_parents]),
        "parents_above_context_budget": sum(len(p.page_content) > settings.DEFAULT_MAX_CONTEXT_CHARS for p in all_parents),
        "parents_above_cloud_byte_limit": sum(len(p.page_content.encode("utf-8")) > settings.DEFAULT_CHROMA_DOCUMENT_MAX_BYTES for p in all_parents),
        "orphan_children": sum(c.metadata["parent_id"] not in parents_by_id for c in all_children),
        "children_missing_required_metadata": sum(any(k not in c.metadata for k in (
            "source", "page", "document_type", "section_title", "parent_id",
        )) for c in all_children),
        "children_missing_document_or_section_prefix": sum(
            "Document:" not in c.page_content.split("\n\n", 1)[0]
            or "Section:" not in c.page_content.split("\n\n", 1)[0] for c in all_children
        ),
        "gold_reachability": reachable_gold,
    }


def contract_probes() -> dict[str, Any]:
    document = Document(page_content="A stable document", metadata={
        "source": "sample.pdf", "source_document_id": "sample", "parent_id": "section",
        "section_id": "section", "record_type": "child", "child_chunk_index": 0,
    })
    first = RetrievedDocument(text="A" * 16000 + " REQUIRED_EVIDENCE", metadata={
        "source": "first.pdf", "file_name": "first.pdf", "page_number": 1,
    }, score=0.01)
    second = RetrievedDocument(text="SECOND_DOCUMENT_EVIDENCE", metadata={
        "source": "second.pdf", "file_name": "second.pdf", "page_number": 1,
    }, score=0.009)
    context = format_context([first, second])
    words = " ".join(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"] * 10)
    chunks = split_text(words, chunk_size=45, chunk_overlap=10)
    observed_words = set(" ".join(chunks).split())
    return {
        "document_id_unchanged_after_index_change": document_id(document, 0) == document_id(document, 100),
        "long_parent_evidence_preserved_in_prompt": "REQUIRED_EVIDENCE" in context,
        "second_source_preserved_in_prompt": "SECOND_DOCUMENT_EVIDENCE" in context,
        "reported_sources": extract_sources([first, second]),
        "word_splitting_probe": {"chunk_size": 45, "overlap": 10, "chunks": chunks,
                                 "non_original_tokens": sorted(observed_words - set(words.split()))},
    }


def to_retrieved(result: dict[str, Any]) -> RetrievedDocument:
    return RetrievedDocument(
        text=result["text"], metadata=result["metadata"], score=result["score"],
        document_id=result.get("id"), search_type=result.get("search_type"),
    )


def cloud_inventory(store: ReadOnlyStore, output: Path) -> None:
    records: dict[str, list[Any]] = {"ids": [], "metadatas": []}
    offset = 0
    while True:
        batch = store.collection.get(include=["metadatas"], limit=300, offset=offset)
        if not batch["ids"]:
            break
        records["ids"].extend(batch["ids"])
        records["metadatas"].extend(batch["metadatas"])
        offset += len(batch["ids"])
    write_json(output / "cloud_inventory.json", {
        "count": store.count(), "fetched_count": len(records["ids"]),
        "collection": store.collection_name,
        "record_types": dict(Counter(m.get("record_type") for m in records["metadatas"])),
        "file_names": sorted({m.get("file_name", "") for m in records["metadatas"]}),
        "orphan_parent_ids": sorted(
            {m.get("parent_id") for m in records["metadatas"] if m.get("record_type") == "child"}
            - {m.get("parent_id") for m in records["metadatas"] if m.get("record_type") == "parent"}
        ),
    })


def live_retrieval(
    store: ReadOnlyStore, cases: list[dict[str, Any]], output: Path,
    variants: list[tuple[str, bool, bool]] | None = None,
    *, expand_to_parent: bool = False,
) -> None:
    if not (output / "cloud_inventory.json").exists():
        cloud_inventory(store, output)
    variants = variants or [("hybrid_grouped", True, True), ("hybrid_ungrouped", True, False),
                            ("dense_ungrouped", False, False)]
    for name, hybrid, group in variants:
        evaluated = []
        for case in cases:
            print(f"[AUDIT] {name}: {case['id']}", flush=True)
            start = time.perf_counter()
            results = store.search(case["question"], top_k=5, candidate_count=60,
                                   use_hybrid_search=hybrid, group_by_source=group,
                                   expand_to_parent=expand_to_parent)
            seconds = time.perf_counter() - start
            for result in results:
                result["metadata"] = {k: v for k, v in result["metadata"].items()
                                      if k != store.sparse_key}
            context = format_context([to_retrieved(r) for r in results])
            row = {
                "id": case["id"], "question": case["question"], "seconds": seconds,
                "metrics_by_k": {str(k): evaluate_case_at_k(case, results, k) for k in (1, 3, 5)},
                "results": results, "prompt_context_characters": len(context),
                "context_document_headers": context.count("[Dokument "),
                "child_body_fully_in_prompt": [
                    " ".join(r.get("child_text", r["text"]).split("\n\n", 1)[-1].split())
                    in " ".join(context.split()) for r in results
                ],
            }
            evaluated.append(row)
            report = {
                "variant": name, "requested_hybrid": hybrid, "group_by_source": group,
                "candidate_pool_size": 60, "group_by_document_limit": store.group_by_document_limit,
                "expand_to_parent": expand_to_parent, "completed_cases": len(evaluated),
                "summary": {str(k): aggregate_metrics([r["metrics_by_k"][str(k)] for r in evaluated])
                            for k in (1, 3, 5)},
                "latency_seconds": distribution([r["seconds"] for r in evaluated]),
                "cases": evaluated,
            }
            write_json(output / f"{name}.json", report)


def rag_smoke(store: ReadOnlyStore, output: Path) -> None:
    llm = GeminiFlashClient()
    rag = RAGPipeline(retriever=ChromaRetriever(vector_store=store), llm=llm)
    cases = [
        ("leaflet", "Koja su nezeljena dejstva leka Norvasc?"),
        ("guide", "Kako dobra apotekarska praksa opisuje skladistenje i cuvanje lekova?"),
        ("interaction", "Sta su farmakokineticke interakcije lekova?"),
        ("missing_medicine", "Sta pise u dokumentima o leku Zorvexalin-TEST-999?"),
        ("out_of_domain", "Ko je osvojio Svetsko prvenstvo u fudbalu 2018. godine?"),
    ]
    rows = []
    for name, question in cases:
        print(f"[AUDIT] RAG {name}", flush=True)
        start = time.perf_counter()
        response = rag.answer(question)
        for document in response.retrieved_documents:
            document.metadata.pop(store.sparse_key, None)
        rows.append({"case": name, "seconds": time.perf_counter() - start,
                     "response": asdict(response)})
        write_json(output / "rag_smoke.json", {
            "model": llm.model_name, "max_context_chars": rag.max_context_chars,
            "max_output_tokens": llm.max_output_tokens, "cases": rows,
            "note": "Smoke checks only; responses require separate evidence-based human assessment.",
        })


def ui_smoke() -> dict[str, Any]:
    """Check Streamlit rerun behavior with a fake backend and no API calls."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from streamlit.testing.v1 import AppTest
    from diplomski.rag_pipeline import RAGResponse

    metadata = {"file_name": "demo.pdf", "section_title": "Demo", "page_number": 1}
    response = RAGResponse(
        question="demo", answer="Demo answer.", prompt="Demo prompt",
        sources=[{"file_name": "demo.pdf", "section": "Demo", "page": 1, "score": 0.1}],
        retrieved_documents=[RetrievedDocument(text="Demo context", metadata=metadata, score=0.1)],
    )
    fake = SimpleNamespace(
        retriever=SimpleNamespace(vector_store=SimpleNamespace(count=lambda: 1)),
        answer=lambda question: response,
    )
    with patch("diplomski.rag_pipeline.create_rag_pipeline", return_value=fake):
        app = AppTest.from_file(str(ROOT / "src/diplomski/ui_app.py"), default_timeout=30).run()
        initial_errors = len(app.exception)
        app.chat_input[0].set_value("Demo question").run()
        after_answer = {"errors": len(app.exception), "source_tables": len(app.dataframe),
                        "context_expanders": len(app.expander), "chat_messages": len(app.chat_message)}
        app.run()
        after_rerun = {"errors": len(app.exception), "source_tables": len(app.dataframe),
                       "context_expanders": len(app.expander), "chat_messages": len(app.chat_message)}
    return {"mode": "Streamlit AppTest with fake backend; no browser layout assessment",
            "initial_errors": initial_errors, "after_answer": after_answer, "after_rerun": after_rerun}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Query the existing Cloud collection.")
    parser.add_argument("--rag", action="store_true", help="Run five real Gemini smoke checks.")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "eval" / "runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f"[AUDIT] Output: {output}", flush=True)
    files = sorted((ROOT / "src" / "diplomski").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    write_json(output / "manifest.json", {
        "started_at_utc": stamp, "python": sys.version,
        "packages": {p: version(p) for p in ("chromadb", "google-genai", "langchain-core", "pypdf", "streamlit")},
        "settings": {k: v for k, v in vars(settings).items() if k.startswith("DEFAULT_")},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "gold_sha256": sha256(ROOT / "eval/retrieval_gold.json"),
        "challenges_sha256": sha256(ROOT / "eval/retrieval_challenges.json"),
        "audit_script_sha256": sha256(Path(__file__)),
        "live_requested": args.live, "rag_requested": args.rag,
    })
    tests = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (output / "unit_tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    write_json(output / "unit_tests.json", {"returncode": tests.returncode})
    print(f"[AUDIT] Unit tests exit code: {tests.returncode}", flush=True)
    cases = load_eval_cases(ROOT / "eval/retrieval_gold.json")
    write_json(output / "corpus.json", corpus_audit(cases))
    write_json(output / "contract_probes.json", contract_probes())
    write_json(output / "ui_smoke.json", ui_smoke())
    if args.live or args.rag:
        store = ReadOnlyStore()
        if args.live:
            live_retrieval(store, cases, output)
            challenges = load_eval_cases(ROOT / "eval/retrieval_challenges.json")
            live_retrieval(store, challenges, output, [
                ("challenges_hybrid_grouped", True, True),
                ("challenges_dense_ungrouped", False, False),
            ])
        if args.rag:
            rag_smoke(store, output)
    print(f"[AUDIT] Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
