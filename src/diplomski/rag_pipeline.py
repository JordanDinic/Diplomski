from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from diplomski.console import configure_console_output
from diplomski.gemini_client import GeminiFlashClient
from diplomski.rag_prompt import (
    RAG_SYSTEM_INSTRUCTION,
    build_rag_prompt,
    extract_sources,
    format_context,
)
from diplomski.retriever import ChromaRetriever, RetrievedDocument
from diplomski.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)


@dataclass
class RAGResponse:
    """Final RAG answer with retrieved context and source traceability."""

    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_documents: list[RetrievedDocument]
    prompt: str


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve context, build prompt, ask Gemini."""

    def __init__(
        self,
        retriever: ChromaRetriever,
        llm: GeminiFlashClient,
        top_k: int = DEFAULT_TOP_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def answer(self, question: str) -> RAGResponse:
        """Answer a user question using retrieved documents as context."""

        retrieved_documents = self.retriever.retrieve(question, k=self.top_k)
        prompt = build_rag_prompt(
            question=question,
            documents=retrieved_documents,
            max_context_chars=self.max_context_chars,
        )

        if not retrieved_documents:
            answer_text = (
                "Nemam dovoljno informacija iz dostupnih dokumenata, "
                "jer retrieval nije vratio relevantan kontekst."
            )
        else:
            answer_text = self.llm.generate(
                prompt=prompt,
                system_instruction=RAG_SYSTEM_INSTRUCTION,
            )

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=extract_sources(retrieved_documents),
            retrieved_documents=retrieved_documents,
            prompt=prompt,
        )


def print_rag_response(
    response: RAGResponse,
    *,
    show_context: bool = False,
    preview_chars: int = 1200,
) -> None:
    """Print the final answer and optional retrieval context."""

    print("=" * 80)
    print("ODGOVOR")
    print("=" * 80)
    print(response.answer)

    print("\n" + "=" * 80)
    print("IZVORI")
    print("=" * 80)
    if not response.sources:
        print("[Nema izvora]")
    else:
        for index, source in enumerate(response.sources, start=1):
            print(
                f"{index}. file={source.get('file_name')} | "
                f"lek={source.get('medicine_name')} | "
                f"aktivna_supstanca={source.get('active_substance')} | "
                f"page={source.get('page')} | "
                f"section={source.get('section')} | "
                f"score={source.get('score'):.4f} | "
                f"source={source.get('source')}"
            )

    if show_context:
        print("\n" + "=" * 80)
        print("KONTEKST")
        print("=" * 80)
        print(
            format_context(
                response.retrieved_documents,
                max_context_chars=preview_chars,
            )
        )


def create_rag_pipeline(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    cloud_host: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    gemini_model: str | None = None,
    max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    use_hybrid_search: bool = True,
    group_by_source: bool = True,
    group_by_document_limit: int = DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> RAGPipeline:
    """Create a configured RAG pipeline from Chroma Cloud and Gemini Flash."""

    retriever = ChromaRetriever(
        collection_name=collection_name,
        cloud_host=cloud_host,
        tenant=tenant,
        database=database,
        candidate_pool_size=candidate_pool_size,
        use_hybrid_search=use_hybrid_search,
        group_by_source=group_by_source,
        group_by_document_limit=group_by_document_limit,
    )
    llm = GeminiFlashClient(
        model_name=gemini_model,
        max_output_tokens=max_output_tokens,
    )

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question using Chroma Cloud search and Gemini Flash.",
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer. Quotes are optional; words are joined with spaces.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved documents to use.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma Cloud collection name.",
    )
    parser.add_argument(
        "--cloud-host",
        default=None,
        help="Chroma Cloud host. Defaults to CHROMA_HOST or api.trychroma.com.",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Chroma Cloud tenant. Defaults to CHROMA_TENANT.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Chroma Cloud database. Defaults to CHROMA_DATABASE.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
        help="Maximum Gemini output tokens.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Maximum number of context characters sent to Gemini.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
        help="Number of dense/sparse candidates before RRF and GroupBy.",
    )
    parser.add_argument(
        "--group-by-document-limit",
        type=int,
        default=DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
        help="Maximum chunks kept per source document when GroupBy is enabled.",
    )
    parser.add_argument(
        "--no-hybrid-search",
        action="store_true",
        help="Use dense query fallback instead of Cloud hybrid RRF search.",
    )
    parser.add_argument(
        "--no-group-by-source",
        action="store_true",
        help="Do not group/deduplicate results by source document.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context after the answer.",
    )
    parser.add_argument(
        "--context-preview-chars",
        type=int,
        default=1200,
        help="Number of context characters to print when --show-context is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_console_output()

    args = _parse_args()
    question_text = " ".join(args.question)

    rag = create_rag_pipeline(
        collection_name=args.collection_name,
        cloud_host=args.cloud_host,
        tenant=args.tenant,
        database=args.database,
        gemini_model=args.gemini_model,
        max_output_tokens=args.max_output_tokens,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        group_by_document_limit=args.group_by_document_limit,
        max_context_chars=args.max_context_chars,
    )
    rag_response = rag.answer(question_text)
    print_rag_response(
        rag_response,
        show_context=args.show_context,
        preview_chars=args.context_preview_chars,
    )
