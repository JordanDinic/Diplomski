from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import streamlit as st

from diplomski.rag_pipeline import RAGPipeline, RAGResponse, create_rag_pipeline
from diplomski.settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
)


PAGE_TITLE = "Apotekarski RAG asistent"
DISCLAIMER = (
    "Sistem je namenjen kao pomoc pri radu u apoteci i ne zamenjuje "
    "strucnu procenu farmaceuta ili lekara."
)


@dataclass(frozen=True)
class UISettings:
    """Runtime settings selected in the Streamlit sidebar."""

    collection_name: str
    top_k: int
    candidate_pool_size: int
    max_context_chars: int
    max_output_tokens: int
    group_by_document_limit: int
    use_hybrid_search: bool
    group_by_source: bool
    show_context: bool


def main() -> None:
    """Run the Streamlit user interface."""

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()

    settings = _render_sidebar()
    st.title(PAGE_TITLE)
    st.caption(DISCLAIMER)

    _render_chat(settings, can_ask=_render_status(settings))


def _render_sidebar() -> UISettings:
    with st.sidebar:
        st.header("Podesavanja")

        collection_name = st.text_input(
            "Chroma kolekcija",
            value=DEFAULT_COLLECTION_NAME,
        ).strip()
        top_k = st.slider(
            "Broj izvora",
            min_value=1,
            max_value=12,
            value=DEFAULT_TOP_K,
        )
        candidate_pool_size = st.slider(
            "Kandidati za pretragu",
            min_value=10,
            max_value=150,
            value=DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
            step=5,
        )
        group_by_document_limit = st.slider(
            "Rezultati po PDF-u",
            min_value=1,
            max_value=5,
            value=DEFAULT_GROUP_BY_DOCUMENT_LIMIT,
        )
        max_context_chars = st.slider(
            "Kontekst za Gemini",
            min_value=2_000,
            max_value=30_000,
            value=DEFAULT_MAX_CONTEXT_CHARS,
            step=1_000,
        )
        max_output_tokens = st.slider(
            "Duzina odgovora",
            min_value=1_000,
            max_value=20_000,
            value=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
            step=1_000,
        )

        use_hybrid_search = st.toggle(
            "Hybrid search",
            value=True,
        )
        group_by_source = st.toggle(
            "Grupisi po PDF-u",
            value=True,
        )
        show_context = st.toggle(
            "Prikazi kontekst",
            value=True,
        )
        if st.button("Ocisti razgovor", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return UISettings(
        collection_name=collection_name or DEFAULT_COLLECTION_NAME,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        max_context_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        group_by_document_limit=group_by_document_limit,
        use_hybrid_search=use_hybrid_search,
        group_by_source=group_by_source,
        show_context=show_context,
    )


def _render_status(settings: UISettings) -> bool:
    try:
        rag = _get_rag_pipeline(settings)
        store = rag.retriever.vector_store
        record_count = store.count()
    except Exception as exc:
        st.error(str(exc))
        return False

    col_collection, col_records, col_search = st.columns(3)
    col_collection.metric("Kolekcija", settings.collection_name)
    col_records.metric("Zapisa", record_count)
    col_search.metric(
        "Pretraga",
        "hybrid" if settings.use_hybrid_search else "dense",
    )
    return True


def _render_chat(settings: UISettings, *, can_ask: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            _render_message_content(message, show_context=settings.show_context)

    prompt = st.chat_input(
        "Unesite pitanje o lekovima, interakcijama ili apotekarskoj praksi",
        disabled=not can_ask,
    )
    if not can_ask:
        return
    if not prompt:
        _render_examples(settings)
        return

    _submit_question(prompt, settings)


def _submit_question(prompt: str, settings: UISettings) -> None:
    """Generate one answer and save its evidence before rendering it."""

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pretrazujem dokumente i generisem odgovor..."):
            try:
                response = _answer_question(prompt, settings)
            except Exception as exc:
                st.error(str(exc))
                return

        message = _assistant_message(response)
        st.session_state.messages.append(message)
        _render_message_content(message, show_context=settings.show_context)


def _assistant_message(response: RAGResponse) -> dict[str, Any]:
    """Snapshot the answer and displayed evidence as session-owned data."""

    return {
        "role": "assistant",
        "content": response.answer,
        "sources": copy.deepcopy(response.sources),
        "context": [
            {
                "text": document.text,
                "metadata": copy.deepcopy(_compact_metadata(document.metadata)),
                "score": document.score,
            }
            for document in response.retrieved_documents
        ],
    }


def _render_message_content(message: dict[str, Any], *, show_context: bool) -> None:
    """Render a saved message, including evidence when available."""

    st.markdown(message["content"])
    if message["role"] != "assistant":
        return

    # Older sessions only stored role/content, so their evidence is unavailable.
    if "sources" in message:
        _render_sources(message["sources"])
    if show_context and message.get("context"):
        _render_context(message["context"])


def _render_examples(settings: UISettings) -> None:
    examples = [
        "Koja su nezeljena dejstva amlodipina?",
        "Koje su vazne interakcije ibuprofena?",
        "Kako se postupa sa reklamacijama u apoteci?",
    ]

    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            _submit_question(example, settings)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.warning("Nema izvora.")
        return

    rows = []
    for source in sources:
        rows.append(
            {
                "Lek": source.get("medicine_name"),
                "Aktivna supstanca": source.get("active_substance"),
                "PDF": source.get("file_name"),
                "Sekcija": source.get("section"),
                "Strana": source.get("page"),
                "Tip": source.get("content_type"),
                "Score": _format_score(source.get("score")),
            }
        )

    st.subheader("Izvori")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_context(documents: list[dict[str, Any]]) -> None:
    st.subheader("Kontekst")
    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]
        label = (
            f"{index}. {metadata.get('file_name') or 'document'}"
            f" | {metadata.get('section_title') or 'bez sekcije'}"
            f" | score {_format_score(document['score'])}"
        )
        with st.expander(label):
            st.text(document["text"])
            st.json(_compact_metadata(metadata), expanded=False)


@st.cache_resource(show_spinner=False)
def _get_rag_pipeline(settings: UISettings) -> RAGPipeline:
    return create_rag_pipeline(
        collection_name=settings.collection_name,
        top_k=settings.top_k,
        candidate_pool_size=settings.candidate_pool_size,
        max_output_tokens=settings.max_output_tokens,
        max_context_chars=settings.max_context_chars,
        use_hybrid_search=settings.use_hybrid_search,
        group_by_source=settings.group_by_source,
        group_by_document_limit=settings.group_by_document_limit,
    )


def _answer_question(question: str, settings: UISettings) -> RAGResponse:
    rag = _get_rag_pipeline(settings)
    rag.top_k = settings.top_k
    rag.max_context_chars = settings.max_context_chars
    return rag.answer(question)


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "file_name",
        "medicine_name",
        "active_substance",
        "page_number",
        "page_numbers",
        "document_type",
        "section_title",
        "content_type",
        "record_type",
        "parent_id",
        "retrieved_child_score",
    )
    return {key: metadata.get(key) for key in keys if metadata.get(key) is not None}


def _format_score(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.05rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 8px;
        }
        textarea {
            min-height: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
