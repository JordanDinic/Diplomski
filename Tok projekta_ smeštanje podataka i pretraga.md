# Tok projekta: smeštanje podataka i pretraga

Ovaj dokument objedinjuje dva procesa:

1. **smeštanje podataka iz PDF dokumenata u Chroma Cloud**;
2. **pretragu kada korisnik postavi pitanje**.

---

# 1. Tok smeštanja podataka iz PDF-a u bazu

Smeštanje podataka može se posmatrati kao ingestion pipeline. Proces počinje pokretanjem `vector_store.py`, a završava se pozivom:

```python
self.collection.upsert(...)
```

Lokalni kod priprema tekst i metapodatke. Dense i sparse embedding vektore generiše Chroma Cloud na osnovu konfiguracije kolekcije.

## 1.1. Pokretanje ingestion procesa

Primer komande:

```powershell
.\.venv\Scripts\python.exe src\diplomski\vector_store.py `
    --data-dir Literatura `
    --reset
```

Glavna funkcija u `vector_store.py` povezuje sve faze:

```python
def main() -> None:
    configure_console_output()
    args = parse_args()

    pages = load_all_documents(args.data_dir)

    prepared = build_parent_child_documents(
        pages,
        chunk_size=args.max_characters,
        chunk_overlap=args.overlap,
    )

    store = ChromaVectorStore(
        collection_name=args.collection_name,
        cloud_host=args.cloud_host,
        tenant=args.tenant,
        database=args.database,
        store_batch_size=args.store_batch_size,
        use_hybrid_search=not args.no_hybrid_search,
        group_by_source=not args.no_group_by_source,
        reset_collection=args.reset,
    )

    store.upload_documents(
        prepared.parent_documents + prepared.child_documents
    )
```

Osnovni redosled je:

```text
parse_args()
    ↓
load_all_documents()
    ↓
build_parent_child_documents()
    ↓
ChromaVectorStore(...)
    ↓
upload_documents()
```

## 1.2. Čitanje komandnih argumenata

Funkcija `parse_args()` definiše parametre komandne linije:

```python
parser.add_argument(
    "--data-dir",
    default="Literatura",
    help="Directory with PDF files.",
)

parser.add_argument(
    "--max-characters",
    type=int,
    default=DEFAULT_CHUNK_MAX_CHARACTERS,
)

parser.add_argument(
    "--overlap",
    type=int,
    default=DEFAULT_CHUNK_OVERLAP,
)

parser.add_argument(
    "--reset",
    action="store_true",
)
```

Najvažniji argumenti su:

| Argument | Uloga |
|---|---|
| `--data-dir` | Direktorijum u kojem se nalaze PDF dokumenti. |
| `--max-characters` | Maksimalna veličina child chunk-a. |
| `--overlap` | Preklapanje susednih child chunk-ova. |
| `--reset` | Brisanje postojeće kolekcije pre novog upisa. |
| `--collection-name` | Naziv Chroma Cloud kolekcije. |
| `--store-batch-size` | Broj zapisa koji se šalje u jednom batch-u. |
| `--no-hybrid-search` | Isključivanje hibridne pretrage. |
| `--no-group-by-source` | Isključivanje grupisanja po PDF izvoru. |

## 1.3. Učitavanje svih PDF dokumenata

U `data_loader.py` poziva se:

```python
pages = load_all_documents(args.data_dir)
```

Implementacija:

```python
def load_all_documents(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[Document]:
    data_path = _validate_directory_path(data_dir)
    documents: list[Document] = []

    for pdf_path in sorted(data_path.rglob("*.pdf")):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:
            print(f"[WARN] Skipping {pdf_path}: {exc}")

    return documents
```

Funkcija:

1. proverava da li direktorijum postoji;
2. rekurzivno pronalazi `.pdf` fajlove;
3. sortira pronađene putanje;
4. za svaki PDF poziva `load_pdf()`;
5. dodaje stranice u zajedničku listu;
6. preskače PDF ako se ne može pročitati.

Rezultat je lista `Document` objekata. U ovoj fazi svaki `Document` predstavlja jednu PDF stranicu.

## 1.4. Obrada jednog PDF dokumenta

Za svaki PDF poziva se `load_pdf(pdf_path)`:

```python
def load_pdf(file_path: str | Path) -> list[Document]:
    path = _validate_pdf_path(file_path)
    reader = _open_pdf(path)

    document_type = detect_document_type(
        path,
        _text_sample(reader),
    )

    source_document_id = _source_document_id(path)
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = _extract_page_text(page)

        if not text:
            continue

        page_number = page_index + 1

        documents.append(
            Document(
                page_content=text,
                metadata=_page_metadata(
                    path=path,
                    page_number=page_number,
                    page_index=page_index,
                    total_pages=len(reader.pages),
                    document_type=document_type,
                    source_document_id=source_document_id,
                ),
            )
        )

    return documents
```

PDF se otvara pomoću `PdfReader` klase:

```python
def _open_pdf(path: Path) -> PdfReader:
    try:
        return PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {path}") from exc
```

Tekst stranice izdvaja se ovako:

```python
def _extract_page_text(page: Any) -> str:
    try:
        return (page.extract_text() or "").strip()
    except Exception as exc:
        print(f"[WARN] Could not extract page text: {exc}")
        return ""
```

Ako stranica nema tekstualni sloj, rezultat je prazan string i stranica se preskače. U ovoj fazi projekat ne koristi OCR.

## 1.5. Prepoznavanje tipa dokumenta

Pre formiranja page-level zapisa poziva se:

```python
document_type = detect_document_type(
    path,
    _text_sample(reader),
)
```

Funkcija u `document_types.py` koristi putanju dokumenta i uzorak teksta:

```python
def detect_document_type(
    file_path: str | Path,
    text_sample: str = "",
) -> DocumentType:
    path = Path(file_path)
    normalized_path = _normalize_text(" ".join(path.parts))
    normalized_text = _normalize_text(text_sample)
    combined = f"{normalized_path}\n{normalized_text}"

    if "vodic" in normalized_path or "dobre apotekarske prakse" in combined:
        return DocumentType.PRACTICE_GUIDE

    if "interakcije lekova" in combined or "interakcije-lekova" in normalized_path:
        return DocumentType.INTERACTION_REFERENCE

    if "lekovi" in normalized_path:
        return DocumentType.MEDICINE_LEAFLET

    if "uputstvo za lek" in combined or CYRILLIC_LEAFLET_MARKER in combined:
        return DocumentType.MEDICINE_LEAFLET

    return DocumentType.UNKNOWN
```

Mogući tipovi su:

```python
class DocumentType(str, Enum):
    MEDICINE_LEAFLET = "medicine_leaflet"
    PRACTICE_GUIDE = "practice_guide"
    INTERACTION_REFERENCE = "interaction_reference"
    UNKNOWN = "unknown"
```

## 1.6. Formiranje metapodataka stranice

Metapodaci se formiraju funkcijom `_page_metadata()`:

```python
def _page_metadata(
    *,
    path: Path,
    page_number: int,
    page_index: int,
    total_pages: int,
    document_type: DocumentType,
    source_document_id: str,
) -> dict[str, Any]:
    metadata = {
        "source": str(path),
        "source_document_id": source_document_id,
        "folder": path.parent.name,
        "file_name": path.name,
        "file_type": ".pdf",
        "document_type": document_type.value,
        "content_type": "text",
        "page": page_number,
        "page_number": page_number,
        "page_index": page_index,
        "total_pages": total_pages,
    }

    metadata.update(_medicine_metadata(path, document_type))
    return metadata
```

Za uputstva za lekove podaci se izvode iz organizacije fajlova:

```python
def _medicine_metadata(path: Path, document_type: DocumentType) -> dict[str, str]:
    if document_type != DocumentType.MEDICINE_LEAFLET:
        return {}

    return {
        "medicine_name": _display_name(path.stem, capitalize_first=True),
        "active_substance": _display_name(path.parent.name),
    }
```

Primer početnog zapisa:

```python
Document(
    page_content="Tekst jedne PDF stranice...",
    metadata={
        "source": ".../xanax.pdf",
        "source_document_id": "source_...",
        "folder": "alprazolam",
        "file_name": "xanax.pdf",
        "document_type": "medicine_leaflet",
        "content_type": "text",
        "page": 1,
        "medicine_name": "Xanax",
        "active_substance": "alprazolam",
    },
)
```

## 1.7. Formiranje parent sekcija i child chunk-ova

U `main()` se poziva:

```python
prepared = build_parent_child_documents(
    pages,
    chunk_size=args.max_characters,
    chunk_overlap=args.overlap,
)
```

Implementacija:

```python
def build_parent_child_documents(
    page_documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_MAX_CHARACTERS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ParentChildDocuments:
    parent_sections = build_parent_sections(page_documents)

    parent_documents = [
        _with_retrieval_context(parent)
        for parent in parent_sections
    ]

    child_documents = chunk_parent_sections(
        parent_sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return ParentChildDocuments(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )
```

Ova funkcija:

1. page-level dokumente pretvara u parent sekcije;
2. parent sekcijama dodaje retrieval kontekst;
3. parent sekcije deli na child chunk-ove.

## 1.8. Formiranje parent sekcija

U `document_sections.py` bira se algoritam prema tipu dokumenta:

```python
def build_parent_sections(
    page_documents: list[Document],
) -> list[Document]:
    parent_sections: list[Document] = []

    for source_documents in _documents_by_source(page_documents):
        document_type = _document_type(source_documents)

        if document_type == DocumentType.MEDICINE_LEAFLET.value:
            sections = _medicine_leaflet_sections(source_documents)
        elif document_type == DocumentType.PRACTICE_GUIDE.value:
            sections = _practice_guide_sections(source_documents)
        elif document_type == DocumentType.INTERACTION_REFERENCE.value:
            sections = _interaction_reference_sections(source_documents)
        else:
            sections = _page_sections(
                source_documents,
                method="page_fallback",
            )

        parent_sections.extend(sections)

    return parent_sections
```

Za uputstva za lekove traže se standardne sekcije. Za Vodič se koriste PDF outline/bookmarks podaci kada postoje. Za dokument o interakcijama koriste se unapred poznati naslovi. Ako odgovarajuće oznake nisu pronađene, koristi se fallback po stranicama.

Parent zapis nastaje ovako:

```python
def _section_document(...):
    metadata = dict(source_document.metadata)
    section_id = _section_id(
        metadata,
        section_index,
        title,
        page_numbers,
    )

    metadata.update(
        {
            "record_type": "parent",
            "content_type": "parent_section",
            "section_index": section_index,
            "section_id": section_id,
            "parent_id": section_id,
            "section_title": title,
            "section_path": path,
            "section_detection": method,
            "page_numbers": page_numbers,
        }
    )

    return Document(
        page_content=text.strip(),
        metadata=metadata,
    )
```

## 1.9. Deljenje parent sekcija na child chunk-ove

Za svaku parent sekciju poziva se `split_text()`:

```python
def chunk_parent_sections(
    parent_sections: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_MAX_CHARACTERS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    child_documents: list[Document] = []

    for parent in parent_sections:
        section_chunks = split_text(
            parent.page_content,
            chunk_size,
            chunk_overlap,
        )

        for child_index, chunk_text in enumerate(section_chunks):
            metadata = _child_metadata(
                parent.metadata,
                child_index,
                len(section_chunks),
            )

            child_documents.append(
                Document(
                    page_content=_text_with_retrieval_context(
                        chunk_text,
                        metadata,
                    ),
                    metadata=metadata,
                )
            )

    return child_documents
```

Child metapodaci sadrže:

```text
record_type = child
content_type = child_chunk
parent_id
section_id
child_chunk_index
child_chunk_count
```

## 1.10. Kako radi `split_text()`

Tekst se deli hijerarhijski:

```text
pasusi
    ↓
linije
    ↓
reči
    ↓
direktno sečenje ako je potrebno
```

```python
def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[str]:
    clean_text = text.strip()

    if not clean_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if len(clean_text) <= chunk_size:
        return [clean_text]

    overlap = max(
        0,
        min(chunk_overlap, chunk_size // 2),
    )

    units = _split_units(clean_text, chunk_size)
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = _join_text(current, unit)

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = _join_text(
                _overlap_tail(current, overlap),
                unit,
            )
        else:
            chunks.append(unit.strip())
            current = ""

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]
```

## 1.11. Dodavanje retrieval prefiksa

Child tekstu se dodaje kontekst:

```python
def _text_with_retrieval_context(
    text: str,
    metadata: dict[str, Any],
) -> str:
    header_lines: list[str] = []

    medicine_name = _metadata_text(
        metadata,
        "medicine_name",
    )
    active_substance = _metadata_text(
        metadata,
        "active_substance",
    )
    document_name = _document_display_name(metadata)
    section = (
        metadata.get("section_path")
        or metadata.get("section_title")
    )

    if medicine_name:
        header_lines.append(f"Lek: {medicine_name}")
    if active_substance:
        header_lines.append(
            f"Aktivna supstanca: {active_substance}"
        )
    if document_name:
        header_lines.append(
            f"Document: {document_name}"
        )
    if section:
        header_lines.append(
            f"Section: {section}"
        )

    return "\n".join(header_lines) + "\n\n" + text.strip()
```

Primer child teksta:

```text
Lek: Norvasc
Aktivna supstanca: amlodipin
Document: norvasc.pdf
Section: 4. Moguca nezeljena dejstva

Amlodipin može izazvati glavobolju i vrtoglavicu.
```

Prefiks je deo teksta koji ulazi u embedding. Metapodaci se čuvaju posebno i koriste se za filtriranje, proširivanje i prikaz izvora.

## 1.12. Spajanje parent i child zapisa

U `main()` se šalju obe liste:

```python
store.upload_documents(
    prepared.parent_documents
    + prepared.child_documents
)
```

Dakle, u kolekciju se upisuju i parent i child zapisi. Primarna pretraga kasnije koristi child zapise, dok se parent zapisi dohvaćaju pomoću `parent_id` vrednosti.

## 1.13. Kreiranje Chroma Cloud objekta

Konstruktor `ChromaVectorStore` kreira dense i sparse embedding funkcije:

```python
self.dense_embedding_function = qwen_embedding_function()
self.sparse_embedding_function = splade_embedding_function()

self.client = chromadb.CloudClient(
    tenant=self.tenant,
    database=self.database,
    api_key=self.api_key,
    cloud_host=self.cloud_host,
)
```

Dense funkcija:

```python
def qwen_embedding_function():
    return ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        task=DEFAULT_CHROMA_QWEN_TASK,
        instructions=QWEN_RETRIEVAL_INSTRUCTIONS,
    )
```

Sparse funkcija:

```python
def splade_embedding_function():
    return ChromaCloudSpladeEmbeddingFunction(
        model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1,
    )
```

Ove funkcije definišu backend koji Chroma Cloud koristi za embedding. Vektori se ne računaju ručno u lokalnom kodu.

## 1.14. Kreiranje kolekcije i šeme

```python
def get_or_create_collection(self) -> Any:
    self.client.get_or_create_collection(
        name=self.collection_name,
        schema=collection_schema(
            dense_embedding_function=self.dense_embedding_function,
            sparse_embedding_function=self.sparse_embedding_function,
            sparse_key=self.sparse_key,
        ),
        embedding_function=None,
        metadata={
            "embedding_backend": "chroma_cloud",
            "dense_model": "Qwen",
            "sparse_model": "SPLADE",
            "sparse_key": self.sparse_key,
        },
    )

    return self.client.get_collection(
        name=self.collection_name,
        embedding_function=self.dense_embedding_function,
    )
```

Prvi poziv obezbeđuje kolekciju i njenu šemu. Drugi vraća objekat koji se koristi za `upsert`, `query`, `search` i `get` operacije.

## 1.15. Priprema zapisa neposredno pre upisa

Pozivom:

```python
prepared_documents = prepare_documents_for_chroma(documents)
```

`chroma_documents.py` proverava veličinu teksta u UTF-8 bajtovima:

```python
def prepare_documents_for_chroma(
    documents: Iterable[Document],
    *,
    max_bytes: int = DEFAULT_CHROMA_DOCUMENT_MAX_BYTES,
) -> list[Document]:
    prepared: list[Document] = []

    for document_index, document in enumerate(documents):
        parts = split_text_by_byte_limit(
            document.page_content,
            max_bytes=max_bytes,
        )

        source_id = source_document_id(document)

        for part_index, part in enumerate(parts):
            metadata = dict(document.metadata)
            metadata.setdefault("record_type", "child")
            metadata.setdefault("content_type", "child_chunk")
            metadata.setdefault("source_document_id", source_id)
            metadata["cloud_chunk_index"] = len(prepared)
            metadata["cloud_subchunk_index"] = part_index
            metadata["document_bytes"] = len(
                part.encode("utf-8")
            )

            prepared.append(
                Document(
                    page_content=part,
                    metadata=metadata,
                )
            )

    return prepared
```

Jedan ulazni dokument može ostati jedan zapis ili biti podeljen na više cloud poddelova.

## 1.16. Čišćenje metapodataka

Pre slanja u bazu poziva se:

```python
clean_metadata(document.metadata)
```

```python
def clean_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    for key in CHROMA_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            cleaned[key] = safe_metadata_value(value)

    return cleaned
```

Čuvaju se podaci potrebni za retrieval, parent-child vezu i izvore, kao što su:

```text
record_type
source
source_document_id
file_name
document_type
medicine_name
active_substance
page
page_numbers
section_id
section_title
section_path
parent_id
child_chunk_index
child_chunk_count
cloud_subchunk_index
```

## 1.17. Formiranje ID-ja

Za svaki zapis poziva se:

```python
document_id(document, start + index)
```

```python
def document_id(
    document: Document,
    index: int,
) -> str:
    metadata = document.metadata
    record_type = str(
        metadata.get("record_type", "child")
    )

    raw_id = "|".join(
        str(part)
        for part in (
            record_type,
            metadata.get("source_document_id"),
            metadata.get("source"),
            metadata.get("parent_id"),
            metadata.get("section_id"),
            metadata.get("child_chunk_index"),
            metadata.get("cloud_subchunk_index"),
            document.page_content[:200],
            index,
        )
    )

    digest = hashlib.sha1(
        raw_id.encode("utf-8")
    ).hexdigest()[:24]

    return f"{record_type}_{digest}"
```

ID se formira hashiranjem tipa zapisa, izvora, parent i section ID-ja, indeksa chunk-a, dela teksta i indeksa upisa.

## 1.18. Upis u batch-evima

Dokumenti se dele u batch-eve:

```python
for start in range(
    0,
    total,
    self.store_batch_size,
):
    end = min(
        start + self.store_batch_size,
        total,
    )

    batch = prepared_documents[start:end]
```

Za svaki batch formiraju se tri liste:

```python
ids = [
    document_id(
        document,
        start + index,
    )
    for index, document in enumerate(batch)
]

documents = [
    document.page_content
    for document in batch
]

metadatas = [
    clean_metadata(document.metadata)
    for document in batch
]
```

Zatim se poziva stvarni upis:

```python
self.collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)
```

Ovo je trenutak kada podaci napuštaju lokalnu aplikaciju i šalju se Chroma Cloud-u.

## 1.19. Šta se šalje u Chroma Cloud?

Za svaki zapis šalju se:

```text
ID zapisa
tekst zapisa
metapodaci
```

U Python obliku:

```python
{
    "id": "child_35ba91...",
    "document": "Lek: Norvasc\n...",
    "metadata": {
        "record_type": "child",
        "file_name": "norvasc.pdf",
        "medicine_name": "Norvasc",
        "active_substance": "amlodipin",
        "parent_id": "section_...",
        "child_chunk_index": 2,
    },
}
```

Lokalni kod ne šalje ručno izračunate dense i sparse vektore. Chroma Cloud na osnovu teksta izvršava registrovane embedding funkcije.

## 1.20. Šta se dešava posle upisa?

Posle svih batch-eva poziva se:

```python
self.count()
```

što odgovara:

```python
def count(self) -> int:
    return self.collection.count()
```

Na kraju se ispisuju:

```python
print("[INFO] Source pages:", len(pages))
print("[INFO] Parent sections:", len(prepared.parent_documents))
print("[INFO] Child chunks:", len(prepared.child_documents))
print("[INFO] Stored Chroma records:", store.count())
```

---

# 2. Tok pretrage nakon postavljanja pitanja

Kada korisnik postavi pitanje, sistem ne pretražuje direktno PDF fajlove. PDF dokumenti su prethodno obrađeni i zapisani u Chroma Cloud kolekciju. Prilikom pitanja pretražuju se postojeći zapisi, prvenstveno child chunk-ovi.

Osnovni tok je:

```text
pitanje korisnika
    ↓
ui_app.py ili rag_pipeline.py
    ↓
RAGPipeline.answer()
    ↓
ChromaRetriever.retrieve()
    ↓
ChromaVectorStore.query()
    ↓
ChromaVectorStore.search()
    ↓
hybrid_search() ili dense_search()
    ↓
rezultati child chunk-ova
    ↓
expand_results_to_parents()
    ↓
RetrievedDocument objekti
    ↓
formiranje prompt-a
    ↓
Gemini model
```

Primarna pretraga i generisanje odgovora predstavljaju dve odvojene faze:

1. **retrieval faza** – pronalaženje relevantnih zapisa;
2. **generativna faza** – slanje pronađenog konteksta Gemini modelu.

## 2.1. Unos pitanja kroz Streamlit

U `ui_app.py` pitanje se unosi preko:

```python
prompt = st.chat_input(
    "Unesite pitanje o lekovima, interakcijama ili apotekarskoj praksi",
    disabled=not can_ask,
)
```

Ako je pitanje uneto, poziva se:

```python
_submit_question(prompt, settings)
```

```python
def _submit_question(
    prompt: str,
    settings: UISettings,
) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("assistant"):
        with st.spinner(
            "Pretrazujem dokumente i generisem odgovor..."
        ):
            response = _answer_question(
                prompt,
                settings,
            )

        message = _assistant_message(response)
        st.session_state.messages.append(message)
        _render_message_content(
            message,
            show_context=settings.show_context,
        )
```

## 2.2. Pozivanje RAG pipeline-a

```python
def _answer_question(
    question: str,
    settings: UISettings,
) -> RAGResponse:
    rag = _get_rag_pipeline(settings)
    rag.top_k = settings.top_k
    rag.max_context_chars = settings.max_context_chars
    return rag.answer(question)
```

Pipeline se kreira funkcijom `create_rag_pipeline()` iz `rag_pipeline.py`:

```python
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
```

Za samu pretragu je najvažniji `ChromaRetriever`. Gemini se koristi tek ako retrieval vrati rezultate.

## 2.3. `RAGPipeline.answer()`

```python
def answer(self, question: str) -> RAGResponse:
    retrieved_documents = self.retriever.retrieve(
        question,
        k=self.top_k,
    )

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
```

Prvi poziv je:

```python
self.retriever.retrieve(question, k=self.top_k)
```

To je početak stvarne pretrage baze.

## 2.4. `ChromaRetriever.retrieve()`

```python
def retrieve(
    self,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[RetrievedDocument]:
    results = self.vector_store.query(
        query,
        top_k=k,
        candidate_count=self.candidate_pool_size,
        use_hybrid_search=self.use_hybrid_search,
        group_by_source=self.group_by_source,
    )

    return [
        RetrievedDocument(
            text=result["text"],
            metadata=result["metadata"],
            score=result["score"],
            document_id=result.get("id"),
            chroma_score=result.get("chroma_score"),
            distance=result.get("distance"),
            search_type=result.get("search_type"),
        )
        for result in results
    ]
```

`ChromaRetriever` ne računa embedding i ne zna kako su PDF-ovi parsirani. On prosleđuje pitanje `ChromaVectorStore` objektu i standardizuje rezultat.

## 2.5. `query()` i `search()`

`query()` je alias za `search()`:

```python
def query(
    self,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    candidate_count: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    use_hybrid_search: bool | None = None,
    group_by_source: bool | None = None,
    expand_to_parent: bool = True,
) -> list[dict[str, Any]]:
    return self.search(
        query_text,
        top_k=top_k,
        candidate_count=candidate_count,
        use_hybrid_search=use_hybrid_search,
        group_by_source=group_by_source,
        expand_to_parent=expand_to_parent,
    )
```

Stvarna logika je u `search()`:

```python
def search(
    self,
    query_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    candidate_count: int = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    use_hybrid_search: bool | None = None,
    group_by_source: bool | None = None,
    expand_to_parent: bool = True,
) -> list[dict[str, Any]]:
    child_count = self.record_count(CHILD_FILTER)

    if child_count == 0:
        return []

    limit = min(
        max(top_k, candidate_count),
        child_count,
    )

    should_hybrid = (
        self.use_hybrid_search
        if use_hybrid_search is None
        else use_hybrid_search
    )

    should_group = (
        self.group_by_source
        if group_by_source is None
        else group_by_source
    )

    if should_hybrid:
        try:
            results = self.hybrid_search(
                query_text,
                limit,
                should_group,
            )
        except Exception:
            results = self.dense_search(
                query_text,
                limit,
            )
    else:
        results = self.dense_search(
            query_text,
            limit,
        )

    if expand_to_parent:
        results = self.expand_results_to_parents(results)

    return results[:top_k]
```

## 2.6. Provera child zapisa

Primarna pretraga koristi filter:

```python
CHILD_FILTER = {
    "record_type": "child"
}
```

Broj child zapisa proverava se ovako:

```python
def record_count(
    self,
    where: dict[str, Any],
) -> int:
    records = self.collection.get(
        where=where,
        include=["metadatas"],
    )
    return len(records.get("ids", []))
```

Ako nema child zapisa, pretraga se prekida. Parent zapisi se ne koriste kao primarni kandidati.

## 2.7. Određivanje broja kandidata

Broj kandidata računa se ovako:

```python
limit = min(
    max(top_k, candidate_count),
    child_count,
)
```

Ako je:

```text
top_k = 5
candidate_count = 60
child_count = 741
```

onda je:

```text
limit = 60
```

Sistem prvo razmatra do 60 child kandidata, a posle parent proširivanja i deduplikacije vraća najviše 5 rezultata.

## 2.8. Hibridna pretraga

Funkcija `hybrid_search()` formira dense i sparse `Knn` izraze:

```python
def hybrid_search(
    self,
    query_text: str,
    limit: int,
    group_by_source: bool,
) -> list[dict[str, Any]]:
    rank = Rrf(
        ranks=[
            Knn(
                query=query_text,
                limit=limit,
                return_rank=True,
            ),
            Knn(
                query=query_text,
                key=self.sparse_key,
                limit=limit,
                return_rank=True,
            ),
        ],
        weights=[
            self.dense_weight,
            self.sparse_weight,
        ],
        k=60,
    )

    search = (
        Search()
        .where(CHILD_FILTER)
        .rank(rank)
    )
```

Prvi `Knn` koristi dense indeks, a drugi sparse indeks.

Dense tok:

```text
pitanje
    ↓
Qwen dense embedding
    ↓
poređenje sa dense embedding-ima
    ↓
dense rang
```

Sparse tok:

```text
pitanje
    ↓
SPLADE sparse reprezentacija
    ↓
poređenje sa sparse embedding-ima
    ↓
sparse rang
```

## 2.9. Kombinovanje rangova pomoću RRF-a

Dense i sparse rangovi se kombinuju:

```python
rank = Rrf(
    ranks=[dense_rank, sparse_rank],
    weights=[self.dense_weight, self.sparse_weight],
    k=60,
)
```

Dokument koji je visoko rangiran u obe liste dobija bolji kombinovani položaj. Dense komponenta obuhvata semantičku sličnost, a sparse komponenta precizno poklapanje termina.

## 2.10. Grupisanje po izvornom PDF-u

Ako je grupisanje uključeno:

```python
search = search.group_by(
    GroupBy(
        keys=K("source_document_id"),
        aggregate=MinK(
            keys=K.SCORE,
            k=self.group_by_document_limit,
        ),
    )
)
```

Grupisanje po `source_document_id` sprečava da jedan PDF zauzme sve rezultate. Ograničenje se odnosi na child rezultate iz jednog izvora.

## 2.11. Slanje upita Chroma Cloud-u

```python
result = self.collection.search(
    search.limit(limit).select(
        K.ID,
        K.DOCUMENT,
        K.SCORE,
        K.METADATA,
        *SEARCH_METADATA_KEYS,
    )
)
```

Chroma Cloud tada:

1. obrađuje tekst pitanja;
2. koristi dense i sparse embedding funkcije;
3. poredi pitanje sa zapisima;
4. primenjuje child filter;
5. kombinuje rangove pomoću RRF-a;
6. primenjuje grupisanje ako je uključeno;
7. vraća rangirane child rezultate.

## 2.12. Formatiranje hibridnih rezultata

Rezultati se obrađuju funkcijom `format_search_results()`:

```python
def format_search_results(
    results: Any,
    *,
    search_type: str,
) -> list[dict[str, Any]]:
    rows_by_query = (
        results.rows()
        if hasattr(results, "rows")
        else []
    )

    rows = (
        rows_by_query[0]
        if rows_by_query
        else []
    )

    formatted = []

    for row in rows:
        chroma_score = float(
            row_value(
                row,
                "score",
                "#score",
            )
            or 0.0
        )

        formatted.append(
            {
                "id": row_value(row, "id", "#id"),
                "score": -chroma_score,
                "chroma_score": chroma_score,
                "distance": None,
                "text": row_value(
                    row,
                    "document",
                    "#document",
                ) or "",
                "metadata": row_metadata(row),
                "search_type": search_type,
            }
        )

    return formatted
```

Rezultat ima oblik:

```python
{
    "id": "...",
    "score": 0.82,
    "chroma_score": 0.82,
    "distance": None,
    "text": "...",
    "metadata": {...},
    "search_type": "hybrid_rrf_child",
}
```

## 2.13. Dense fallback pretraga

Ako je hibridna pretraga isključena ili ne uspe, koristi se:

```python
def dense_search(
    self,
    query_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    result = self.collection.query(
        query_texts=[query_text],
        n_results=min(
            limit,
            self.record_count(CHILD_FILTER),
        ),
        where=CHILD_FILTER,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return format_query_results(
        result,
        search_type="dense_child",
    )
```

Dense pretraga koristi tekst pitanja, dense embedding funkciju kolekcije i filter nad child zapisima.

## 2.14. Formatiranje dense rezultata

```python
def format_query_results(
    results: dict[str, Any],
    *,
    search_type: str,
) -> list[dict[str, Any]]:
    formatted = []

    rows = zip(
        results.get("ids", [[]])[0],
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    )

    for item_id, text, metadata, distance in rows:
        distance_value = float(distance)

        formatted.append(
            {
                "id": item_id,
                "score": 1.0 - distance_value,
                "chroma_score": distance_value,
                "distance": distance_value,
                "text": text or "",
                "metadata": metadata or {},
                "search_type": search_type,
            }
        )

    return formatted
```

Kod dense pretrage manja udaljenost znači veću sličnost. Aplikacija izračunava score kao `1.0 - distance`.

## 2.15. Proširivanje child rezultata na parent sekcije

Posle dense ili hibridne pretrage poziva se:

```python
results = self.expand_results_to_parents(results)
```

Funkcija prvo izvlači parent ID-jeve:

```python
parent_ids = [
    result["metadata"].get("parent_id")
    for result in results
    if result["metadata"].get("parent_id")
]
```

Zatim se parent zapisi dohvaćaju:

```python
parents = self.parents_by_id(parent_ids)
```

## 2.16. Dohvatanje parent zapisa

```python
def parents_by_id(
    self,
    parent_ids: Iterable[Any],
) -> dict[str, dict[str, Any]]:
    unique_ids = [
        str(parent_id)
        for parent_id in dict.fromkeys(parent_ids)
        if parent_id
    ]

    if not unique_ids:
        return {}

    records = self.collection.get(
        where={
            "$and": [
                PARENT_FILTER,
                {
                    "parent_id": {
                        "$in": unique_ids
                    }
                },
            ]
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    return format_parent_records(records)
```

Filter za parent zapise je:

```python
PARENT_FILTER = {
    "record_type": "parent"
}
```

Ovo nije semantička pretraga. To je direktno dohvaćanje zapisa na osnovu `parent_id` vrednosti.

## 2.17. Ponovno sastavljanje parent sekcija

Ako je parent bio podeljen na više cloud delova, `format_parent_records()` ih spaja prema `cloud_subchunk_index`:

```python
def format_parent_records(
    records: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    parts_by_parent = {}

    for text, metadata in zip(
        records.get("documents", []),
        records.get("metadatas", []),
    ):
        metadata = metadata or {}
        parent_id = metadata.get("parent_id")

        if parent_id:
            part_index = int_or_default(
                metadata.get("cloud_subchunk_index"),
                0,
            )

            parts_by_parent.setdefault(
                str(parent_id),
                [],
            ).append(
                (
                    part_index,
                    text or "",
                    metadata,
                )
            )

    parents = {}

    for parent_id, parts in parts_by_parent.items():
        ordered = sorted(
            parts,
            key=lambda item: item[0],
        )

        parents[parent_id] = {
            "text": "\n".join(
                part[1]
                for part in ordered
                if part[1]
            ).strip(),
            "metadata": dict(
                ordered[0][2]
            ),
        }

    return parents
```

## 2.18. Zamena child teksta parent tekstom

Za svaki pronađeni child rezultat formira se prošireni rezultat:

```python
for result in results:
    child_metadata = result["metadata"]
    parent_id = child_metadata.get("parent_id")
    parent = parents.get(parent_id)

    if not parent_id or parent is None:
        expanded.append(result)
        continue

    if parent_id in seen_parent_ids:
        continue

    seen_parent_ids.add(parent_id)

    metadata = dict(parent["metadata"])
    metadata.update(
        {
            "retrieved_child_id": result.get("id"),
            "retrieved_child_score": result.get("score"),
            "retrieved_child_chunk_index": (
                child_metadata.get("child_chunk_index")
            ),
            "retrieval_record_type": "child",
        }
    )

    expanded.append(
        {
            **result,
            "text": parent["text"],
            "metadata": metadata,
            "parent_id": parent_id,
            "child_text": result["text"],
            "search_type": (
                f"{result['search_type']}_parent"
            ),
        }
    )
```

Nakon toga rezultat sadrži:

```text
text       → parent tekst
child_text → originalni pronađeni child tekst
```

Score ostaje vezan za child rezultat, jer je child bio jedinica koja je pronađena semantičkom ili hibridnom pretragom.

## 2.19. Deduplikacija parent sekcija

Ako više child chunk-ova pripada istom parent-u, parent se prikazuje samo jednom:

```python
seen_parent_ids: set[str] = set()
```

i:

```python
if parent_id in seen_parent_ids:
    continue
```

Zbog ovoga konačan broj parent rezultata može biti manji od `top_k`.

## 2.20. Završno ograničavanje na `top_k`

Nakon parent proširivanja i deduplikacije izvršava se:

```python
return results[:top_k]
```

Redosled je:

```text
candidate_count kandidata
    ↓
dense/hybrid rang
    ↓
grupisanje po PDF-u
    ↓
parent proširivanje
    ↓
parent deduplikacija
    ↓
top_k rezultata
```

## 2.21. Pretvaranje u `RetrievedDocument`

`ChromaRetriever` dobija obične Python rečnike i pretvara ih u:

```python
RetrievedDocument(
    text=result["text"],
    metadata=result["metadata"],
    score=result["score"],
    document_id=result.get("id"),
    chroma_score=result.get("chroma_score"),
    distance=result.get("distance"),
    search_type=result.get("search_type"),
)
```

Definicija:

```python
@dataclass
class RetrievedDocument:
    text: str
    metadata: dict[str, Any]
    score: float
    document_id: str | None = None
    chroma_score: float | None = None
    distance: float | None = None
    search_type: str | None = None
```

Ostatak sistema koristi ovaj standardizovani oblik rezultata.

## 2.22. Formiranje prompt-a

Kada retrieval završi, `RAGPipeline.answer()` poziva:

```python
prompt = build_rag_prompt(
    question=question,
    documents=retrieved_documents,
    max_context_chars=self.max_context_chars,
)
```

`build_rag_prompt()` formira tekst koji sadrži:

```text
Kontekst
Pitanje
Zadatak
```

Za svaki rezultat dodaju se score, izvor, lek, aktivna supstanca, sekcija, tip sadržaja i parent tekst.

Ako kontekst ne stane u ograničenje:

```python
if len(block) > remaining_chars:
    block = block[:remaining_chars].rstrip()
```

## 2.23. Slanje konteksta Gemini modelu

Ako retrieval vrati rezultate, `RAGPipeline` poziva:

```python
answer_text = self.llm.generate(
    prompt=prompt,
    system_instruction=RAG_SYSTEM_INSTRUCTION,
)
```

Ako nema rezultata, Gemini se ne poziva i vraća se poruka:

```text
Nemam dovoljno informacija iz dostupnih dokumenata,
još retrieval nije vratio relevantan kontekst.
```

---

# 3. Primer kompletnog toka pitanja

Za pitanje:

```text
Koja su neželjena dejstva amlodipina?
```

sistem izvršava sledeće:

1. Streamlit prima pitanje preko `st.chat_input()`.
2. `ui_app.py` poziva `_answer_question()`.
3. `_answer_question()` poziva `RAGPipeline.answer()`.
4. `RAGPipeline.answer()` poziva `ChromaRetriever.retrieve()`.
5. Retriever poziva `ChromaVectorStore.query()`.
6. `query()` poziva `search()`.
7. `search()` proverava da li postoje child zapisi.
8. Izračunava se broj kandidata.
9. Pokreće se hibridna ili dense pretraga.
10. Chroma Cloud koristi Qwen dense i SPLADE sparse reprezentacije ako je hibridna pretraga uključena.
11. Dense i sparse rangovi se kombinuju RRF metodom.
12. Pretraga se ograničava na child zapise.
13. Po potrebi se rezultati grupišu po PDF dokumentu.
14. Vraćaju se rangirani child rezultati.
15. Iz child metapodataka čitaju se `parent_id` vrednosti.
16. Parent sekcije se dohvaćaju pomoću `collection.get()`.
17. Parent delovi se po potrebi ponovo sastavljaju.
18. Child rezultat se proširuje parent tekstom.
19. Duplikati parent sekcija se uklanjaju.
20. Rezultati se pretvaraju u `RetrievedDocument` objekte.
21. `rag_prompt.py` formira kontekst i prompt.
22. Gemini dobija pitanje i pronađeni kontekst.
23. `RAGResponse` objedinjuje odgovor, izvore, rezultate i prompt.
24. Streamlit prikazuje odgovor, izvore i opciono kontekst.

---

# 4. Sažetak funkcija i fajlova

| Redosled | Fajl | Funkcija | Uloga |
|---:|---|---|---|
| 1 | `ui_app.py` | `_submit_question()` | Prima pitanje u UI-ju. |
| 2 | `ui_app.py` | `_answer_question()` | Prosleđuje pitanje RAG pipeline-u. |
| 3 | `rag_pipeline.py` | `RAGPipeline.answer()` | Orkestrira retrieval i generisanje. |
| 4 | `retriever.py` | `ChromaRetriever.retrieve()` | Poziva Chroma pretragu. |
| 5 | `vector_store.py` | `query()` | Alias za `search()`. |
| 6 | `vector_store.py` | `search()` | Kontroliše celu pretragu. |
| 7 | `vector_store.py` | `record_count()` | Broji child zapise. |
| 8 | `vector_store.py` | `hybrid_search()` | Dense + sparse + RRF pretraga. |
| 9 | `vector_store.py` | `dense_search()` | Dense fallback pretraga. |
| 10 | `vector_store.py` | `expand_results_to_parents()` | Child rezultate pretvara u parent rezultate. |
| 11 | `vector_store.py` | `parents_by_id()` | Dohvata parent zapise. |
| 12 | `chroma_documents.py` | `format_search_results()` | Formatira hibridne rezultate. |
| 13 | `chroma_documents.py` | `format_query_results()` | Formatira dense rezultate. |
| 14 | `chroma_documents.py` | `format_parent_records()` | Spaja parent poddelove. |
| 15 | `retriever.py` | `RetrievedDocument` | Standardizuje rezultat za RAG sloj. |
| 16 | `rag_prompt.py` | `build_rag_prompt()` | Pravi prompt za Gemini. |
| 17 | `gemini_client.py` | `generate()` | Poziva Gemini model. |

---

# 5. Ključni princip

Primarna pretraga vrši se nad child chunk-ovima, ali se nakon pretrage rezultat proširuje na parent sekciju:

```text
child se koristi za pronalaženje
parent se koristi za kontekst
```

Dense i sparse embedding funkcije koriste se u Chroma Cloud-u. Lokalni Python kod prosleđuje tekst pitanja i dobija rezultate. Konačan rezultat sadrži parent tekst, metapodatke, score pronađenog child-a, `parent_id`, ID child zapisa i tip izvršene pretrage.

Tek nakon završetka retrieval-a rezultat prelazi u prompt i šalje se Gemini modelu.
