# Diplomski RAG projekat

Ovaj projekat je lokalni RAG sistem za pretragu i odgovaranje na pitanja na osnovu PDF dokumenata o lekovima.

Glavni tok je:

```text
PDF dokumenti
-> Unstructured partition_pdf()
-> Unstructured elementi
-> obogacivanje tabela kontekstom
-> chunk_by_title()
-> LangChain Document objekti
-> SentenceTransformer embedding
-> ChromaDB vector store
-> retrieval + reranking
-> RAG prompt
-> Gemini Flash odgovor
```

Sistem je podeljen po fazama da bi ingestion, embedding, pretraga i generisanje odgovora mogli da se testiraju nezavisno.

## Struktura projekta

```text
src/diplomski/
  data_loader.py          PDF loading pomocu Unstructured
  embedding_pipeline.py   priprema elemenata, chunking i embedding
  vector_store.py         cuvanje embeddinga u ChromaDB
  retriever.py            retrieval iz ChromaDB + lexical reranking
  search_documents.py     CLI za pronalazenje top-k dokumenata
  rag_prompt.py           pravljenje RAG prompta i izvora
  gemini_client.py        Gemini API klijent
  rag_pipeline.py         kompletan RAG tok: query -> answer
  settings.py             centralna podesavanja
  console.py              UTF-8 podesavanje za Windows terminal
```

## 1. PDF loading

Fajl: `src/diplomski/data_loader.py`

Loader koristi `unstructured.partition.pdf.partition_pdf()` i vraca originalne Unstructured elemente, a ne LangChain `Document` objekte.

Najvaznije funkcije:

- `load_pdf(file_path)` ucitava jedan PDF.
- `load_all_elements(data_dir)` prolazi kroz folder i podfoldere i ucitava sve PDF fajlove.
- `_add_file_metadata()` dodaje informacije potrebne za traceability.

Za svaki element se dodaje metadata kao:

```python
metadata.source
metadata.folder
metadata.file_name
metadata.file_type
metadata.element_index
metadata.category
metadata.content_type
```

Ova faza namerno ne radi chunking i ne pravi embeddinge. Ona samo izvlaci sirove PDF elemente.

## 2. Embedding pipeline

Fajl: `src/diplomski/embedding_pipeline.py`

Ovaj modul pretvara Unstructured elemente u podatke spremne za vector store.

Tok rada:

```text
elements
-> prepare_elements()
-> chunk_elements()
-> chunks_to_documents()
-> embed_documents()
```

### Tabele

Pre chunkovanja, svaka tabela dobija prethodna 2 relevantna tekstualna elementa iz istog PDF-a.

Kontekst se dodaje direktno u `Table.text`, na primer:

```text
Context:
Tekst koji prethodi tabeli.

Jos jedan relevantan tekstualni element.

Table:
| Kolona 1 | Kolona 2 |
| --- | --- |
| ... | ... |
```

Ovo je vazno jer kontekst onda ulazi u `Document.page_content`, pa se embeduje zajedno sa tabelom.

Elementi kao `Header`, `Footer`, `PageBreak`, `Image`, `Picture`, `Figure`, `Table` i `TableChunk` ne ulaze u table context.

Ako tabela ima `metadata.text_as_html`, originalni HTML ostaje sacuvan u metadata, a sadrzaj tabele se za embedding pretvara u Markdown preko `markdownify`.

### Chunking

Chunking koristi Unstructured `chunk_by_title()`:

```python
chunk_by_title(
    prepared_elements,
    max_characters=...,
    new_after_n_chars=...,
    overlap=...,
    include_orig_elements=True,
    skip_table_chunking=True,
    isolate_table=True,
)
```

Bitna podesavanja:

- `skip_table_chunking=True`: tabela se ne deli na manje delove.
- `isolate_table=True`: tabela ostaje poseban chunk.
- `include_orig_elements=True`: metadata cuva vezu sa originalnim elementima.

### Embedding model

Podrazumevani embedding model je:

```text
Qwen/Qwen3-Embedding-0.6B
```

Model se ucitava lenjo, tek kada prvi put treba da napravi embedding.

Device se podesava preko:

```text
auto
cpu
cuda
cuda:0
```

Ako je `device="auto"`, kod koristi CUDA ako je dostupna, inace CPU.

## 3. ChromaDB vector store

Fajl: `src/diplomski/vector_store.py`

`ChromaVectorStore` cuva dokumente, metadata i embeddinge u lokalnu ChromaDB bazu.

Podrazumevani folder baze je:

```text
chroma_db
```

Podrazumevana kolekcija je:

```text
diplomski_rag
```

Najvaznije metode:

- `build_from_elements(elements)` pokrece embedding pipeline i puni ChromaDB.
- `build_from_documents(documents)` embeduje vec pripremljene LangChain dokumente.
- `add_documents(documents, embeddings)` dodaje dokumente u kolekciju.
- `query(query_text, top_k)` vraca slicne dokumente iz ChromaDB.
- `reset()` brise i ponovo kreira kolekciju.
- `count()` vraca broj sacuvanih vektora.

Metadata se pre cuvanja prilagodjava ChromaDB ogranicenjima. Slozenije vrednosti, kao liste i dict objekti, serijalizuju se u JSON string.

## 4. Retrieval

Fajl: `src/diplomski/retriever.py`

`ChromaRetriever` koristi ChromaDB za vektorsku pretragu, ali zatim radi dodatni lexical reranking.

To znaci:

1. Prvo se iz ChromaDB uzme siri skup kandidata.
2. Kandidati se dodatno boduju na osnovu poklapanja termina iz pitanja.
3. Vraca se finalnih top-k dokumenata.

Ovo pomaze kod pitanja gde je naziv leka jako vazan, na primer:

```text
Koja su nezeljena dejstva amlodipina?
```

Bez lexical rerankinga, slicni opsti medicinski tekstovi mogu nekad da se rangiraju previsoko. Reranking pojacava dokumente gde se u tekstu ili metadata pominje trazeni lek.

## 5. Search CLI

Fajl: `src/diplomski/search_documents.py`

Ovo je alat za proveru retrieval-a bez Gemini modela.

Primer:

```powershell
.\.venv\Scripts\python.exe src\diplomski\search_documents.py "Koja su nezeljena dejstva amlodipina?" --device cuda -k 5
```

Koristi se kada zelis da vidis koji dokumenti se pronalaze pre generisanja finalnog odgovora.

Ispisuje:

- score
- vector_score
- lexical_score
- distance
- source
- file_name
- page
- content_type
- preview teksta

## 6. RAG prompt

Fajl: `src/diplomski/rag_prompt.py`

Ovaj modul pravi tekst koji se salje LLM-u.

Najvaznije funkcije:

- `build_rag_prompt()` pravi kompletan prompt od pitanja i pronadjenih dokumenata.
- `format_context()` formatira retrieval rezultate u ogranicen kontekst.
- `extract_sources()` pravi listu izvora za prikaz korisniku.
- `format_source_label()` pravi citljiv naziv izvora.

Sistemska instrukcija trazi da model:

- odgovara na srpskom,
- koristi samo prosledjeni kontekst,
- ne izmislja medicinske informacije,
- jasno kaze kada nema dovoljno informacija,
- navede izvore kada je korisno.

## 7. Gemini client

Fajl: `src/diplomski/gemini_client.py`

`GeminiFlashClient` je tanak wrapper oko Google Gen AI SDK-a.

API key se cita iz `.env`:

```text
GEMINI_API_KEY=...
```

ili:

```text
GOOGLE_API_KEY=...
```

Model se podesava preko:

```text
GEMINI_MODEL=gemini-3.6-flash
```

Ako `.env` nije popunjen validnim kljucem, RAG pipeline ne moze da generise odgovor, ali retrieval i dalje moze da se testira preko `search_documents.py`.

## 8. Kompletan RAG pipeline

Fajl: `src/diplomski/rag_pipeline.py`

Ovo je glavni end-to-end tok:

```text
pitanje
-> ChromaRetriever.retrieve()
-> build_rag_prompt()
-> GeminiFlashClient.generate()
-> odgovor + izvori
```

Primer:

```powershell
.\.venv\Scripts\python.exe src\diplomski\rag_pipeline.py "Koja su nezeljena dejstva amlodipina?" --embedding-device cuda -k 5
```

Za prikaz konteksta koji je poslat Gemini modelu:

```powershell
.\.venv\Scripts\python.exe src\diplomski\rag_pipeline.py "Koja su nezeljena dejstva amlodipina?" --embedding-device cuda -k 5 --show-context
```

## 9. Tipican workflow

### 1. Instalacija dependency-ja

Ako koristis `uv`:

```powershell
uv sync
```

Ili preko pip-a:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Podesavanje Gemini kljuca

Kopiraj `.env.example` u `.env` i unesi svoj API key:

```text
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.6-flash
```

`.env` ne treba commitovati.

### 3. Izgradnja ChromaDB baze

```powershell
.\.venv\Scripts\python.exe src\diplomski\vector_store.py --data-dir Literatura\Lekovi --device cuda
```

Ako zelis CPU:

```powershell
.\.venv\Scripts\python.exe src\diplomski\vector_store.py --data-dir Literatura\Lekovi --device cpu
```

### 4. Test retrieval-a

```powershell
.\.venv\Scripts\python.exe src\diplomski\search_documents.py "Koja su nezeljena dejstva amlodipina?" --device cuda -k 5
```

### 5. Pokretanje RAG odgovora

```powershell
.\.venv\Scripts\python.exe src\diplomski\rag_pipeline.py "Koja su nezeljena dejstva amlodipina?" --embedding-device cuda -k 5
```

## 10. Glavna podesavanja

Fajl: `src/diplomski/settings.py`

```python
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "diplomski_rag"

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_EMBEDDING_BATCH_SIZE = 1
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 256

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 12000

DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE = 60
DEFAULT_MAX_CONTEXT_CHARS = 14000
```

Za slabiju graficku karticu ili CPU, najbitnije vrednosti su:

- `DEFAULT_EMBEDDING_BATCH_SIZE`
- `DEFAULT_EMBEDDING_MAX_SEQ_LENGTH`
- `DEFAULT_EMBEDDING_DEVICE`

## 11. Zasto je pipeline ovako podeljen

Loading je odvojen od chunkinga zato sto su to razlicite odgovornosti.

`data_loader.py` samo cita PDF i pravi Unstructured elemente. To olaksava debugging PDF ekstrakcije.

`embedding_pipeline.py` odlucuje kako se elementi pripremaju za RAG: kako se tabele obogacuju kontekstom, kako se chunkuju tekstovi i kako nastaju LangChain `Document` objekti.

`vector_store.py` cuva embeddinge u ChromaDB i nema logiku o PDF parsiranju.

`retriever.py` se bavi pronalazenjem relevantnog konteksta.

`rag_pipeline.py` spaja retrieval i LLM odgovor.

Ova podela znaci da mozes posebno da testiras:

- da li PDF parsing radi,
- da li chunkovi izgledaju dobro,
- da li ChromaDB vraca dobre rezultate,
- da li Gemini daje odgovor na osnovu dobrog konteksta.

## 12. Napomene

Ovaj sistem pomaze u pretrazi dokumenata o lekovima, ali odgovor modela nije zamena za savet lekara ili farmaceuta.

Kod treba posmatrati kao RAG demo za diplomski rad: ima loading, embedding, vector store, retrieval, prompt construction i LLM odgovor, ali u ozbiljnoj produkciji bi jos trebalo dodati evaluaciju, testove nad poznatim pitanjima, bolji monitoring i jasnije citiranje izvora po recenicama.
