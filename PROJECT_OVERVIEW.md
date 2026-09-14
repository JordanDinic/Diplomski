# Diplomski RAG projekat

Ovaj projekat je RAG sistem za podrsku radu u apoteci. Sistem pretrazuje PDF
dokumente o lekovima, interakcijama i dobroj apotekarskoj praksi, a zatim na
osnovu pronadjenog konteksta generise odgovor.

Glavni tok je:

```text
PDF
-> page-level loading
-> prepoznavanje tipa dokumenta
-> prepoznavanje sekcija
-> parent-child chunking
-> Chroma Cloud upload
-> Chroma Cloud hybrid search
-> RAG prompt
-> Gemini odgovor
```

## Fajlovi

```text
src/diplomski/
  data_loader.py              ucitava PDF strane u LangChain Document objekte
  document_types.py           prepoznaje tip PDF dokumenta
  document_sections.py        pravi parent sekcije po tipu dokumenta
  chunking.py                 pravi child chunkove za pretragu
  embedding_pipeline.py       tanak compatibility sloj za pripremu dokumenata
  chroma_documents.py         metadata cleanup i 16 KiB zastita za Chroma upload
  vector_store.py             Chroma Cloud kolekcija, upload i search
  retriever.py                pitanje -> Chroma search -> top-k rezultati
  search_documents.py         CLI za proveru retrieval-a bez LLM-a
  evaluate_retrieval.py       precision@k i recall@k evaluacija retrieval-a
  rag_prompt.py               formatiranje konteksta i izvora
  gemini_client.py            Gemini API klijent
  rag_pipeline.py             kompletan RAG tok
  ui_app.py                   Streamlit UI za lokalni demo
  migrate_to_chroma_cloud.py  migracija PDF-ova ili stare lokalne Chroma baze
  settings.py                 centralna podesavanja
  console.py                  UTF-8 podesavanje za Windows terminal
```

## Loading

`data_loader.py` ucitava samo PDF fajlove. Za svaki neprazan PDF page pravi
jedan LangChain `Document`.

Metadata za svaku stranu sadrzi:

```text
source
source_document_id
folder
file_name
file_type
document_type
medicine_name
active_substance
page
page_number
page_index
total_pages
```

Loader namerno ne radi chunking, embedding, Chroma upload ili LLM poziv.

## Tipovi Dokumenata

`document_types.py` razvrstava PDF-ove u:

```text
medicine_leaflet
practice_guide
interaction_reference
unknown
```

Ovo je vazno jer uputstva za lekove, vodic dobre apotekarske prakse i dokument
o interakcijama nemaju istu strukturu.

## Sekcije

`document_sections.py` pravi parent sekcije:

- `medicine_leaflet`: trazi standardnih 6 sekcija uputstva za lek.
- `practice_guide`: koristi PDF outline/bookmarks kada postoje.
- `interaction_reference`: koristi jednostavna pravila za poznate naslove.
- `unknown`: fallback je jedna parent sekcija po strani.

Svaka parent sekcija dobija:

```text
record_type=parent
content_type=parent_section
section_id
parent_id
section_title
section_path
section_detection
page_numbers
```

## Chunking

`chunking.py` ne mesa dve sekcije u isti chunk.

Jedna parent sekcija moze dati vise child chunkova:

```text
parent sekcija
-> child chunk 0
-> child chunk 1
-> child chunk 2
```

Svaki child chunk dobija retrieval prefiks:

```text
Lek: Norvasc
Aktivna supstanca: amlodipin
Document: norvasc.pdf
Section: 4. Moguca nezeljena dejstva

<tekst chunk-a>
```

Child metadata zadrzava vezu sa parent sekcijom:

```text
record_type=child
content_type=child_chunk
parent_id
section_id
section_title
child_chunk_index
child_chunk_count
```

## Chroma Cloud

`vector_store.py` je namerno sveden na Chroma odgovornosti:

- kreiranje/get Chroma Cloud kolekcije,
- reset kolekcije,
- upload vec pripremljenih `Document` objekata,
- search nad child chunkovima,
- vracanje parent sekcije kao sireg konteksta.

`chroma_documents.py` radi pripremu dokumenata za upload:

- metadata cleanup,
- stabilan `document_id`,
- zastita od Chroma limita od 16 KiB po dokumentu.

Kolekcija koristi:

```text
Dense:  Chroma Cloud Qwen
Sparse: Chroma Cloud Splade
Search: dense + sparse + RRF
```

Upis salje samo:

```text
ids
documents
metadatas
```

Embeddinge generise Chroma Cloud.

## Parent-Child Retrieval

Pretraga se radi nad malim child chunkovima jer su precizniji za embedding.
Kada Chroma pronadje dobar child chunk, `vector_store.py` preko `parent_id`
pronalazi parent sekciju i nju vraca kao kontekst za RAG.

To daje bolji balans:

- child chunk = precizna pretraga,
- parent section = siri kontekst za odgovor.

## Retriever

`retriever.py` je tanak sloj:

```text
pitanje
-> ChromaVectorStore.search()
-> list[RetrievedDocument]
```

Retriever ne zna nista o PDF parsiranju, sekcijama ili chunkingu.

## RAG Pipeline

`rag_pipeline.py` radi samo:

```text
pitanje
-> retriever
-> rag_prompt
-> Gemini
-> odgovor + izvori
```

Ako retrieval ne vrati kontekst, Gemini se ne poziva i korisnik dobija poruku da
nema dovoljno informacija iz dostupnih dokumenata.

## UI Istorija

Uz svaki odgovor u Streamlit sesiji cuvaju se izvori i pronadjeni kontekst.
Oni ostaju povezani sa tim odgovorom pri novom pitanju i promeni podesavanja.
Opcija `Prikazi kontekst` menja samo prikaz; sacuvani kontekst se ne brise i
odgovor se ne generise ponovo. Isto vazi za pitanja pokrenuta dugmadima sa primerima.

`Ocisti razgovor` uklanja poruke, izvore i kontekst iz tekuce sesije. Istorija
nije trajno upisana na disk: nova browser sesija ili restart servera je ne obnavlja.
Odgovori nastali pre ove izmene nemaju sacuvane izvore koje je moguce naknadno prikazati.

Regresione provere su u `tests/test_ui_history.py` i koriste lazan backend,
bez Chroma/Gemini API poziva.

## Workflow Komande

Napravi ili osvezi Chroma Cloud bazu:

```powershell
.\.venv\Scripts\python.exe src\diplomski\vector_store.py --data-dir Literatura --reset
```

Testiraj retrieval bez Gemini modela:

```powershell
.\.venv\Scripts\python.exe src\diplomski\search_documents.py "nezeljena dejstva amlodipina" -k 5
```

Izmeri `precision@k` i `recall@k` nad golden pitanjima:

```powershell
.\.venv\Scripts\python.exe src\diplomski\evaluate_retrieval.py -k 1 3 5 --show-results
```

Pokreni RAG odgovor:

```powershell
.\.venv\Scripts\python.exe src\diplomski\rag_pipeline.py "Koja su nezeljena dejstva amlodipina?" -k 5 --show-context
```

Pokreni lokalni UI:

```powershell
.\.venv\Scripts\streamlit.exe run src\diplomski\ui_app.py
```

Pokreni lokalne testove:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Environment

`.env` treba da sadrzi:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash

CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=...
CHROMA_TENANT=7017286c-bbfb-415f-b058-2b37f136fe70
CHROMA_DATABASE=diplomski
```

`.env` se ne commituje.

## Testovi

Testovi pokrivaju:

- prepoznavanje tipa dokumenta,
- loading PDF strana,
- 6 standardnih sekcija za uputstva o lekovima,
- cirilicna uputstva,
- PDF outline za vodic,
- pravila za interakcije lekova,
- da chunk ne mesa dve sekcije,
- da chunk ima `Document:` i `Section:` prefiks,
- metadata polja `source`, `page`, `document_type`, `section_title`,
- Chroma 16 KiB zastitu,
- stabilan document id,
- parent-child prosirenje rezultata,
- tanak retriever,
- RAG tok sa i bez pronadjenog konteksta.

## Napomena

Sistem je demo za diplomski rad i pomoc u pretrazi dokumenata. Odgovor modela
nije zamena za savet lekara ili farmaceuta.
