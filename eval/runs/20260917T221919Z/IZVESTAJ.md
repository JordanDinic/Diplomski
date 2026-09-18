# Ponovljeno testiranje child-only RAG konfiguracije

Datum: 18.09.2026. po lokalnom vremenu Europe/Budapest.
Oznaka izvrsavanja `20260917T221919Z` je vreme pocetka u UTC.

## Obuhvat i konfiguracija

Pokrenuta komanda:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u eval/audit_project.py --live --rag
```

- 37 lokalnih softverskih testova, provera korpusa i Streamlit AppTest.
- Osnovni gold skup: 15 pitanja u tri konfiguracije, ukupno 45 pretraga.
- Dodatni skup: 4 pitanja u dve konfiguracije, ukupno 8 pretraga.
- Pet stvarnih RAG poziva sa modelom `gemini-3.6-flash`.
- Sve pretrage koriste `expand_to_parent=False`, do pet child rezultata i 60 kandidata.
- Hibridna pretraga: Qwen + SPLADE, RRF tezine 0.7/0.3; grupisana varijanta zadrzava najvise dva child pogotka po PDF-u.
- Budzet RAG konteksta: 14.000 znakova.

Izvrsavanje je zavrseno sa izlaznim kodom 0. Koriscen je `ReadOnlyStore`, bez upisa, resetovanja ili brisanja kolekcije. Nisu menjani izvorni kod, referentna ocekivanja, zavrsni rad ni prethodni rezultati. API pozivi mogu trositi kvotu provajdera.

## Lokalne provere

- Svih 37 testova prolazi; trajanje testova je 10.616 sekundi.
- Korpus: 31 PDF, 339 stranica, 493 logicke parent sekcije i 1.361 child segment.
- Nakon tehnicke podele velikih roditelja: 495 parent zapisa i 1.361 child zapis, ukupno 1.856. Inventar postojece Cloud kolekcije takodje sadrzi 1.856 zapisa.
- Nema child segmenata bez parent veze, obaveznih metapodataka ili prefiksa `Document:` i `Section:` u lokalnoj proveri. Cloud inventar nema orphan parent reference.
- UI provera sa laznim backendom nema prijavljenih izuzetaka; izvori i kontekst ostaju prikazani nakon ponovnog izvrsavanja. Ovo nije vizuelna ni stvarna Cloud UI provera.
- Streamlit i dalje ispisuje upozorenja za `use_container_width` i testno izvrsavanje bez `ScriptRunContext`. Ona nisu oborila testove.

## Osnovni skup

Makroproseci za 15 pitanja, na prvih pet rezultata:

| Konfiguracija | Precision@5 | Recall@5 | Hit@5 | MRR@5 | Prosek vremena | Medijana vremena |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid sa grupisanjem | 0.3467 | 0.8667 | 0.8667 | 0.8333 | 2.702 s | 2.295 s |
| Hybrid bez grupisanja | 0.6133 | 0.9000 | 0.9333 | 0.8500 | 2.574 s | 2.501 s |
| Dense bez grupisanja | 0.4800 | 0.8333 | 0.8667 | 0.7889 | 25.153 s | 10.122 s |

Najsporiji dense upit trajao je 114.993 sekunde. Ovo je merenje jednog izvrsavanja udaljenih servisa, ne izolovan benchmark embedding modela. U hibridnim rezultatima nema dense fallback-a.

### Nepotpuna poklapanja sa gold skupom

- Hybrid sa grupisanjem: `guide_dispensing` i `guide_storage` imaju Recall@5 = 0.
- Hybrid bez grupisanja: `guide_dispensing` ima Recall@5 = 0, a `amlodipin_adverse_effects` ima 0.5.
- Dense bez grupisanja: `guide_dispensing` i `guide_storage` imaju Recall@5 = 0, a `amlodipin_adverse_effects` ima 0.5.

Za `amlodipin_adverse_effects`, hybrid bez grupisanja vraca pet child pogodaka iste sekcije iz `amlopin-combo.pdf`. Drugi ocekivani izvor, `norvasc.pdf`, nije u top 5. Visok broj relevantnih child pogodaka ne garantuje pokrivenost svih ocekivanih izvora.

Gold ocekivanja uglavnom proveravaju naziv PDF-a i sekciju, ne konkretan dokazni pasus. Zato vise child segmenata iste sekcije moze povecati Precision@5 bez dodatnog dokaza za odgovor. Kod pitanja o vodicu formalni promasaj naslova ne dokazuje sam po sebi da su svi vraceni pasusi beskorisni; potrebna je sadrzajna provera anotacija i rezultata.

## Dodatna pitanja

| Konfiguracija | Precision@5 | Recall@5 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid sa grupisanjem | 0.1000 | 0.2500 | 0.2500 | 0.2500 |
| Dense bez grupisanja | 0.4500 | 0.7500 | 0.7500 | 0.7500 |

| Slucaj | Hybrid Hit@5 | Dense Hit@5 |
|---|---:|---:|
| `cyrillic_norvasc` | 0 | 1 |
| `english_xanax_missed_dose` | 0 | 1 |
| `norvasc_grapefruit` | 1 | 1 |
| `xanax_driving` | 0 | 0 |

Poslednja tri slucaja ukljucuju uslove nad tekstom (`must_contain`), a ne samo nad naslovom sekcije. Rezultati otkrivaju da pronalazenje parent sekcije i pronalazenje konkretnog child dokaza nisu ista provera. Ovaj mali skup ne predstavlja dovoljnu osnovu za opstu ocenu kvaliteta. Poredjenje ovih varijanti istovremeno menja rangiranje i grupisanje, pa ne izoluje njihov pojedinacni uticaj.

## Kontekst i Gemini

- Sve 53 evaluacione pretrage vracaju samo child zapise, ukupno 244 rezultata.
- U proveri normalizovanog teksta nijedno telo vracenog child segmenta nije izostalo iz formiranog konteksta.
- Svih pet Gemini poziva je zavrseno i svaki koristi samo child rezultate. Svih 17 vracenih tekstova u tim RAG pozivima nalazi se u odgovarajucem promptu.
- Za uputstvo, vodic i interakcije generisani su neprazni odgovori uz navodjenje izvora.
- Za izmisljeni lek model navodi da nema informacija, a za fudbalsko pitanje navodi da kontekst nije dovoljan za odgovor.
- Ukupna vremena pojedinacnih RAG poziva: 14.314 s, 16.170 s, 10.023 s, 7.055 s i 7.136 s; prosek 10.939 s.

Ovo su funkcionalne provere i ogranicen pregled ponasanja, ne strucna ocena tacnosti, potpunosti ili medicinske bezbednosti svakog odgovora. Nisu sprovedeni klinicka validacija ni nezavisno ocenjivanje farmaceuta.

## Poredjenje sa arhiviranim parent rezultatima

Arhiva `20260909T100414Z` koristi parent prosirivanje. SHA-256 otisci lokalnih PDF-ova, gold skupa i dodatnih pitanja nisu promenjeni. Broj i raspodela Cloud zapisa se poklapaju; to samo po sebi nije dokaz identicnog sadrzaja svakog Cloud zapisa ili jednakih uslova servisa.

| Konfiguracija | Stari parent Recall@5 | Novi child Recall@5 | Stari parent MRR@5 | Novi child MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid sa grupisanjem | 0.8667 | 0.8667 | 0.8333 | 0.8333 |
| Hybrid bez grupisanja | 0.9333 | 0.9000 | 0.8667 | 0.8500 |
| Dense bez grupisanja | 0.9333 | 0.8333 | 0.8056 | 0.7889 |

Na dodatnom skupu stari Hit@5 bio je 0.75 za grupisani hybrid i 1.00 za dense; sada je 0.25 i 0.75. Ovo nije kontrolisan istovremeni A/B eksperiment. Promenjena je jedinica rezultata i uklonjena deduplikacija po parent sekciji; vremena i odgovori udaljenih servisa takodje nisu deterministicki.

## Preostala ogranicenja

`contract_probes.json` ponovo pokazuje zavisnost ID-a od indeksa upisa i presecene reci u sintetickom testu malih chunkova. Sinteticki test sa tekstom preko 14.000 znakova takodje pokazuje moguce izostavljanje dokaza i neusaglasenost liste izvora sa promptom. Taj test ne predstavlja stvarni child kontekst u ovom izvrsavanju, u kojem nije uoceno odsecanje vracenog child teksta. Ove dijagnosticke provere nisu neuspehi u skupu od 37 automatizovanih testova.

Zakljucak: child-only tok je funkcionalan, ali sam prelazak na child rezultate ne garantuje bolju pretragu ni potpuniji odgovor. Sledeci korak treba da bude provera dokaznih pasusa, gold anotacija i pokrivenosti izvora na pitanjima koja nisu zadovoljila ocekivanja. U ovom izvrsavanju nisu menjani ni kod ni ocekivanja da bi se popravili skorovi.

## Sirovi dokazi

- `unit_tests.txt`, `unit_tests.json`: softverski testovi.
- `corpus.json`, `cloud_inventory.json`: lokalni korpus i inventar kolekcije.
- `hybrid_grouped.json`, `hybrid_ungrouped.json`, `dense_ungrouped.json`: osnovni skup, ukljucujuci metrike za k = 1, 3, 5 i vracene tekstove.
- `challenges_hybrid_grouped.json`, `challenges_dense_ungrouped.json`: dodatna pitanja.
- `rag_smoke.json`: stvarni promptovi, izvori, odgovori i trajanja Gemini poziva.
- `ui_smoke.json`, `contract_probes.json`: UI i dodatne dijagnosticke provere.
- `manifest.json`: vreme pocetka, verzije biblioteka, podesavanja i otisci fajlova.
