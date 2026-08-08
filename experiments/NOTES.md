# Lab Notebook

Decisions and findings, recorded as work proceeds. Source material for the
technical report.

---

## Task 1 — Scaffolding

Decisions:

- Every pipeline setting lives in a YAML config validated by Pydantic at
  startup; nothing is hardcoded. Each run stores its full resolved config and
  git SHA, so any reported number can be reproduced from a committed file.
- Experiment tables (`runs`, `run_results`) are append-only. Experiment history
  is evidence and is never overwritten.
- Ground truth labels are stored as document character spans rather than chunk
  IDs. Chunk IDs change whenever chunking strategy changes, which is the first
  planned experiment; span-based labels survive re-chunking.
- Single store (PostgreSQL + pgvector) rather than Postgres plus a separate
  vector database. A chunk and its embedding are inserted in one transaction,
  metadata filtering and vector search combine in a single query, and there is
  no synchronisation problem between two systems.
- Streamlit rather than a React frontend, and no authentication layer. Both
  demonstrate web engineering rather than retrieval engineering; the time was
  spent on experimentation instead.

---

## Task 2 — Corpus selection

Source: CFPB Credit Card Agreement Database, Q1-2026 collection, pinned to one
quarter so evaluation labels stay valid against fixed text.

Decisions:

- Issuers were selected by US consumer market presence, not by how many
  filings they submit. Archive document count reflects paperwork volume, not
  how many people hold the card.
- That selection also produces the retrieval condition the system must handle.
  Eight confusable issuer families occur naturally: Chase, American Express,
  Wells Fargo, PNC, TD and Comenity each file under two legal entities, and
  Capital One / Capital Bank and Citizens / First-Citizens collide by name.
- Retail store-card issuers are sampled more heavily on purpose. Their
  brand-specific agreements share a common template and differ mainly in brand
  name and fee amounts.
- Content-hash deduplication was necessary: one issuer submitted 165 filings of
  which 77 were byte-identical duplicates.
- Two exclusion rules. Filings over 60 pages are consolidated submissions
  covering many products at once — one issuer filed 609 pages containing every
  product variant, which would have been 18% of the corpus while making
  citations unattributable. Licensing and marketing agreements between issuers
  and co-branding partners are business contracts, not cardholder terms; three
  were found and removed, the last only after parsing revealed its content.
- The corpus is committed as a manifest of SHA-256 hashes plus a fetch script,
  not as PDFs. Reproduction is verified byte-for-byte rather than by filename.

Selected: 247 agreements from 46 issuers.

---

## Task 3 — Parsing

Result: 247 documents, 2,722 pages, 11.5M characters, 744 detected tables,
123 pages recovered by OCR. Product names resolved from document text for 141
filings (57%).

Findings that changed the design:

- Manifest-level `is_scanned` uses a document-wide average and misses filings
  that mix a text layer with scanned images. Detection moved to per-page: 123
  pages required OCR against only 10 documents flagged at manifest level.
- Fee tables mix live text cells with pasted image cells. Cell-level OCR
  recovers values that would otherwise present a fee label with no amount.
- PyMuPDF detects indented lists and multi-column prose as sparse tables, and
  detected regions are excluded from body text — so false positives were
  deleting real prose from affected documents. A fill-ratio threshold (at
  least 50% populated cells) fixed it; Comenity filings fell from 349 detected
  tables to 46.
- Five filings extract as structurally valid but semantically empty text:
  their embedded fonts carry no valid character mapping, so extraction returns
  glyph codes. Detected by common-word density — affected documents score
  0.0–0.8 occurrences per 1,000 characters against a corpus range of 20–45,
  with no documents in between. Recovered by page-level OCR, since the
  rendered page is unaffected. Left undetected, this would have placed 90,000
  characters of noise in the index as retrievable content.
- Largest-first document selection systematically preferred Spanish
  translations, which are longer than their English originals. Every Bank of
  America filing selected was Spanish-only, leaving a top-three issuer
  unusable by English embedding models. Fixed by skipping translations during
  selection so the English filing takes the slot rather than leaving it empty.
- Chase files generic cardmember agreements that name no product, identified
  only by an internal collection code. Six of its seven JPMorgan-entity
  filings were named by joining on collection codes shared with its Chase Bank
  USA filings, confirmed by content overlap.

Known limitations:

- 64 of 247 documents yield no structured tables; those issuers use borderless
  layouts PyMuPDF cannot detect. Inspection confirms the content extracts as
  readable prose in correct reading order.
- OCR-recovered pages carry character-level errors — mangled phone numbers,
  bullets rendered as punctuation. Immaterial for fee and rate questions.
- Seven documents contain no interest-rate language. All are legitimate: four
  American Express charge cards, which have no APR because the balance is due
  in full; two rewards program documents; one statement sample.
- Rewards program terms remain in the corpus. They are consumer-facing and in
  scope, and create realistic retrieval ambiguity worth measuring.

Verification is automated in `scripts/verify_parsing.py`: six structural checks
that block a freeze, and five content checks reported as warnings.