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

## Task 4 — Chunking

Nine chunk sets built and stored concurrently, so strategies are chosen by
measurement rather than inspection.

| set                       | chunks | mean tokens |
|---------------------------|--------|-------------|
| fixed_256_ov0             | 10,272 | 244 |
| fixed_512_ov0             |  5,274 | 467 |
| fixed_512_ov15            |  5,767 | 472 |
| fixed_1024_ov0            |  2,665 | 915 |
| recursive_512_ov0         |  5,487 | 440 |
| recursive_512_ov15        |  5,873 | 442 |
| structure_aware_512_ov0   |  6,517 | 370 |
| structure_aware_512_ov15  |  6,663 | 379 |
| parent_doc_512_ov0        | 10,813 | 446 |

Decisions:

- A single canonical tokenizer produces every chunk set, so a chunk set is
  independent of the embedding model applied to it. Embedding models are then
  compared on identical text rather than on boundaries their own tokenizers
  produced.
- A 256-token set exists because all-MiniLM-L6-v2 accepts only 256 tokens and
  discards the remainder silently. Evaluating it on 512-token chunks would
  throw away half of every chunk with no error raised.
- All strategies split documents into atomic spans and pack them to a budget.
  Detected tables are single atoms; separating a fee value from its row label
  makes the value unusable.
- Chunks are stored as character spans into the document text, so each chunk
  resolves to a physical page and span-based evaluation labels can be matched
  to chunks regardless of the strategy that produced them.

Four defects found by verification, none of which raised an error:

- Structure-aware chunking built prose atoms across whole sections including
  table regions, then appended the tables again. Table text was counted twice,
  filling the budget at double speed: 15,612 chunks averaging 168 tokens
  against a 512 budget, with 490 chunk boundaries falling inside tables.
- Fixed-window chunking produced atoms as large as the chunk budget, leaving
  nothing to carry forward, so overlap silently did nothing — the 15% overlap
  set differed from the zero-overlap set by 4 chunks out of 5,733.
- Heading detection fired on capitalised clause blocks, which are standard in
  credit agreements and wrap into runs of short unpunctuated lines. Sections
  below a minimum length are now merged into their neighbour, which bounds the
  effect of heuristic heading detection regardless of why it misfires.
- Heading detection matched known headings by prefix, so any line beginning
  with a known heading word was accepted before the punctuation and length
  guards ran — prose fragments such as "fees and foreign transaction fees."
  became section boundaries. Exact matching fixed it. Residual false positives
  are mailing addresses, which are title-cased and unpunctuated and cannot be
  distinguished heuristically; they affect section labels only, not chunk
  boundaries.

The verification tool was itself wrong twice before it was right. Postgres
`btrim` with one argument strips spaces but not newlines, so paragraph
separators registered as lost content; and a `lag` window function cannot
order overlapping chunk ranges, so it compared non-adjacent chunks and
reported gaps that did not exist. Coverage is now computed by merging
intervals per document. A verification tool that reports false failures costs
as much as one that misses real ones.


## Task 5 — Evaluation set

76 questions and 105 labels: single fact (20), entity-confusable (21), table
lookup (9), comparison (11), unanswerable (10), ambiguous (5). Fifteen
questions carry no labels by design; 23 carry more than one.

Design:

- Ground truth is stored as character spans into document text rather than
  chunk identifiers. Chunk identifiers change whenever chunking strategy
  changes, which is itself the first planned experiment; span-based labels
  remain valid across all nine chunk sets, and a chunk scores as relevant when
  its character range overlaps a labelled span.
- 51 candidate questions were drafted by an LLM from passages sampled across
  market segments and biased toward text containing monetary amounts. Each was
  verified by hand; 31 were kept. Rejections fell into four groups: questions
  targeting federally mandated billing-rights language that appears in most of
  the corpus and so has hundreds of equally valid sources; quotes too short to
  locate uniquely; cards identified only by internal filing codes, which no
  cardholder would use; and questions whose quote did not answer them.
- The remaining 45 questions were written by hand. The entity-confusable,
  unanswerable, comparison and ambiguous categories all depend on knowing what
  the corpus contains and what it lacks, which an LLM drafting from a single
  passage cannot know.
- The loader rejects any quote that is absent from its document or that appears
  more than once without an explicit flag, and rolls back rather than loading
  partially. This caught 13 defective labels across two loads. An ambiguous
  quote resolves to an arbitrary position, producing a label that points at
  text which does not answer the question — an error that raises no exception
  and would depress every recall figure thereafter.

Entity-confusable questions take three forms, in increasing difficulty:

- Cross-document siblings whose values differ. OpenSky Gold charges a $59
  annual fee; OpenSky Plus charges none. The rest of both fee tables is
  identical.
- Within-document siblings. One Comenity filing states 31.99% for the Saks
  World Mastercard and 35.99% for the Saks store card, so document-level
  metadata filtering cannot help and only chunk-level retrieval can succeed.
- Identical-content siblings. The Venmo Visa and Venmo Visa Signature filings
  are byte-identical, so no amount of content matching distinguishes them and
  only product metadata can.

Unanswerable questions cover three distinct failure modes rather than one:
facts that live in other document types, such as sign-up bonuses; genuine
corpus gaps, such as Chase filings that name no product; and correct refusals
that are not gaps at all — the American Express Platinum Card discloses no APR
because it is a charge card, so an invented figure is wrong in a different way
than a guess at a missing fact.

Extending labels to sibling filings could not be automated by document
similarity. One issuer's template is shared closely enough that two different
products score 0.84 on sampled-passage overlap while a genuine duplicate pair
scores 0.68; the distributions overlap, so no threshold separates them. Five
equivalence groups are enumerated explicitly instead, adding 15 labels. Where
a sibling turned out to be a partial rather than a complete duplicate, the
quote was absent and no label was added — the Saks store-card filing omits the
World Mastercard rate table.

The same shared-template property makes several questions harder than their
category suggests. The minimum interest charge is stated identically across 18
Synchrony filings, so retrieving any of the other 17 returns the correct value
from the wrong document and is scored as a miss. This separates retrieval
quality from answer correctness, which is the reason a citation-based system
is worth building rather than simply asking a model.