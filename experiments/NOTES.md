# Lab Notebook

Decisions and findings, recorded as work proceeds. Source material for the
technical report.

---

The experiment order has been revised three times by measurement rather than
assumption: metadata filtering demoted after error analysis showed only 6 of 26
failures were wrong-document; hybrid search demoted after complementarity
analysis showed a 5-point ceiling; reranking promoted after a depth probe
showed an 18-point ceiling. Each revision came from decomposing an aggregate
score into diagnostics, not from a better aggregate score.

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


## Task 6 — Evaluation harness and lexical baseline

A single command now takes a configuration, retrieves over the evaluation set,
scores the result and writes a permanent record. BM25 was implemented first
because keyword retrieval needs no model, so the harness could be validated end
to end on real data before any embeddings existed — and the lexical baseline
that every later result is measured against arrived with it.

Design:

- Labelled character spans are resolved to chunks by range overlap: a chunk is
  relevant when it lies in a labelled document and their character ranges
  intersect. One evaluation set therefore scores all nine chunk sets without
  modification.
- Hit rate and recall are both reported because they answer different
  questions. A single-fact question is satisfied by any one relevant passage
  reaching the context; a comparison question requires every side. Reporting
  either alone would systematically flatter or penalise whole categories.
- Precision@k is computed but not used to choose between configurations. With
  roughly two relevant chunks, precision@5 cannot exceed 0.4 regardless of
  ranking quality, and observed values top out at 0.157.
- Document-level hit rate is recorded separately as a diagnostic. It separates
  retrieval of the wrong issuer from retrieval of the right document at the
  wrong passage — two failures with different fixes.
- The 15 unanswerable and ambiguous questions are excluded from retrieval
  scoring rather than scored as zero; they are evaluated in generation.
  Counting them as failures would depress every aggregate by a fifth.
- Every aggregate carries a bootstrap 95% confidence interval over 1,000
  resamples of the question set.
- Metrics are unit-tested against a hand-computed example. A wrong metric does
  not raise an error; it returns a plausible number and invalidates everything
  built on it.
- Runs store their resolved configuration and git commit and are never
  overwritten. Command-line overrides are folded into the stored configuration,
  so the record describes what ran rather than what the file said.

Lexical baseline, 61 scored questions:

| chunk set                 | hit@5 | hit@10 | recall@5 | MRR   | doc hit@5 |
|---------------------------|-------|--------|----------|-------|-----------|
| fixed_1024_ov0            | 0.639 | 0.770  | 0.549    | 0.486 | 0.869 |
| fixed_512_ov0             | 0.574 | 0.623  | 0.492    | 0.388 | 0.902 |
| fixed_512_ov15            | 0.541 | 0.639  | 0.451    | 0.321 | 0.918 |
| recursive_512_ov15        | 0.492 | 0.623  | 0.410    | 0.290 | 0.902 |
| recursive_512_ov0         | 0.459 | 0.492  | 0.373    | 0.271 | 0.869 |
| structure_aware_512_ov0   | 0.426 | 0.525  | 0.353    | 0.277 | 0.852 |
| structure_aware_512_ov15  | 0.426 | 0.525  | 0.339    | 0.282 | 0.836 |
| parent_doc_512_ov0        | 0.426 | 0.541  | 0.353    | 0.280 | 0.836 |
| fixed_256_ov0             | 0.213 | 0.295  | 0.193    | 0.161 | 0.852 |

Findings:

- **Comparing chunk sizes at fixed k was misleading.** 1024-token chunks
  appeared to beat 512 by 6.5 points, but five of them carry twice the context.
  At equal token budget the two are indistinguishable: 1024 at k=5 scores 0.639
  against 512 at k=10 scoring 0.623, both around 5,100 tokens of context.
- **256-token chunks are genuinely worse, not merely under-budgeted.** At equal
  budget of roughly 2,560 tokens, 512 at k=5 scores 0.574 against 256 at k=10
  scoring 0.295. This constrains the embedding comparison directly:
  all-MiniLM-L6-v2 accepts only 256 tokens, so it is confined to the one chunk
  configuration that demonstrably underperforms.
- **Document identification is easy; passage identification is not.** Document
  hit rate sits between 0.84 and 0.92 across every configuration while chunk
  hit rate ranges from 0.21 to 0.64. Of 26 failures at k=5 on the best set, 20
  retrieved the correct agreement and the wrong passage from it; only 6
  retrieved the wrong document.
- **Every within-document sibling question failed.** All three Saks questions,
  both myAcademy questions, and the Target and OpenSky questions where one
  filing covers two products. Those questions were written so that
  document-level information could not solve them, and it did not.
- **Only the 256-token set is clearly separated statistically.** The remaining
  eight configurations have heavily overlapping confidence intervals; the
  widest gap, between fixed_1024 and the structure-aware group, is marginal.
  No chunking strategy can yet be declared better than another.

Consequence for the planned experiments: metadata filtering can address at most
6 of 26 failures, while reranking targets the remaining 20 directly, since a
cross-encoder scores query and passage together and that is precisely the
"which paragraph within this document" problem. Reranking moves earlier in the
ladder and metadata filtering later.

Methodological note for later comparisons: independent confidence intervals are
a conservative test. Because every configuration is evaluated on the same
questions, paired testing — McNemar's test on per-question outcomes — will
detect smaller real differences and should be used in the retrieval
experiments.

## Task 7 — Embeddings and dense retrieval

Three open-weight embedding models compared on identical chunk sets using one
canonical tokenizer, so model quality is the only variable. Exact search
throughout; no approximate index, since its recall cost has not yet been
measured.

| retriever                    | hit@5 | doc hit@5 | entity-confusable |
|------------------------------|-------|-----------|-------------------|
| bm25, 512-token chunks       | 0.574 | 0.902     | 0.524 |
| bge-small (33M, 384 dims)    | 0.541 | 0.836     | 0.571 |
| bge-base (109M, 768 dims)    | 0.492 | 0.770     | 0.524 |
| minilm on 512-token chunks   | 0.410 | 0.754     | 0.333 |
| minilm on 256-token chunks   | 0.148 | 0.705     | 0.143 |

Findings:

- **Keyword search beat every dense model.** The corpus is near-identical legal
  documents where questions turn on exact product names and specific amounts —
  precisely what exact-token matching is for, and what semantic similarity
  blurs. Confidence intervals overlap, so the difference is not statistically
  decisive, but the direction is consistent across every metric.
- **Model capacity did not help.** bge-base has three times the parameters and
  twice the dimensions of bge-small, and scored worse on every measure,
  including a 7-point drop in document identification.
- **Truncation outperformed fitting the model to the chunk.** MiniLM reads 256
  tokens. On 256-token chunks it scored 0.148; on 512-token chunks, where it
  sees only the first half, 0.410. Ranking on the first half while delivering
  the whole chunk to the context beats halving the chunk size, because
  256-token granularity performs badly for both retrievers. Counterintuitive,
  and only visible because both arms were run.
- **The BGE query prefix had a small effect**: identical hit rate, MRR down 8%
  without it (0.373 to 0.344). Real but smaller than the model card implies.
- **Dense rescued three questions, all of one shape** — "what number do I call"
  against text reading "Customer Care", where the question and the answer share
  no vocabulary. That is the textbook case for embeddings, and on this corpus
  it is worth three questions.

Headroom analysis, which reordered the remaining experiments:

- **Hybrid retrieval has a 5-point ceiling.** An oracle choosing the better of
  BM25 and dense per question would reach 0.623 against BM25's 0.574. Only 3 of
  61 questions are found by dense and missed by BM25.
- **Reranking has an 18-point ceiling.** 46 of 61 answers are already retrieved
  by depth 50 but ranked below the cut; at k=5 only 35 are. Reranking exists to
  fix exactly that, so it moves ahead of both hybrid search and metadata
  filtering in the experiment order.
- **Hybrid candidate generation adds little even at depth.** BM25 alone reaches
  0.754 at k=50; the union of BM25 and dense reaches 0.787 — two questions. The
  planned architecture is therefore single-stage lexical retrieval followed by
  reranking, not two retrievers fused.
- **13 of 61 questions (21%) are missed by every configuration at depth 50** and
  cannot be recovered by reranking. They are carried into error analysis.

Methodological note: MRR depends on retrieval depth, since a relevant chunk at
rank 12 contributes nothing at k=5 and 1/12 at k=50. MRR is comparable only
across runs retrieving to the same depth.

Storage: pgvector columns are fixed-width, so models are separated by
dimensionality into embeddings_384 and embeddings_768, with the model key and
prefix scheme stored per row. Several models and the prefix ablation therefore
coexist over the same chunks without re-indexing. Vectors are normalised to
unit length so cosine distance, inner product and Euclidean distance rank
identically.

## Task 8 — Vector indexing

Approximate nearest-neighbour search was benchmarked against exact search
rather than adopted by default.

| method       | recall vs exact | median | p95    |
|--------------|-----------------|--------|--------|
| exact scan   | 1.000           | 13.6 ms| 15.0 ms|
| hnsw ef=10   | 0.786           |  0.9 ms|  2.9 ms|
| hnsw ef=20   | 0.900           |  1.0 ms|  1.7 ms|
| hnsw ef=40   | 0.951           |  1.1 ms|  1.7 ms|
| hnsw ef=100  | 0.987           |  1.5 ms|  2.0 ms|
| hnsw ef=200  | 0.987           |  2.0 ms|  2.9 ms|

- Partial indexes are required, one per model and prefix scheme. Queries always
  filter on those columns, and an unfiltered ANN index combined with a WHERE
  clause post-filters — returning neighbours across all models, discarding most
  of them, and yielding fewer than k results.
- HNSW is nine to twelve times faster than exact scan and reaches 0.987 recall
  at ef=100. Recall saturates there; ef=200 costs 33% more time for no gain.
- **The index was not retained.** It saves roughly twelve milliseconds on a
  request where generation takes one to five seconds — under one percent of
  end-to-end latency — while introducing a recall loss that would make recorded
  evaluation runs incomparable with later ones. Approximate search becomes
  worthwhile two to three orders of magnitude higher in corpus size.
- Part of the 13.6 ms exact-search cost is join and filter overhead rather than
  distance computation, so the achievable saving is smaller than the raw
  comparison implies.


  ## Task 9 — Reranking and retrieval experiments

Final configuration: lexical retrieval over metadata-augmented chunk text to a
pool of 50, reranked by bge-reranker-base with the same metadata prepended,
returning 5. Chunk set: fixed 512 tokens, no overlap.

| configuration                        | hit@5 | hit@50 | MRR   | doc hit@5 | entity-conf |
|--------------------------------------|-------|--------|-------|-----------|-------------|
| baseline: bm25, plain index          | 0.574 | 0.754  | 0.388 | 0.902     | 0.524 |
| + ms-marco reranker, pool 50         | 0.492 | 0.754  | 0.352 | 0.803     | 0.571 |
| + ms-marco reranker, pool 20         | 0.492 |   —    | 0.361 | 0.820     | 0.524 |
| + bge reranker, pool 50              | 0.623 | 0.754  | 0.413 | 0.836     | 0.619 |
| + bge reranker, pool 100             | 0.607 | 0.787  | 0.418 | 0.803     | 0.619 |
| + metadata in reranker input         | 0.639 | 0.754  | 0.493 | 0.902     | 0.667 |
| + metadata in indexed text (final)   | 0.787 | 0.967  | 0.606 | 0.967     | 0.762 |

Final per-category hit@5: single fact 0.850, table lookup 0.889,
entity-confusable 0.762, comparison 0.636.

Chunking, re-tested under reranking: fixed 0.623, structure-aware 0.508,
recursive 0.443 — the same ordering as under plain keyword search.

Findings:

- **Supplying document identity was the largest lever, and it applied twice.**
  Many chunks are bare table fragments such as "| Annual Fee | $89 |" that name
  no product, so neither retriever nor reranker can connect them to a question
  naming a card. Chunking had removed context the document itself provided.
  Prepending product, issuer and section to the reranker's input raised hit@5
  from 0.623 to 0.639 and restored document accuracy from 0.836 to 0.902.
  Applying the same augmentation to the indexed text raised hit@5 to 0.787 and
  hit@50 from 0.754 to 0.967 — improving the candidate pool itself, not merely
  its ordering. Single-fact questions gained 35 points against baseline and
  table lookups 33.
- **The reranker model mattered more than any other single component choice.**
  An identical pipeline swung 13 points between two cross-encoders:
  ms-marco-MiniLM lost 8 points, bge-reranker-base gained 5. Pool size made no
  difference for ms-marco, so it is systematically wrong rather than noisy —
  consistent with a model trained on conversational web passages misjudging
  Markdown fee tables. Under ms-marco, table-lookup and single-fact accuracy
  fell while entity-confusable rose, the signature of a reranker that
  downranks tabular text.
- **Larger candidate pools did not help.** Pool 100 scored below pool 50: more
  candidates give the reranker more opportunities to promote the wrong one.
- **A headroom estimate is conditional on the retriever that produced it.** The
  0.754 ceiling measured in Task 7 was a property of the plain BM25 pool, not
  of the corpus. Changing what BM25 indexes moved it to 0.967.
- **Concentration traded against diversity.** Because every chunk in a document
  carries the same header, the augmented index draws more chunks from the
  highest-scoring document into the pool. Every single-document category
  improved; comparison questions, which require passages from two documents,
  fell from 0.818 to 0.636 — two of eleven. A measured cost, not a defect.
- **Chunking results were stable across ranking mechanisms.** Fixed-size
  chunking beat recursive and structure-aware both with and without reranking.
  Agreement across two unrelated ranking methods is stronger evidence than
  either result alone.

Statistical treatment:

- Two paired tests are reported because they measure different things.
  McNemar's test asks whether a relevant chunk reached the top five, which is
  binary and registers nothing when an answer moves from rank 9 to rank 6.
  Wilcoxon signed-rank on per-question reciprocal rank uses the magnitude of
  every change.
- Reranking alone against baseline: McNemar p = 0.34 (not significant),
  Wilcoxon p = 0.031 (significant). McNemar saw 10 questions change; Wilcoxon
  saw 31. Reporting only the binary test would have understated a real effect.
- Metadata-augmented index against plain index: McNemar p = 0.035, Wilcoxon
  p = 0.036.
- Final configuration against baseline: hit@5 0.574 to 0.787 (19 fixed, 6
  broken, p = 0.0146); MRR 0.388 to 0.606 (35 improved, 11 worsened,
  p = 0.0007).

Method note: the indexed text includes metadata drawn from the document's own
parsed fields, so the retriever is matching against enriched text rather than
the document's raw wording. This is contextual retrieval and is described as
such rather than reported as plain BM25.

Remaining headroom: 0.787 against a hit@50 of 0.967. Eleven questions sit in
the candidate pool without being promoted into the top five, and two are not
retrieved at depth 50 at all. Both groups are carried into error analysis.
Larger pools were tested and rejected, so further gains would require a
stronger reranker rather than better candidates.


## Task 10 — Generation

A grounded answering layer over the frozen retrieval configuration: passages
rendered with citation numbers, a prompt fixing the grounding contract, two
refusal tokens for two distinct situations, and mechanical post-checks. The
generator is a local 3B model over Ollama, so the whole loop costs nothing.

Generation over all 76 questions, corrected scoring:

| measure                          | value  |
|----------------------------------|--------|
| answered                         | 53     |
| expected figure present          | 37/40 (0.925) |
| correct abstention, unanswerable | 7/10   |
| correct abstention, ambiguous    | 1/5    |
| false abstention                 | 8/61   |
| confabulations                   | 7      |
| uncited claims                   | 21 sentences |

Decisions:

- Validation is deliberately mechanical: whether cited passage numbers exist,
  and whether the figure the reference answer turns on appears in the text.
  Whether an answer is *good* needs a judge and is deferred; whether it cites
  passages that were never supplied can be settled by code, and in a regulated
  domain that is the failure that matters.
- Two refusal tokens rather than one. Missing information and an
  under-specified question are different situations for a support agent: one
  needs escalation, the other needs a follow-up question.
- All 76 questions are generated for, while retrieval metrics continue to
  average over the 61 labelled ones. The 15 unlabelled questions are the point
  of generation evaluation and would otherwise depress every retrieval
  aggregate by a fifth.

Findings:

- **Refusal detection by literal string match misread 12 of 76 answers.** The
  prompt specifies `INSUFFICIENT_CONTEXT`; the model writes `INSUFFICIENT
  CONTEXT`. Correcting the match moved correct abstention from 2/15 to 8/15,
  false abstention from 2/61 to 8/61, and grounded accuracy from 37/46 to
  37/40 — six answers had been counted as wrong-figure when the model had
  declined to give one. Both numerator and denominator were wrong, in opposite
  directions. Matching now tolerates spacing, case and punctuation.
- **The model swaps the two refusal tokens.** It refused an unanswerable
  question with `NEEDS_CLARIFICATION` and an ambiguous one with
  `INSUFFICIENT_CONTEXT`. A 3B model can determine *that* it should refuse but
  not *why*, so abstention is treated as binary and the kind is not reported.
- **Generation exposed a retrieval defect that retrieval metrics could not
  see.** Six of the eight false abstentions had incomplete evidence: the model
  was shown one side of a two-sided comparison and correctly declined. Five
  further questions had incomplete evidence and answered anyway, producing
  comparisons against a card never supplied — scored as successes throughout.
- **hit_rate@k credits a question as soon as one relevant chunk arrives**, so a
  comparison scores 1.0 while half its evidence is missing. Added `evidence@k`:
  the fraction of a question's documents contributing a labelled passage. It
  equals hit rate for single-document questions and diverges sharply for
  comparisons — 0.636 against 0.409 at k=5. The headline 0.787 was carrying
  that inflation.
- **Blind per-document caps were measured and rejected.** Limiting any document
  to 3, 2 or 1 of the five slots moved evidence between categories without
  creating any: overall 0.738 → 0.746 → 0.721 → 0.656, with comparisons gaining
  0.136 and table lookups losing 0.111. The missing passages sit at ranks 12 to
  29, and a cap can only promote what sits at rank 6.
- **Entity-aware selection works because it is conditional.** Cards named in a
  question are detected against product and issuer names already in the corpus;
  when two or more are named, slots are reserved for each, rotating across
  cards and preferring an unrepresented document within each. Questions naming
  one card take an early exit and are returned exactly as the reranker ordered
  them.

| configuration                    | hit@5 | evidence@5 | comparison ev. |
|----------------------------------|-------|------------|----------------|
| reranked baseline                | 0.787 | 0.738      | 0.409 |
| + per-document cap 3             | 0.787 | 0.746      | 0.545 |
| + entity-aware selection (final) | 0.820 | 0.775      | 0.614 |

Entity-confusable, single-fact and table-lookup categories are identical to
baseline to three decimals; the entire gain comes from comparisons.
Wilcoxon on evidence@5: 5 improved, 1 worsened, p = 0.0348. McNemar on hit@5:
2 fixed, 0 broken, p = 0.5000 — the improvement is nearly invisible to the
metric the project began with, which is the argument for having added a second.

Vocabulary construction:

- Generic words appear inside product names — "balance", "statement", "points"
  — and would each reserve a slot for an unrelated document. They are removed
  by body-text document frequency rather than a hand-written stoplist: a brand
  is rare in the corpus body, generic vocabulary is not.
- The threshold was set by inspecting detection quality against these same 76
  questions. **The p-value above is therefore optimistic**; a clean estimate
  needs held-out questions. At 0.2 the filter removed genuine brands and cost
  four comparison questions their second entity; 0.5 retains 9 of 11
  comparisons while dropping single-fact interference to 0 of 20.

Known limitations:

- The one apparent regression, `cmp_apr_bealls_vs_jcpenney_mc` (0.75 → 0.50),
  is a scoring artifact. Its gold set contains three near-identical Bealls
  filings plus JCPenney; the baseline retrieved three copies of one side and
  scored higher, the balanced result holds both sides and scores lower.
  Equivalent filings should count as one group — the groups are already
  enumerated from Task 5.
- Seven confabulations remain, three on cards absent from the corpus. Whether
  these are the 3B model's limits or the prompt's is the Task 11 comparison.
- Four of five ambiguous questions were answered by silently choosing a card.
- Citation validity cannot be recomputed from stored results, since the passage
  list is not persisted; it is only available from a live run.


## Task 11 — Generation experiments

Two questions: are the remaining refusal failures caused by model capacity or by
the prompt, and can a small local model serve this task at all.

Provider comparison, identical retrieval and prompt, 76 questions:

| measure                 | 3B local (Ollama) | 8B hosted (Groq) |
|-------------------------|-------------------|------------------|
| correct abstention      | 10/15             | 9/15   |
| false abstention        | 5/61              | 6/61   |
| invalid citations       | 3                 | 2      |
| expected figure present | 39/43 (0.907)     | 39/42 (0.929) |

Prompt revisions, measured on the 26 questions where refusal is at stake:

| measure          | v1   | v2   | v3 + gate |
|------------------|------|------|-----------|
| unanswerable     | 9/10 | 9/10 | 8/10 |
| ambiguous        | 1/5  | 0/5  | 5/5  |
| false abstention | 4/11 | 2/11 | 3/11 |
| uncited claims   | 22   | 22   | 5    |

Final configuration, 76 questions: correct abstention 13/15, false abstention
4/61, grounded accuracy 40/44 (0.909), 2 answers with invalid citations.

Findings:

- **Model capacity was not the constraint.** Nearly three times the parameters
  moved every measure by at most one question. Once retrieval supplies the
  right passages this task is extraction rather than reasoning, and a 3B model
  running locally is sufficient — which is the difference between a system that
  costs nothing to operate and one that does not.
- **The refusal rule was wrong in both directions at once.** The v1 condition
  fired when a question named *two* cards and failed when it named *none*,
  because the model read the second clause and ignored the first. Comparisons
  were refused; ambiguous questions were answered by silently picking whichever
  card had been retrieved.
- **Rewriting the rule as an explicit procedure made it worse.** v2 instructed
  the model to count the cards named in the question and judge that from the
  question alone. It complied and got the count wrong, taking card names out of
  the passages — "Since the question names only one card, I will answer for
  that card" on a question naming none. Ambiguous fell to 0/5, and narrating
  the procedure broke the two-sentence limit.
- **Entity counting belongs in code, not in the prompt.** The entity detector
  built for retrieval already answers "does this question name a card". Moving
  the decision out of the generator took ambiguous questions from 1/5 to 5/5
  deterministically, with no model call at all. Verified before adoption: all 5
  ambiguous questions name no recognisable card, against 0 of 11 comparisons,
  0 of 21 entity-confusable and 0 of 9 table lookups.
- **A prompt that says less performs better once the hard decision is removed.**
  v3 dropped the ambiguity rule entirely, kept v1's brevity, added a comparison
  carve-out and an instruction not to narrate. Uncited claims on the refusal
  subset fell from 22 to 5.

Free-tier constraints, worth recording as engineering findings:

- The 70B model allows 100,000 tokens per day. One 76-question run needs about
  250,000, so a complete evaluation is impossible within a day at that size —
  the comparison arm became the 8B model for this reason.
- Both hosted models throttle by tokens per minute (12,000 and 6,000). A loop
  issuing requests as fast as they complete fails on the first question.
  Client-side pacing against a rolling window was required.
- Rate-limit errors were indistinguishable until the response body was
  surfaced: a per-minute limit and a per-day limit fail identically, nine
  minutes apart.

Known limitations:

- `unans_amex_platinum_apr` answers "12.74% 21.74% [1]" — with a citation. The
  model is not recalling Amex's rate from training; it is reading another
  card's APR out of a retrieved passage. The failure is not ungrounded
  generation but unverified card identity, which the instruction "never use
  outside knowledge" does not address.
- `unans_alliant_billing_rights` answers generically about billing rights from
  a different issuer's filing.
- Three comparison questions are still refused, all with incomplete evidence.
- One single-fact question names no card the detector recognises and is
  therefore asked for clarification. A measured cost of the gate: one question
  lost against four gained.
- Abstention kind remains unreliable — the model picks the right decision and
  the wrong label. Abstention is treated as binary throughout.


LLM-as-judge, calibrated:

The mechanical check asks only whether the reference figure appears somewhere
in the answer, so "the annual fee is $59", "the annual fee is not $59" and
"$59 is the cash advance fee" all score as correct. A judge grades each answer
against the reference as correct, partial or wrong.

A judge is not evidence until it agrees with a human. 30 answers were graded by
hand, stratified across categories, on a sheet with the judge's verdicts hidden.

| measure                          | value |
|----------------------------------|-------|
| raw agreement                    | 0.933 |
| Cohen's kappa                    | 0.871 |
| judge over 59 answers: correct   | 0.593 |
| judge over 59 answers: partial   | 0.102 |
| judge over 59 answers: wrong     | 0.305 |
| mechanical figure check          | 0.909 |

- **The mechanical check overstated grounded accuracy by roughly 30 points.**
  Human grading of 30 answers gives 0.633 correct; the calibrated judge gives
  0.593 over all 59. Figure presence gives 0.909. Every number reported before
  this section that relies on figure presence is an upper bound.
- **Both disagreements are adjacent categories.** The judge called one correct
  answer partial and one wrong answer partial; it never confused correct with
  wrong. Its errors are calibration of the middle category, not of quality.
- **An 8B judge was sufficient.** Grading against a written reference is an
  easier task than answering, and does not require a larger model than the one
  being graded.
- Kappa rather than raw agreement: with 63% of answers correct, a judge
  answering "correct" every time would score 0.63 raw agreement while carrying
  no information.

Golden set audit, prompted by grading:

Task 5 machine-verified every labelled quote but nothing verified the reference
answers, which are prose written from those quotes. All 76 were audited: 51
containing a figure or phone number were checked automatically against their
quotes, 10 prose references were read by hand, and 15 refusal references have no
labels by design. Two errors were found and corrected, both the same shape — a
clause asserting more than the quote supported. The audit is committed as
`scripts/audit_references.py`.

Grading and the judge run both used the pre-correction wording; neither
correction removes a fact that would change a verdict.