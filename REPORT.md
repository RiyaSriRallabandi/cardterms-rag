# CardTerms: building and measuring a RAG system

A question-answering system over 247 US credit card agreements, built to a
constraint of zero cost. This report covers how it was built, what each
experiment showed, and how those results changed the next decision.

The short version: retrieval improved from 57.4% to 82.0% hit@5, end-to-end
accuracy reached 82.9%, and three of the most interesting results were things
that did *not* work.

---

## 1. The problem and why this corpus

A customer support agent needs a fee or rate from a specific cardholder
agreement. There are hundreds of agreements, they run to dozens of pages, and
they are written to be near-identical.

That last property is what makes the corpus a good test. Issuers file from a
shared template, so documents differ by brand name and a handful of numbers. A
system that retrieves "a credit card agreement" is useless; it has to retrieve
*the right one*. Selecting issuers by US market presence produced eight
naturally confusable families, including Chase, American Express and Comenity
each filing under two legal entities.

**Corpus:** 247 agreements, 46 issuers, 2,722 pages, from the CFPB Q1 2026
release, pinned to one quarter so labels stay valid.

---

## 2. Ingestion: three failures that would have been invisible

PDF parsing looks like solved plumbing. It was not, and each defect below would
have silently degraded every downstream measurement.

**Detected tables were deleting real text.** PyMuPDF reads indented lists and
multi-column prose as sparse tables, and detected regions are excluded from body
text. A fill-ratio threshold fixed it; one issuer's detected tables fell from
349 to 46.

**Five documents extracted as valid but meaningless text.** Their embedded fonts
carry no character mapping, so extraction returned glyph codes. Detection was by
common-word density: affected documents scored 0.0 to 0.8 occurrences per
thousand characters against a corpus range of 20 to 45, with nothing in between.
Page-level OCR recovered them. Undetected, this would have put 90,000 characters
of noise in the index as retrievable content.

**Largest-file-first selection preferred Spanish translations,** which are longer
than their English originals. Every Bank of America filing selected was
Spanish-only, silently removing a top-three issuer from an English-language
system.

*Takeaway carried forward:* a parsing bug does not raise an exception. It
produces plausible-looking text. Every stage after this point got a verification
script that could fail the build.

---

## 3. Evaluation before optimisation

The evaluation set was built before any retrieval tuning, because a system
optimised against a metric invented afterwards is optimised against nothing.

**76 questions, 105 labels.** Six categories, including 15 questions the corpus
*cannot* answer, so refusal is measurable rather than assumed.

**Ground truth is stored as character spans, not chunk IDs.** Chunking strategy
was the first planned experiment, and chunk IDs change whenever it does.
Span-based labels let one evaluation set score all nine chunk sets unchanged.
This single decision made every later comparison possible.

**The loader rejects any quote absent from its document or appearing more than
once, and rolls back rather than loading partially.** This caught 13 defective
labels. An ambiguous quote resolves to an arbitrary position and produces a
label pointing at text that does not answer the question, which raises no error
and depresses every recall figure thereafter.

---

## 4. Retrieval: 57.4% → 82.0%

Nine chunk sets and three embedding models were compared on identical text using
one canonical tokenizer, so only the variable under test varied.

| configuration | hit@5 |
|---|---|
| BM25 baseline, 512-token chunks | 0.574 |
| bge-small dense retrieval | 0.541 |
| + bge-reranker-base, pool 50 | 0.623 |
| + metadata in reranker input | 0.639 |
| + metadata in the indexed text | 0.787 |
| + entity-aware selection (final) | **0.820** |

**Keyword search beat every dense model.** The corpus is near-identical legal
text where questions turn on exact product names and amounts, which is what
lexical matching is for and what semantic similarity blurs.

**The largest single gain came from restoring context that chunking removed.**
Many chunks are bare table fragments like `| Annual Fee | $89 |` that name no
product. Prepending product and issuer to the indexed text raised hit@5 from
0.639 to 0.787, and hit@50 from 0.754 to 0.967. That is contextual retrieval,
and it improved the candidate pool itself rather than merely reordering it.

**Diagnostics, not aggregates, drove the plan.** Three planned features were
reordered or dropped after decomposing a score:

- *Metadata filtering, dropped.* Of 26 failures, 20 retrieved the right document
  and the wrong passage. Filtering could address at most 6.
- *Hybrid search, dropped.* An oracle choosing the better of lexical and dense
  per question reached 0.623 against 0.574. Only 3 of 61 questions were found by
  dense and missed by lexical.
- *Reranking, promoted.* 46 of 61 answers were already retrieved by depth 50 but
  ranked below the cut. That is an 18-point ceiling, and exactly what a
  cross-encoder is for.

**HNSW indexing was benchmarked and rejected.** It reached 0.987 recall at 9 to
12 times the speed of exact search, saving roughly 12 ms on a request where
generation takes seconds. Not worth a recall loss that would make recorded runs
incomparable.

Every gain was checked with paired tests: McNemar on binary hit, Wilcoxon on
per-question reciprocal rank. Reranking alone gave McNemar p = 0.34 and Wilcoxon
p = 0.031 — the binary test saw 10 questions change, the signed-rank test saw 31.
Reporting only the binary test would have understated a real effect.

---

## 5. The metric that was hiding a failure

Comparison questions scored 0.636 on hit@5, which looked mediocre but tolerable.

They were much worse than that. `hit_rate@k` credits a question as answered the
moment *one* relevant chunk arrives. A question comparing two cards scored 1.0
while the model was shown five passages about one of them.

So I added **evidence@k**: the fraction of a question's documents that
contributed a labelled passage. It equals hit rate for single-document questions
and diverges sharply for comparisons.

| | hit@5 | evidence@5 |
|---|---|---|
| comparison | 0.636 | **0.409** |
| entity-confusable | 0.762 | 0.762 |
| table lookup | 0.889 | 0.889 |

A 23-point gap the standard metric could not see.

The fix went through two attempts. A blind cap limiting any document to a fixed
share of the context moved evidence between categories without creating any
(+0.008 overall, and −0.111 on table lookups). What worked was **entity-aware
selection**: detect which cards a question names using product and issuer names
already in the corpus, and reserve context slots for each. Questions naming one
card take an early exit and are untouched.

Comparison evidence went 0.409 → 0.614 with the three single-card categories
bit-identical to baseline. Wilcoxon p = 0.0348. McNemar on hit@5 gave p = 0.50 —
the improvement was nearly invisible to the metric the project started with.

*Takeaway carried forward:* a healthy aggregate can conceal a systematic
failure. The generation layer is what made this one visible.

---

## 6. Generation: what worked was not prompt engineering

The grounding contract is deliberately narrow: answer only from supplied
passages, cite every claim, and emit one of two refusal tokens when the answer
is absent or the question is under-specified.

Three prompt revisions were measured. **Two were rejected.**

- **v2** rewrote the refusal rules as an explicit procedure: count the cards
  named in the question, then follow the matching rule. The model complied and
  miscounted, reading card names out of the *retrieved passages* rather than the
  question. Ambiguous questions fell from 1/5 to 0/5.
- **v4** added an instruction to begin yes/no answers with Yes or No. It
  corrected one of eleven known failures and degraded several: "Yes, there is no
  annual fee" became a bare "Yes", and two answers became incoherent.

Instructing a small model to state a decision produced the decision token as a
reflex rather than a conclusion.

**What worked instead was moving the decision out of the prompt.** Some questions
name no card at all ("How much is the late fee?"), and with 247 agreements the
only correct response is to ask which. The entity detector built for retrieval
already answers that question deterministically, so the check runs in code
before the model is called. Ambiguous questions went from 1/5 to **5/5**, by
construction rather than by persuasion.

**The other thing that worked was changing the model.** Eleven grounding
failures were re-run under an identical prompt and retrieval configuration:

| generator | corrected |
|---|---|
| llama3.2:3b, local | 0 / 11 |
| llama-3.1-8b, hosted | 8 / 11 |
| llama-3.3-70b, hosted | 9 / 11 |

The 8B became the default. It is one question behind the 70B, which is noise at
this sample size, and it is the largest model whose full evaluation fits inside
the free tier's daily token budget. A model that cannot be evaluated cannot be
reported.

**Then the provider retired all three.** Every Llama model was removed from the
free tier after this experiment was run, and the shipped system returned 404 on
its next request. This is the ordinary condition of building on someone else's
inference, and it is the case for owning an evaluation harness rather than a
benchmark score: qualifying the replacement was one command, not a judgement
call. `openai/gpt-oss-20b` was substituted, the full 76 questions re-run, and
retrieval came back identical to four decimal places — the expected result when
only the generator changes, and a check that nothing else had drifted. End-to-end
accuracy rose from 73.7% to **82.9%**. The table above is kept as recorded
history; the numbers reported everywhere else in this document are the re-run.

---

## 7. Measuring answers honestly

The mechanical check asked whether the reference figure appears anywhere in the
answer. It reported **91%** accuracy.

That check cannot distinguish "the annual fee is $59" from "the annual fee is
not $59" or "$59 is the cash advance fee".

So I built an LLM judge, and then calibrated it rather than trusting it: 30
answers were graded by hand on a sheet with the judge's verdicts hidden.

| | |
|---|---|
| raw agreement | 0.967 |
| **Cohen's κ** | **0.929** |
| judge verdicts over all answered questions | 0.678 correct |
| the mechanical check | 0.909 |

The single disagreement was between adjacent categories, and in the lenient
direction; the judge never confused a correct answer with a wrong one. Kappa
rather than raw agreement, because on a set where most answers are correct, a
judge that says "correct" every time scores well and knows nothing.

**Real accuracy was about 23 points below what the cheap check reported.**

One design point matters here. The judge must not be the model being graded: a
model shown its own output tends to prefer it. The first judge was the generator
itself, which is the weaker arrangement; the current one is a larger, different
model, and κ against the same 30 hand labels rose from 0.871 to 0.929. The
calibration is what makes that claim checkable rather than plausible.

Grading also prompted an audit of the ground truth itself. The Task 5 loader had
machine-verified every quote, but nothing had verified the *reference answers*,
which are prose written from those quotes. All 76 were checked; two were wrong,
both the same way, asserting a clause the quote did not support. The audit is a
committed script.

---

## 8. Where the errors actually are

Every question was classified by the component that would have to change to fix
it.

| outcome | n |
|---|---|
| answered correctly | 49 |
| correctly refused | 14 |
| refused, evidence incomplete — defensible | 5 |
| refused despite complete evidence | 1 |
| wrong: passage in the pool, not in the top 5 | 2 |
| wrong: answered on incomplete evidence | 1 |
| wrong: evidence complete (grounding failure) | 3 |
| answered what the corpus cannot answer | 1 |

**63 correct (82.9%), 8 retrieval-caused, 5 generation-caused.**

The bottleneck moved twice. Early on, retrieval dominated: 26 failures, 20 of
them right-document-wrong-passage. After reranking and entity-aware selection,
the 3B model made generation twice the problem. Replacing the generator
rebalanced it. Each stage was bottlenecked somewhere different, and no aggregate
score ever revealed where.

---

## 9. Latency

| stage | p50 | share |
|---|---|---|
| retrieve | 94 ms | 1% |
| rerank | 6,947 ms | **90%** |
| generate | 693 ms | 9% |
| total | 7,652 ms | |

**The language model is 9% of response time; the local cross-encoder is 90%.**
Corpus size is not the constraint — lexical retrieval over 5,274 chunks returns
in 95 ms. A cross-encoder reads question and passage together in one forward
pass, so 50 passages at up to 512 tokens is roughly 25,000 tokens through a 278M
model per request, on a laptop GPU.

Halving the candidate pool would roughly halve latency. I did not do it: the
accuracy cost is unmeasured, and testing it properly means re-running the
evaluation and re-validating every reported number. What *is* measured is that a
pool of 100 is worse than 50, and that a cheaper cross-encoder costs 8 points of
retrieval accuracy. Both obvious speed optimisations are either unhelpful or
known to be expensive.

Separately: the free tier sustains about 1.3 questions per minute, with a median
wait of 38 s for token budget before each request. That is a throughput cap, not
latency, and the two are reported separately.

---

## 10. Final system and limitations

```
BM25 over metadata-augmented text  →  50 candidates
  →  bge-reranker-base cross-encoder
  →  entity-aware selection        →  5 passages
  →  openai/gpt-oss-20b, grounding contract, clarification gate
```

| | |
|---|---|
| End-to-end correct | 63 / 76 (82.9%) |
| Document retrieval | 96.7% |
| Passage retrieval hit@5 | 82.0% |
| Evidence coverage@5 | 77.5% |
| Correct refusals | 14 / 15 |
| Wrong answers carrying a citation | 6 (7.9%) |
| Judge agreement with hand labels (κ) | 0.929 |
| Median response | 7.7 s |
| Cost | $0 |

**Six answers are wrong while citing a passage.** This is why the interface
always shows the source text beside the answer, and why the system is positioned
as a research tool for an agent rather than something that answers cardholders
directly.

**Comparison questions remain weakest** at 0.614 evidence coverage.

**A local fallback** runs entirely offline at 69.7% accuracy and 40 seconds per
question, trading 13 points of accuracy for privacy and no rate limit. The two
configurations differ only in the generator, and the error analysis separates
cleanly: 5 generation-caused failures hosted against 14 local, of which 3 and 10
respectively are grounding failures on evidence that was complete in both. The
small model retrieves the same passages and then misreads them.

**With more time, in order:** structured extraction of fee tables, since the
remaining grounding failures are APR-range and comparison cases where the wrong
row of a table was read; candidate pool size versus
latency, currently unmeasured; and collapsing equivalent filings in the evidence
metric, which currently counts three near-identical Bealls documents as three
separate pieces of evidence.

---

## Running it

Requires Docker, Python 3.12, [uv](https://docs.astral.sh/uv/), and a free
[Groq](https://console.groq.com) API key.

```bash
uv sync
cp .env.example .env          # add GROQ_API_KEY
docker compose up -d          # Postgres with pgvector

uv run python scripts/init_db.py
uv run python scripts/build_manifest.py
uv run python scripts/fetch_corpus.py
uv run python scripts/load_documents.py
uv run python scripts/parse_documents.py   # slow, roughly 20 minutes
uv run python scripts/build_chunks.py
uv run python scripts/load_golden_set.py

uv run uvicorn cardterms.api:app --port 8000
```

Open http://localhost:8000. First start takes 20 to 40 seconds while the
retrieval index and reranker load.

Things worth trying in the interface:

- *"What's the annual fee on the OpenSky Gold card?"* — the answer, its citation,
  and the amount highlighted in the agreement text
- *"How much is the late fee?"* — the clarification gate, answered in code
- *"What's the APR on the American Express Platinum Card?"* — a card absent from
  the corpus, and one of the two known confabulations

To reproduce the reported evaluation:

```bash
uv run python scripts/run_eval.py --chunk-set fixed_512_ov0 \
    --rerank bge --candidates 50 --augment-rerank --entity-select \
    --generate --provider groq --gen-model openai/gpt-oss-20b \
    --prompt answer_v3 --clarify-gate
```

Grading uses a different and larger model than the one being graded:

```bash
uv run python scripts/judge_run.py --run-id RUN --model openai/gpt-oss-120b
```

Run `git checkout eval-v1` for the exact state that produced the reported
numbers. The full decision log, including everything cut for length here, is in
[experiments/NOTES.md](experiments/NOTES.md).
