"""The question-answering pipeline as a callable service.

Until now the pipeline existed only as a loop inside the evaluation script, so
nothing but the evaluation could run it. This assembles the same components in
the same order and exposes one method, which keeps the served system and the
measured system identical by construction — a serving path that reimplements
retrieval is a serving path whose accuracy is unknown.

Retriever, reranker and entity vocabulary are built once at startup. The BM25
index covers 5,274 chunks and the entity vocabulary scans every document, so
building either per request would dominate latency.
"""

import re
import time
from dataclasses import dataclass, field

from cardterms.config import ExperimentConfig
from cardterms.db import connect
from cardterms.eval.llm import last_throttle_seconds
from cardterms.generate.answer import Answer, generate
from cardterms.generate.context import Passage
from cardterms.logging import log
from cardterms.retrieve.bm25 import BM25Retriever
from cardterms.retrieve.diversify import select_by_entity
from cardterms.retrieve.entities import build_index, detect
from cardterms.retrieve.rerank import CrossEncoderReranker

# Words that describe the document rather than the card. Stripped from the
# ends of a name, not the middle: "Credit Card" is noise trailing "Venmo Visa
# Signature" and meaningful inside "Bealls Credit Card Rewards".
# fmt: off
DOC_WORDS = {
    "account", "accounts", "agreement", "agreements", "and", "application",
    "card", "cardholder", "cardmember", "cards", "conditions", "consumer",
    "credit", "current", "details", "disclosure", "disclosures", "for",
    "information", "pricing", "program", "solicitation", "statement",
    "terms", "the", "addendum", "final", "rates", "fees",
}
# fmt: on

# Filing codes that survive into filenames: "CC 10.1.22", "v211", "sky2".
_CODE_RE = re.compile(
    r"\b(?:cc\s*)?v?\d+(?:[._]\d+)+\b"  # 10.1.22
    r"|\bv\d{2,}\b"  # v211
    r"|\b[a-z]{2,6}\d{1,3}\b",  # sky2, plus018
    re.IGNORECASE,
)


# Words that begin a sentence, not a card name. Product names were extracted
# from document text and roughly 40% of filings state none, so the field
# sometimes holds a fragment of surrounding prose: "About your", "to the
# Standard.". Those are rejected in favour of the issuer, which comes from
# CFPB metadata and is always clean.
# fmt: off
FRAGMENT_STARTS = {
    "about", "after", "all", "any", "as", "at", "based", "before", "by",
    "each", "how", "if", "in", "is", "it", "may", "of", "on", "or", "our",
    "please", "refer", "see", "that", "this", "to", "we", "what", "when",
    "will", "you", "your",
}
# fmt: on


def _is_card_like(label: str) -> bool:
    """Whether a cleaned name reads as a product rather than stray prose."""
    if not label or len(label) < 4:
        return False
    if any(ch in label for ch in "():;"):
        return False
    if label[0].islower():
        return False

    words = label.split()
    if words[0].lower().strip(".,") in FRAGMENT_STARTS:
        return False
    # A single word can be a brand ("Venmo"), but not if it trails punctuation
    # from the middle of a sentence.
    return not label.rstrip().endswith(".") or len(words) > 1


def _card_label(product: str, issuer: str) -> str:
    """A name a cardholder would recognise, or the issuer if none survives."""
    words = _CODE_RE.sub(" ", product or "").replace("_", " ").split()

    # Trim document vocabulary from both ends until something distinctive
    # remains. Anything left in the middle is part of the card's name.
    while words and words[0].lower().strip(".,-") in DOC_WORDS:
        words.pop(0)

    def _droppable(index: int) -> bool:
        word = words[index].lower().strip(".,-")
        if word in DOC_WORDS:
            return True
        # Filenames repeat the variant as a suffix code: "OpenSky Plus Card
        # Agreement plus". A trailing word already used earlier is that code.
        return any(w.lower().strip(".,-") == word for w in words[:index])

    while words and _droppable(len(words) - 1):
        words.pop()

    label = " ".join(words).strip(" -–—,")
    return label if _is_card_like(label) else issuer.title()


@dataclass
class Timing:
    """Milliseconds per stage, so a slow request can be attributed."""

    retrieve: float = 0.0
    rerank: float = 0.0
    generate: float = 0.0
    # Time spent waiting on the provider's rate limit. Excluded from latency:
    # a throttled request is not a slow one, and the free tier's throughput cap
    # is a separate fact from how fast the system computes an answer.
    throttle: float = 0.0

    @property
    def total(self) -> float:
        return self.retrieve + self.rerank + self.generate


@dataclass
class Result:
    question: str
    answer: str
    abstained: bool
    abstention_kind: str | None
    cited: list[int]
    passages: list[Passage]
    entities: list[str]
    timing: Timing = field(default_factory=Timing)


class CardTerms:
    def __init__(
        self,
        config_path: str = "configs/base.yaml",
        chunk_set: str = "fixed_512_ov0",
        reranker: str = "bge",
        candidates: int = 50,
        prompt: str = "answer_v3",
        provider: str = "groq",
        model: str | None = "openai/gpt-oss-20b",
    ) -> None:
        self.config = ExperimentConfig.from_yaml(config_path)
        self.chunk_set = chunk_set
        self.candidates = candidates
        self.prompt = prompt
        self.provider = provider
        self.model = model

        started = time.perf_counter()
        self._conn = connect()
        self.retriever = BM25Retriever.from_chunk_set(self._conn, chunk_set)
        self.reranker = CrossEncoderReranker(self._conn, reranker, augment=True)
        self.entity_index = build_index(self._conn)
        log.info(
            "service_ready",
            seconds=round(time.perf_counter() - started, 1),
            vocabulary=len(self.entity_index),
        )

    def close(self) -> None:
        self._conn.close()

    def products(self, query: str = "", limit: int = 40) -> list[dict]:
        """Cards matching a substring, for the "which card?" picker.

        Product names were resolved from document text and about 40% of filings
        state none, leaving a filename that is often a section heading rather
        than a card: "Application and Solicitation Disclosure". Those are
        cleaned here rather than in the database, because the stored values
        feed the retrieval index and changing them would invalidate every
        recorded evaluation run.
        """
        rows = self._conn.execute(
            """
            SELECT id, issuer,
                   coalesce(product_name, filename_product, '') AS product
            FROM documents
            WHERE %s = ''
               OR coalesce(product_name, filename_product, '') ILIKE %s
               OR issuer ILIKE %s
            ORDER BY issuer, product
            """,
            (query, f"%{query}%", f"%{query}%"),
        ).fetchall()

        seen: set[tuple[str, str]] = set()
        cards: list[dict] = []
        for row in rows:
            label = _card_label(row["product"], row["issuer"])
            key = (row["issuer"], label)
            if key in seen:
                continue
            seen.add(key)
            cards.append({"id": row["id"], "issuer": row["issuer"], "product": label})
            if len(cards) >= limit:
                break
        return cards

    def ask(self, question: str) -> Result:
        timing = Timing()
        top_k = self.config.retrieval.top_k

        # A question naming no card is answered here, not by the generator: a
        # small model asked to make this judgement reads a card name out of the
        # retrieved passages and answers for whichever card was returned.
        entities = detect(question, self.entity_index)
        if not entities:
            return Result(
                question=question,
                answer="NEEDS_CLARIFICATION",
                abstained=True,
                abstention_kind="needs_clarification",
                cited=[],
                passages=[],
                entities=[],
                timing=timing,
            )

        started = time.perf_counter()
        pool = self.retriever.search(question, self.candidates)
        timing.retrieve = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        ranked = self.reranker.rerank(question, pool, self.candidates)
        ranked = select_by_entity(ranked, top_k, entities)
        timing.rerank = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        answer: Answer = generate(
            self._conn,
            question,
            ranked[:top_k],
            prompt_version=self.prompt,
            provider=self.provider,
            model=self.model,
            max_context_tokens=self.config.generation.max_context_tokens,
        )
        timing.throttle = last_throttle_seconds() * 1000
        timing.generate = (time.perf_counter() - started) * 1000 - timing.throttle

        return Result(
            question=question,
            answer=answer.text,
            abstained=answer.abstained,
            abstention_kind=answer.abstention_kind,
            cited=answer.cited,
            passages=answer.passages,
            entities=[e.token for e in entities],
            timing=timing,
        )
