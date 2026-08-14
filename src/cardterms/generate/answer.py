"""Produce a grounded answer from retrieved passages."""

import re
from dataclasses import dataclass
from pathlib import Path

from cardterms.eval.llm import complete
from cardterms.generate.context import Passage, build_passages, render
from cardterms.retrieve.entities import detect

PROMPT_DIR = Path("prompts")

# The canonical tokens, as written in the prompt and emitted when the pipeline
# refuses without consulting the model.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"
NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

# Models reproduce these tokens loosely — "INSUFFICIENT CONTEXT", lowercase, or
# wrapped in punctuation. Matching the literal string silently misclassifies a
# correct refusal as a confabulation, so match the shape instead.
INSUFFICIENT_RE = re.compile(r"INSUFFICIENT[\s_\-]*CONTEXT", re.IGNORECASE)
NEEDS_CLARIFICATION_RE = re.compile(r"NEEDS[\s_\-]*CLARIFICATION", re.IGNORECASE)

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    abstained: bool
    abstention_kind: str | None
    cited: list[int]
    passages: list[Passage]


def load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"{version}.txt").read_text()


def generate(
    conn,
    question: str,
    retrieved: list[tuple[int, int, float]],
    prompt_version: str = "answer_v1",
    provider: str = "ollama",
    model: str | None = None,
    max_context_tokens: int = 3000,
    entity_index: dict[str, frozenset[int]] | None = None,
) -> Answer:
    # Deciding whether a question names a card is entity recognition, and a 3B
    # model does it badly: asked to count cards in the question, it reads a name
    # out of the passages instead and answers for whichever card was retrieved.
    # The corpus already provides a reliable answer, so the decision is made
    # here rather than delegated to the generator.
    if entity_index is not None and not detect(question, entity_index):
        return Answer(
            text=NEEDS_CLARIFICATION,
            abstained=True,
            abstention_kind="needs_clarification",
            cited=[],
            passages=[],
        )

    passages = build_passages(conn, retrieved, max_context_tokens)
    prompt = load_prompt(prompt_version).format(
        passages=render(passages), question=question
    )

    raw = complete(prompt, provider=provider, model=model, temperature=0.0).strip()

    kind = None
    if INSUFFICIENT_RE.search(raw):
        kind = "insufficient_context"
    elif NEEDS_CLARIFICATION_RE.search(raw):
        kind = "needs_clarification"

    cited = sorted({int(n) for n in CITATION_RE.findall(raw)})

    return Answer(
        text=raw,
        abstained=kind is not None,
        abstention_kind=kind,
        cited=cited,
        passages=passages,
    )
