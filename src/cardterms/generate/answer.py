"""Produce a grounded answer from retrieved passages."""

import re
from dataclasses import dataclass
from pathlib import Path

from cardterms.eval.llm import complete
from cardterms.generate.context import Passage, build_passages, render

PROMPT_DIR = Path("prompts")

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
) -> Answer:
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
