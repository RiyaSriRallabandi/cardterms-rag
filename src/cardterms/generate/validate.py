"""Checks on a generated answer that require no judgement.

These are deliberately mechanical. Whether an answer is *good* needs an
evaluator; whether its citations point at passages that were actually supplied,
and whether the expected figure appears at all, can be established by code — and
those checks catch the failures that matter most in a regulated domain.
"""

import re

from cardterms.generate.answer import Answer

# Sentences shorter than this are fragments rather than claims.
MIN_CLAIM_CHARS = 25

# Monetary amounts and percentages: what these questions actually turn on.
FIGURE_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?\s?%")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CITATION_RE = re.compile(r"\[\d+\]")


def validate(answer: Answer, reference: str | None) -> dict:
    if answer.abstained:
        return {
            "citations_valid": True,
            "invalid_citations": 0,
            "uncited_claims": 0,
            "contains_reference_figure": None,
        }

    valid_numbers = {p.number for p in answer.passages}
    invalid = [n for n in answer.cited if n not in valid_numbers]

    uncited = 0
    for sentence in SENTENCE_SPLIT_RE.split(answer.text):
        if len(sentence.strip()) >= MIN_CLAIM_CHARS and not CITATION_RE.search(
            sentence
        ):
            uncited += 1

    # A cheap correctness proxy: the figure the reference answer turns on should
    # appear somewhere in the generated text. Only applies where the reference
    # states one.
    contains = None
    if reference:
        figures = FIGURE_RE.findall(reference)
        if figures:
            normalised = answer.text.replace(" ", "")
            contains = any(f.replace(" ", "") in normalised for f in figures)

    return {
        "citations_valid": not invalid,
        "invalid_citations": len(invalid),
        "uncited_claims": uncited,
        "contains_reference_figure": contains,
    }
