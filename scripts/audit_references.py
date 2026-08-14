"""Check reference answers against the text they were written from.

Task 5 verified that every labelled quote appears verbatim in its document.
Nothing verified the reference answers, which are prose written from those
quotes by hand — and a wrong reference is worse than a wrong system answer,
because it silently corrupts every measurement taken against it.

Every monetary amount or percentage in a reference answer should appear in at
least one of that question's quotes. This cannot check prose claims, so it is a
first pass rather than a proof.

    uv run python scripts/audit_references.py
"""

import re

from cardterms.db import get_conn

FIGURE_RE = re.compile(
    r"\$\s?[\d,]+(?:\.\d+)?"  # dollar amounts
    r"|\d+(?:\.\d+)?\s?%"  # percentages
    r"|\d[\d\-().\s]{6,}\d"  # phone numbers
)


def normalise(text: str) -> str:
    """Compare figures without spacing, separator or trailing-zero differences."""
    # The decimal point is significant — stripping it turns 2.00% into 200%.
    for character in " ,-()":
        text = text.replace(character, "")
    return text


def variants(figure: str) -> set[str]:
    """$40 and $40.00 are the same amount; so are 2% and 2.00%."""
    base = normalise(figure)
    forms = {base}

    prefix, suffix, number = "", "", base
    if base.startswith("$"):
        prefix, number = "$", base[1:]
    elif base.endswith("%"):
        suffix, number = "%", base[:-1]

    if number.endswith(".00"):
        forms.add(f"{prefix}{number[:-3]}{suffix}")
    elif "." not in number:
        forms.add(f"{prefix}{number}.00{suffix}")
    return forms


def main() -> None:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT q.question_uid, q.category, q.reference_answer,
                   string_agg(l.quote, E'\\n---\\n') AS quotes
            FROM eval_questions q
            LEFT JOIN eval_labels l ON l.question_id = q.id
            WHERE q.reference_answer IS NOT NULL
            GROUP BY q.question_uid, q.category, q.reference_answer
            ORDER BY q.question_uid
            """
        ).fetchall()

    unsupported = []
    unchecked = []

    for row in rows:
        figures = FIGURE_RE.findall(row["reference_answer"])
        if not figures:
            unchecked.append(row)
            continue

        quotes = normalise(row["quotes"] or "")
        missing = [f for f in figures if not any(v in quotes for v in variants(f))]
        if missing:
            unsupported.append((row, missing))

    print(f"\n  {len(rows)} reference answers checked\n")

    print(f"  figures not found in any labelled quote: {len(unsupported)}")
    for row, missing in unsupported:
        print(f"\n    {row['question_uid']}  [{row['category']}]")
        print(f"      reference: {row['reference_answer']}")
        print(f"      missing:   {', '.join(missing)}")
        print(f"      quotes:    {(row['quotes'] or '')[:160]}")

    # Refusal questions carry no labels by design: the corpus lacks the answer,
    # which is the point of the question. Nothing to check them against.
    by_design = [row for row in unchecked if not row["quotes"]]
    needs_eye = [row for row in unchecked if row["quotes"]]

    print(
        f"\n  {len(by_design)} references are refusals with no labels by design "
        "(nothing to verify against)"
    )

    print(
        f"\n  {len(needs_eye)} prose references carry no checkable figure — "
        "read these against their quotes:"
    )
    for row in needs_eye:
        print(f"\n    {row['question_uid']}  [{row['category']}]")
        print(f"      reference: {row['reference_answer']}")
        print(f"      quote:     {' / '.join(row['quotes'].splitlines())[:220]}")


if __name__ == "__main__":
    main()
