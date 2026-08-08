"""Resolve a card product name from document text.

Filenames are unreliable: several issuers name filings with internal codes
(for example 'COL00097'). Product names are therefore recovered from the
document body.

Issuers declare the product in several grammatical forms, so a set of patterns
is tried in descending order of precision. The pattern that succeeded is
recorded per document, so extraction quality can be measured rather than
assumed. Some filings — notably generic cardmember agreements identified only
by an internal collection code — name no product at all, and fall back to the
filename.
"""

import re

CARD_TOKEN = (
    r"(?:Mastercard|MasterCard|Visa|American Express|Amex|Discover|"
    r"Credit Card|Card)"
)

# "Credit Card Agreement for Bass Pro Shops CLUB Cards in Capital One, N.A."
AGREEMENT_TITLE_RE = re.compile(
    r"(?:Consumer\s+)?Credit\s+Card\s+Agreement\s+for\s+(?:the\s+|your\s+)?"
    r"(?P<name>[^\n]{3,90}?)"
    r"(?:\s+in\s+[A-Z][^\n]*)?$",
    re.MULTILINE,
)

# "AEO, INC. VISA® CARD ACCOUNT AGREEMENT" — an upper-case title line.
CAPS_TITLE_RE = re.compile(
    r"^(?P<name>[A-Z0-9][A-Z0-9&'’\-.,()/®™ ]{4,80}?)\s+ACCOUNT\s+AGREEMENT\s*$",
    re.MULTILINE,
)

# "...is part of the Credit Card Agreement for the Bealls Family of Stores
# Credit Card Account."
ACCOUNT_DECLARATION_RE = re.compile(
    r"\bfor\s+(?:the|your)\s+(?P<name>[A-Z][\w&'’\-.®™()/ ]{2,70}?)\s+Account\b"
)

# "Prior to applying for a Saks World Elite Mastercard® Credit Card or ..."
APPLICATION_NOTICE_RE = re.compile(
    r"applying\s+for\s+(?:a|an|the)\s+(?P<name>[^\n,;.]{3,90})",
    re.IGNORECASE,
)

# The product name alone on its own line.
PRODUCT_LINE_RE = re.compile(
    rf"^(?P<name>[A-Z0-9][\w&'’\-.,()/ ]{{2,70}}?{CARD_TOKEN}[®™]?"
    r"(?:\s+(?:Account|Agreement|Card))*)\s*$",
    re.MULTILINE,
)

SUMMARY_HEADING_RE = re.compile(r"rate and fee summary", re.IGNORECASE)

# Internal reference codes such as 'FR833282333' or 'WF13625080X'.
REFERENCE_CODE_RE = re.compile(r"\b[A-Z]*\d{5,}\w*\b")

# Declarative patterns, ordered by precision. Each exposes a group "name".
DECLARATIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("agreement_title", AGREEMENT_TITLE_RE),
    ("caps_title", CAPS_TITLE_RE),
    ("account_declaration", ACCOUNT_DECLARATION_RE),
    ("application_notice", APPLICATION_NOTICE_RE),
]

# Matches that identify no particular product.
GENERIC_NAMES = {
    "account",
    "card",
    "cards",
    "credit",
    "credit card",
    "credit cards",
    "credit card account",
    "credit card agreement",
    "cardmember agreement",
    "consumer credit card",
    "consumer credit cards",
    "new accounts",
    "this card",
    "your account",
    "visa",
    "mastercard",
}

HEAD_CHARS = 6000
SUMMARY_WINDOW_CHARS = 800


def _tidy(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" .,;:-–—")


def _accept(name: str | None) -> str | None:
    """Reject matches that identify no particular product."""
    if not name:
        return None
    cleaned = _tidy(name)
    if len(cleaned) < 4 or cleaned.lower() in GENERIC_NAMES:
        return None
    if not re.search(r"[A-Za-z]", cleaned):
        return None
    if REFERENCE_CODE_RE.search(cleaned):
        return None
    return cleaned


def extract_product_name(text: str) -> tuple[str | None, str]:
    """Return (product_name, source).

    Sources, in the order attempted:
      agreement_title     - "Credit Card Agreement for <name> in <issuer>"
      caps_title          - "<NAME> ACCOUNT AGREEMENT"
      account_declaration - "...Agreement for the <name> Account"
      application_notice  - "applying for a <name>"
      rate_fee_summary    - heading line following the Rate and Fee Summary
      document_head       - heading line anywhere in the opening pages
      filename            - the filing names no product
    """
    head = text[:HEAD_CHARS]

    for source, pattern in DECLARATIVE_PATTERNS:
        match = pattern.search(head)
        if match and (name := _accept(match.group("name"))):
            return name, source

    heading = SUMMARY_HEADING_RE.search(head)
    if heading:
        window = head[heading.end() : heading.end() + SUMMARY_WINDOW_CHARS]
        match = PRODUCT_LINE_RE.search(window)
        if match and (name := _accept(match.group("name"))):
            return name, "rate_fee_summary"

    match = PRODUCT_LINE_RE.search(head)
    if match and (name := _accept(match.group("name"))):
        return name, "document_head"

    return None, "filename"
