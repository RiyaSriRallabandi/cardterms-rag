"""Canonical token counting for chunk construction.

A single tokenizer is used for every chunk set so that a chunk set is
independent of the embedding model applied to it. Embedding models are then
compared on identical text rather than on boundaries their own tokenizers
happened to produce.
"""

from functools import lru_cache

from transformers import AutoTokenizer

# BERT WordPiece, shared in form by the candidate embedding models.
CANONICAL_TOKENIZER = "BAAI/bge-small-en-v1.5"

# Maximum input length of each candidate model, used to flag chunk sets that a
# given model cannot consume without truncation.
MODEL_TOKEN_LIMITS = {
    "sentence-transformers/all-MiniLM-L6-v2": 256,
    "BAAI/bge-small-en-v1.5": 512,
    "nomic-ai/nomic-embed-text-v1.5": 8192,
}


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(CANONICAL_TOKENIZER)


def count_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def models_that_fit(max_chunk_tokens: int) -> list[str]:
    """Models able to consume chunks of this size without truncation."""
    return [
        name for name, limit in MODEL_TOKEN_LIMITS.items() if max_chunk_tokens <= limit
    ]
