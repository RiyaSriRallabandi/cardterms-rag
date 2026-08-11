"""Embedding model registry and encoding.

Several models are asymmetric: they expect queries and passages to be marked
with different instruction prefixes, and omitting them costs accuracy without
raising any error. Prefixes are therefore part of the model definition rather
than something callers remember to apply.

Vectors are normalised to unit length so that cosine distance, inner product
and Euclidean distance rank identically, and the pgvector operator can be
chosen for speed rather than correctness.
"""

from dataclasses import dataclass
from functools import lru_cache

BATCH_SIZE = 32


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    hf_name: str
    dimensions: int
    max_tokens: int
    query_prefix: str = ""
    passage_prefix: str = ""


REGISTRY = {
    "minilm": EmbeddingModel(
        key="minilm",
        hf_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        max_tokens=256,
    ),
    "bge-small": EmbeddingModel(
        key="bge-small",
        hf_name="BAAI/bge-small-en-v1.5",
        dimensions=384,
        max_tokens=512,
        # BGE marks the query only; passages are embedded unmodified.
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-base": EmbeddingModel(
        key="bge-base",
        hf_name="BAAI/bge-base-en-v1.5",
        dimensions=768,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


@lru_cache(maxsize=2)
def load(hf_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(hf_name)


def encode_passages(
    model: EmbeddingModel, texts: list[str], use_prefix: bool = True
) -> list[list[float]]:
    encoder = load(model.hf_name)
    prefix = model.passage_prefix if use_prefix else ""
    vectors = encoder.encode(
        [prefix + text for text in texts],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def encode_query(model: EmbeddingModel, text: str, use_prefix: bool = True):
    """Return a normalised query vector.

    A numpy array is returned rather than a list: pgvector's psycopg adapter
    maps arrays to the vector type, while a Python list is sent as a float
    array, which Postgres accepts on insert but not as an operator argument.
    """
    encoder = load(model.hf_name)
    prefix = model.query_prefix if use_prefix else ""
    return encoder.encode(
        prefix + text, normalize_embeddings=True, convert_to_numpy=True
    )
