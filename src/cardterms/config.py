"""Typed configuration loaded from YAML + environment variables."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive", "structure_aware", "parent_doc"] = "fixed"
    chunk_tokens: int = Field(512, gt=0, le=8192)
    overlap_pct: float = Field(0.0, ge=0.0, lt=0.9)


class EmbeddingConfig(BaseModel):
    model_name: str
    query_prefix: str = ""
    doc_prefix: str = ""
    batch_size: int = Field(32, gt=0)
    normalize: bool = True


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid_rrf", "hybrid_weighted"] = "dense"
    top_k: int = Field(5, gt=0)
    candidate_pool: int = Field(20, gt=0)
    rrf_k: int = 60
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    metadata_filter: bool = False
    index_type: Literal["exact", "hnsw"] = "exact"
    ef_search: int = 40


class RerankingConfig(BaseModel):
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class GenerationConfig(BaseModel):
    provider: Literal["ollama", "groq", "gemini"] = "ollama"
    model_name: str = "llama3.2:3b"
    prompt_version: str = "answer_v1"
    max_context_tokens: int = 3000
    temperature: float = 0.0
    context_order: Literal["rank", "interleave"] = "rank"


class CorpusConfig(BaseModel):
    manifest_path: str = "data/corpus_manifest.csv"
    raw_dir: str = "data/corpus/raw"


class EvaluationConfig(BaseModel):
    golden_set_path: str = "data/eval/golden_set.jsonl"
    k_values: list[int] = [1, 3, 5, 10]
    bootstrap_samples: int = 1000


class Settings(BaseSettings):
    """Secrets come from .env; pipeline settings come from YAML."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://cardterms:cardterms@localhost:5433/cardterms"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    log_level: str = "INFO"


class ExperimentConfig(BaseModel):
    """The full, validated description of one pipeline configuration."""

    run_name: str = "base"
    corpus: CorpusConfig = CorpusConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
