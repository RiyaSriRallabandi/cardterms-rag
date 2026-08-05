CREATE EXTENSION IF NOT EXISTS vector;

-- ============ CORPUS ============

CREATE TABLE IF NOT EXISTS documents (
    id                SERIAL PRIMARY KEY,
    doc_uid           TEXT UNIQUE NOT NULL,   -- stable ID from the manifest
    issuer            TEXT NOT NULL,
    product_name      TEXT,
    effective_quarter TEXT,
    source_url        TEXT,
    sha256            TEXT NOT NULL,
    page_count        INT,
    is_scanned        BOOLEAN DEFAULT FALSE,
    raw_text          TEXT,
    ingested_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_issuer ON documents(issuer);

-- ============ CHUNKS ============
-- Multiple chunk sets coexist so chunking strategies can be compared
-- without re-ingesting or destroying earlier results.

CREATE TABLE IF NOT EXISTS chunk_sets (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,      -- e.g. "fixed_512_ov0"
    config      JSONB NOT NULL,            -- the chunking config that made it
    n_chunks    INT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_set_id    INT NOT NULL REFERENCES chunk_sets(id) ON DELETE CASCADE,
    doc_id          INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    text            TEXT NOT NULL,
    token_count     INT NOT NULL,
    page_start      INT,
    page_end        INT,
    char_start      INT,        -- offset into documents.raw_text
    char_end        INT,        -- needed to match against span-based labels
    section_path    TEXT,
    is_table        BOOLEAN DEFAULT FALSE,
    parent_chunk_id BIGINT REFERENCES chunks(id),
    content_hash    TEXT NOT NULL,
    UNIQUE (chunk_set_id, doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_set  ON chunks(chunk_set_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);

-- ============ EVALUATION ============

CREATE TABLE IF NOT EXISTS eval_questions (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    category    TEXT NOT NULL,   -- single_fact | entity_confusable | table |
                                 -- comparison | unanswerable | ambiguous
    issuer_hint TEXT,
    notes       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Ground truth as document + character span, NOT chunk id.
-- Chunk ids change whenever chunking strategy changes; spans do not.
CREATE TABLE IF NOT EXISTS eval_labels (
    id           SERIAL PRIMARY KEY,
    question_id  INT NOT NULL REFERENCES eval_questions(id) ON DELETE CASCADE,
    doc_id       INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page         INT,
    char_start   INT NOT NULL,
    char_end     INT NOT NULL,
    quote        TEXT NOT NULL,   -- the literal supporting text, for auditing
    relevance    SMALLINT NOT NULL DEFAULT 2,  -- 2 = fully answers, 1 = partial
    UNIQUE (question_id, doc_id, char_start, char_end)
);

-- ============ EXPERIMENTS ============
-- Append-only. Never UPDATE or DELETE a run.

CREATE TABLE IF NOT EXISTS runs (
    id            UUID PRIMARY KEY,
    run_name      TEXT NOT NULL,
    git_sha       TEXT,
    config        JSONB NOT NULL,       -- full resolved config
    config_hash   TEXT NOT NULL,        -- detects duplicate runs
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT DEFAULT 'running',
    metrics       JSONB,                -- aggregate metrics + CIs
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(run_name);

CREATE TABLE IF NOT EXISTS run_results (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    question_id  INT NOT NULL REFERENCES eval_questions(id),
    retrieved    JSONB,      -- [{chunk_id, score, rank, stage}, ...]
    answer       TEXT,
    citations    JSONB,
    abstained    BOOLEAN DEFAULT FALSE,
    metrics      JSONB,      -- per-question metrics
    latencies    JSONB,      -- per-stage latency in ms
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (run_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_run_results_run ON run_results(run_id);