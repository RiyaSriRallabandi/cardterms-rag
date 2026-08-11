-- Vector columns in pgvector are fixed-width, so models are grouped by
-- dimensionality. The model key and prefix scheme are stored per row so that
-- several models, and the with/without-prefix ablation, coexist over the same
-- chunks without re-indexing.

CREATE TABLE IF NOT EXISTS embeddings_384 (
    chunk_id      BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model         TEXT   NOT NULL,
    prefix_scheme TEXT   NOT NULL DEFAULT 'standard',
    vec           vector(384) NOT NULL,
    PRIMARY KEY (chunk_id, model, prefix_scheme)
);

CREATE TABLE IF NOT EXISTS embeddings_768 (
    chunk_id      BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model         TEXT   NOT NULL,
    prefix_scheme TEXT   NOT NULL DEFAULT 'standard',
    vec           vector(768) NOT NULL,
    PRIMARY KEY (chunk_id, model, prefix_scheme)
);

CREATE INDEX IF NOT EXISTS idx_emb384_model ON embeddings_384(model, prefix_scheme);
CREATE INDEX IF NOT EXISTS idx_emb768_model ON embeddings_768(model, prefix_scheme);