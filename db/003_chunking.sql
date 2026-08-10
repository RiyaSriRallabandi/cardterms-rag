-- Parent chunks are stored alongside their children in the same chunk set.
-- Retrieval matches against children and expands to parents, so the two must
-- be distinguishable.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_parent BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_chunks_set_parent ON chunks(chunk_set_id, is_parent);