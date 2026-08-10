-- Market segment from the corpus manifest. Recorded per document so that
-- retrieval metrics can be reported by issuer type rather than only in
-- aggregate; the segments differ in document length, template reuse and fee
-- structure, and are expected to differ in retrieval difficulty.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS bucket TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_bucket ON documents(bucket);