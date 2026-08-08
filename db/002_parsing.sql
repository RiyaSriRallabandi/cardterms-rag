-- Parsing metadata on documents.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_at           TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser              TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS char_count          INT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_page_count      INT DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS table_count         INT DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS product_name_source TEXT;

-- Page boundaries within documents.raw_text.
-- Character offsets let a chunk report the page its text came from, which is
-- what makes citations resolvable to a physical page.
CREATE TABLE IF NOT EXISTS document_pages (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INT NOT NULL,
    char_start  INT NOT NULL,
    char_end    INT NOT NULL,
    text        TEXT NOT NULL,
    char_count  INT NOT NULL,
    ocr_applied BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_regions INT NOT NULL DEFAULT 0,
    UNIQUE (doc_id, page_no)
);

CREATE INDEX IF NOT EXISTS idx_document_pages_doc ON document_pages(doc_id);

-- Table regions within documents.raw_text.
-- Recorded so that chunking can treat a table as an indivisible unit; splitting
-- a fee table across a chunk boundary separates values from their row labels.
CREATE TABLE IF NOT EXISTS document_tables (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INT NOT NULL,
    table_index INT NOT NULL,
    char_start  INT NOT NULL,
    char_end    INT NOT NULL,
    n_rows      INT NOT NULL,
    n_cols      INT NOT NULL,
    empty_cells      INT NOT NULL DEFAULT 0,
    ocr_cells_filled INT NOT NULL DEFAULT 0,
    UNIQUE (doc_id, page_no, table_index)
);

CREATE INDEX IF NOT EXISTS idx_document_tables_doc ON document_tables(doc_id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS filename_product TEXT;
UPDATE documents SET filename_product = product_name WHERE filename_product IS NULL;

-- Columns added after these tables were first created. CREATE TABLE IF NOT
-- EXISTS does not modify an existing table, so new columns need explicit
-- ALTER statements to apply to databases created by an earlier run.
ALTER TABLE document_pages  ADD COLUMN IF NOT EXISTS ocr_regions      INT NOT NULL DEFAULT 0;
ALTER TABLE document_tables ADD COLUMN IF NOT EXISTS empty_cells      INT NOT NULL DEFAULT 0;
ALTER TABLE document_tables ADD COLUMN IF NOT EXISTS ocr_cells_filled INT NOT NULL DEFAULT 0;