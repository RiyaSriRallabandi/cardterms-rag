-- Provenance and review state for evaluation questions.
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS source           TEXT;
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS reference_answer TEXT;
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS question_uid     TEXT UNIQUE;

CREATE INDEX IF NOT EXISTS idx_eval_questions_category
    ON eval_questions(category);
CREATE INDEX IF NOT EXISTS idx_eval_labels_doc ON eval_labels(doc_id);