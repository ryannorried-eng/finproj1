-- Add alpha scoring columns to rec_snapshots for CLV-by-alpha analytics.
ALTER TABLE rec_snapshots ADD COLUMN alpha_score INTEGER;
ALTER TABLE rec_snapshots ADD COLUMN alpha_label TEXT;
