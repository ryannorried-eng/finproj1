ALTER TABLE rec_snapshots ADD COLUMN run_id TEXT;
CREATE INDEX IF NOT EXISTS idx_rec_snap_run_id ON rec_snapshots (run_id);
