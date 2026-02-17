-- Add recommendation linkage columns to bets table.
ALTER TABLE bets ADD COLUMN source_page TEXT;
ALTER TABLE bets ADD COLUMN recommendation_id TEXT;
ALTER TABLE bets ADD COLUMN rank_at_pick INTEGER;
ALTER TABLE bets ADD COLUMN quality_tier_at_pick TEXT;
ALTER TABLE bets ADD COLUMN edge_pct_at_pick REAL;
ALTER TABLE bets ADD COLUMN consensus_prob_at_pick REAL;
ALTER TABLE bets ADD COLUMN execution_delta_decimal REAL;
