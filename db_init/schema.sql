-- Raw station status data from GBFS API
CREATE TABLE IF NOT EXISTS station_status_raw (
    ts         TIMESTAMPTZ NOT NULL,    -- Timestamp of the status snapshot
    json_doc   JSONB NOT NULL           -- Raw GBFS response
);

-- Create time-based partitioning for efficient historical queries and data retention
CREATE TABLE IF NOT EXISTS station_status_raw_partitioned (
    ts         TIMESTAMPTZ NOT NULL,
    json_doc   JSONB NOT NULL
) PARTITION BY RANGE (ts);

-- Create last 3 months of partitions (adjust ranges as needed for demo)
-- Note: These will fail silently if partition already exists, which is fine
CREATE TABLE IF NOT EXISTS station_status_raw_y2025m08 PARTITION OF station_status_raw_partitioned
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE IF NOT EXISTS station_status_raw_y2025m07 PARTITION OF station_status_raw_partitioned
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE IF NOT EXISTS station_status_raw_y2025m06 PARTITION OF station_status_raw_partitioned
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS ix_station_status_ts ON station_status_raw_partitioned (ts DESC);
-- Index for JSON path operations (helps when querying specific stations)
CREATE INDEX IF NOT EXISTS ix_station_status_jsonb ON station_status_raw_partitioned USING GIN (json_doc);

-- Raw trip data from Citi Bike system
CREATE TABLE IF NOT EXISTS trips_raw (
    ride_id          BIGINT PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ NOT NULL,
    start_station_id INT NOT NULL,
    end_station_id   INT NOT NULL,
    member_type      TEXT NOT NULL,
    -- Add constraints for data quality
    CONSTRAINT valid_times CHECK (ended_at > started_at),
    CONSTRAINT valid_duration CHECK (
        EXTRACT(EPOCH FROM (ended_at - started_at))/60 BETWEEN 1 AND 24*60
    )
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS ix_trips_station_time ON trips_raw (start_station_id, started_at);
CREATE INDEX IF NOT EXISTS ix_trips_end_station ON trips_raw (end_station_id, ended_at);
CREATE INDEX IF NOT EXISTS ix_trips_member_type ON trips_raw (member_type, started_at);

-- Drop and recreate view (views can't use IF NOT EXISTS)
DROP VIEW IF EXISTS trips_features;
CREATE VIEW trips_features AS
SELECT 
    start_station_id,
    date_trunc('hour', started_at) as hour_ts,
    COUNT(*) as trip_count,
    AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60.0) as avg_duration_minutes,
    COUNT(DISTINCT member_type) as unique_member_types
FROM trips_raw
GROUP BY start_station_id, date_trunc('hour', started_at);

-- Drop and recreate materialized view
DROP MATERIALIZED VIEW IF EXISTS hourly_station_stats;
CREATE MATERIALIZED VIEW hourly_station_stats AS
SELECT 
    start_station_id,
    date_trunc('hour', started_at) as hour_ts,
    COUNT(*) as trips,
    AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60.0) as avg_duration,
    COUNT(DISTINCT member_type) as member_types
FROM trips_raw
GROUP BY start_station_id, date_trunc('hour', started_at)
WITH NO DATA;  -- Will be refreshed by the pipeline

-- Index the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS ix_hourly_stats_pk ON hourly_station_stats (start_station_id, hour_ts);
