-- ITS Intelligence Hub v2 - Database Schema
-- Run this in Supabase SQL Editor

-- Research papers table
CREATE TABLE IF NOT EXISTS researches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    abstract TEXT,
    author VARCHAR(500) NOT NULL,
    supervisors TEXT[],           -- Array of supervisor names (optional)
    degree VARCHAR(50) NOT NULL,  -- S1, S2, S3
    year INTEGER,
    dept VARCHAR(200),
    url TEXT UNIQUE NOT NULL,     -- Unique constraint for upserts
    data_confidence FLOAT,        -- 0.0 to 1.0 quality score
    metadata JSONB DEFAULT '{}',  -- Flexible additional data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_researches_degree ON researches(degree);
CREATE INDEX IF NOT EXISTS idx_researches_year ON researches(year);
CREATE INDEX IF NOT EXISTS idx_researches_dept ON researches(dept);
CREATE INDEX IF NOT EXISTS idx_researches_confidence ON researches(data_confidence);

-- Full-text search index on title and abstract
CREATE INDEX IF NOT EXISTS idx_researches_fts ON researches 
USING gin(to_tsvector('indonesian', coalesce(title, '') || ' ' || coalesce(abstract, '')));

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_researches_updated_at ON researches;
CREATE TRIGGER update_researches_updated_at
    BEFORE UPDATE ON researches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (optional, adjust as needed)
-- ALTER TABLE researches ENABLE ROW LEVEL SECURITY;
