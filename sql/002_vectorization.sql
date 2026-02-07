-- ITS Intelligence Hub v2 - Phase 2: Vectorization Migration
-- Run this in Supabase SQL Editor AFTER the base schema

-- ============================================================================
-- 1. Enable pgvector Extension
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 2. Add embedding column to researches table
-- ============================================================================
ALTER TABLE researches 
ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Index for tracking unprocessed records
CREATE INDEX IF NOT EXISTS idx_researches_embedding_null 
ON researches ((embedding IS NULL)) 
WHERE embedding IS NULL;

-- ============================================================================
-- 3. Create lecturers table for Expertise Profiling
-- ============================================================================
CREATE TABLE IF NOT EXISTS lecturers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    dept TEXT,
    research_count INTEGER DEFAULT 0,
    expertise_vector vector(768),      -- Centroid of supervised research embeddings
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'        -- Additional info (email, titles, etc.)
);

-- Indexes for lecturers
CREATE INDEX IF NOT EXISTS idx_lecturers_name ON lecturers(name);
CREATE INDEX IF NOT EXISTS idx_lecturers_dept ON lecturers(dept);
CREATE INDEX IF NOT EXISTS idx_lecturers_research_count ON lecturers(research_count DESC);

-- ============================================================================
-- 4. HNSW Index for Fast Similarity Search on Researches
-- ============================================================================
-- HNSW provides faster queries than IVFFlat at slightly higher memory cost
-- Using cosine distance (vector_cosine_ops) for normalized embeddings

CREATE INDEX IF NOT EXISTS idx_researches_embedding_hnsw 
ON researches 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 5. HNSW Index for Lecturer Expertise Vectors
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_lecturers_expertise_hnsw 
ON lecturers 
USING hnsw (expertise_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 6. Similarity Search Function (RPC)
-- ============================================================================
-- Search for research papers by semantic similarity
-- Returns papers sorted by cosine similarity (1 - distance)

CREATE OR REPLACE FUNCTION match_research(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    abstract TEXT,
    author VARCHAR(500),
    supervisors TEXT[],
    degree VARCHAR(50),
    year INTEGER,
    dept VARCHAR(200),
    url TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.title,
        r.abstract,
        r.author,
        r.supervisors,
        r.degree,
        r.year,
        r.dept,
        r.url,
        1 - (r.embedding <=> query_embedding) AS similarity
    FROM researches r
    WHERE r.embedding IS NOT NULL
      AND 1 - (r.embedding <=> query_embedding) >= match_threshold
    ORDER BY r.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 7. Find Similar Lecturers Function (RPC)
-- ============================================================================
-- Search for lecturers with similar expertise profiles

CREATE OR REPLACE FUNCTION match_lecturers(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    dept TEXT,
    research_count INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        l.id,
        l.name,
        l.dept,
        l.research_count,
        1 - (l.expertise_vector <=> query_embedding) AS similarity
    FROM lecturers l
    WHERE l.expertise_vector IS NOT NULL
      AND 1 - (l.expertise_vector <=> query_embedding) >= match_threshold
    ORDER BY l.expertise_vector <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- 8. Helper: Get Embedding Statistics
-- ============================================================================
CREATE OR REPLACE FUNCTION get_embedding_stats()
RETURNS TABLE (
    total_researches BIGINT,
    embedded_count BIGINT,
    pending_count BIGINT,
    total_lecturers BIGINT,
    profiled_lecturers BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM researches)::BIGINT AS total_researches,
        (SELECT COUNT(*) FROM researches WHERE embedding IS NOT NULL)::BIGINT AS embedded_count,
        (SELECT COUNT(*) FROM researches WHERE embedding IS NULL)::BIGINT AS pending_count,
        (SELECT COUNT(*) FROM lecturers)::BIGINT AS total_lecturers,
        (SELECT COUNT(*) FROM lecturers WHERE expertise_vector IS NOT NULL)::BIGINT AS profiled_lecturers;
END;
$$;
