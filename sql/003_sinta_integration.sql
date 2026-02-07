-- ITS Intelligence Hub v2 - Phase 2.5: SINTA Integration Migration
-- Run this in Supabase SQL Editor AFTER 002_vectorization.sql

-- ============================================================================
-- 1. Extend lecturers table with SINTA-specific fields
-- ============================================================================

-- SINTA Author ID (unique identifier from sinta.kemdiktisaintek.go.id)
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_id TEXT UNIQUE;

-- NIDN (Nomor Induk Dosen Nasional) - National Lecturer ID
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS nidn TEXT;

-- SINTA H-Index scores
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS h_index_scopus INTEGER DEFAULT 0;

ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS h_index_gscholar INTEGER DEFAULT 0;

-- Publication titles from Scopus and Google Scholar
-- Stores array of publication objects: [{"title": "...", "source": "scopus|gscholar", "year": 2024}, ...]
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_publications JSONB DEFAULT '[]'::jsonb;

-- Timestamp of last SINTA scrape
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_scraped_at TIMESTAMP WITH TIME ZONE;

-- ============================================================================
-- 2. Indexes for SINTA fields
-- ============================================================================

-- Fast lookup by SINTA ID
CREATE INDEX IF NOT EXISTS idx_lecturers_sinta_id 
ON lecturers(sinta_id) 
WHERE sinta_id IS NOT NULL;

-- Fast lookup by NIDN
CREATE INDEX IF NOT EXISTS idx_lecturers_nidn 
ON lecturers(nidn) 
WHERE nidn IS NOT NULL;

-- Index for finding lecturers needing vectorization
CREATE INDEX IF NOT EXISTS idx_lecturers_sinta_needs_vector 
ON lecturers ((expertise_vector IS NULL))
WHERE sinta_publications != '[]'::jsonb AND expertise_vector IS NULL;

-- ============================================================================
-- 3. Helper function: Get SINTA scrape statistics
-- ============================================================================

CREATE OR REPLACE FUNCTION get_sinta_stats()
RETURNS TABLE (
    total_lecturers BIGINT,
    with_sinta_id BIGINT,
    with_publications BIGINT,
    with_expertise_vector BIGINT,
    pending_vectorization BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*) FROM lecturers)::BIGINT AS total_lecturers,
        (SELECT COUNT(*) FROM lecturers WHERE sinta_id IS NOT NULL)::BIGINT AS with_sinta_id,
        (SELECT COUNT(*) FROM lecturers WHERE sinta_publications != '[]'::jsonb)::BIGINT AS with_publications,
        (SELECT COUNT(*) FROM lecturers WHERE expertise_vector IS NOT NULL)::BIGINT AS with_expertise_vector,
        (SELECT COUNT(*) FROM lecturers 
         WHERE sinta_publications != '[]'::jsonb 
         AND expertise_vector IS NULL)::BIGINT AS pending_vectorization;
END;
$$;

-- ============================================================================
-- 4. Upsert function for SINTA data (RPC callable from Python)
-- ============================================================================

CREATE OR REPLACE FUNCTION upsert_sinta_lecturer(
    p_sinta_id TEXT,
    p_name TEXT,
    p_nidn TEXT DEFAULT NULL,
    p_dept TEXT DEFAULT NULL,
    p_h_index_scopus INTEGER DEFAULT 0,
    p_h_index_gscholar INTEGER DEFAULT 0,
    p_publications JSONB DEFAULT '[]'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_lecturer_id UUID;
BEGIN
    INSERT INTO lecturers (sinta_id, name, nidn, dept, h_index_scopus, h_index_gscholar, sinta_publications, sinta_scraped_at)
    VALUES (p_sinta_id, p_name, p_nidn, p_dept, p_h_index_scopus, p_h_index_gscholar, p_publications, NOW())
    ON CONFLICT (sinta_id) DO UPDATE SET
        name = EXCLUDED.name,
        nidn = COALESCE(EXCLUDED.nidn, lecturers.nidn),
        dept = COALESCE(EXCLUDED.dept, lecturers.dept),
        h_index_scopus = EXCLUDED.h_index_scopus,
        h_index_gscholar = EXCLUDED.h_index_gscholar,
        sinta_publications = EXCLUDED.sinta_publications,
        sinta_scraped_at = NOW(),
        last_updated = NOW()
    RETURNING id INTO v_lecturer_id;
    
    RETURN v_lecturer_id;
END;
$$;

-- ============================================================================
-- 5. Grant permissions (if using Row Level Security)
-- ============================================================================
-- Uncomment if RLS is enabled:
-- GRANT EXECUTE ON FUNCTION get_sinta_stats() TO anon, authenticated;
-- GRANT EXECUTE ON FUNCTION upsert_sinta_lecturer(TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, JSONB) TO authenticated;

-- ============================================================================
-- Verification query (run after migration)
-- ============================================================================
-- SELECT column_name, data_type FROM information_schema.columns 
-- WHERE table_name = 'lecturers' ORDER BY ordinal_position;
