-- ITS Intelligence Hub v2 - Phase 2.6: Extended SINTA Data Migration
-- Run this in Supabase SQL Editor AFTER 003_sinta_integration.sql

-- ============================================================================
-- 1. Add new columns for comprehensive SINTA data
-- ============================================================================

-- Expertise tags extracted from author profile
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS expertise_tags TEXT[] DEFAULT '{}';

-- Researches from SINTA (research projects)
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_researches JSONB DEFAULT '[]';

-- Community services
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_services JSONB DEFAULT '[]';

-- Intellectual Property Rights
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_iprs JSONB DEFAULT '[]';

-- Books authored
ALTER TABLE lecturers
ADD COLUMN IF NOT EXISTS sinta_books JSONB DEFAULT '[]';

-- ============================================================================
-- 2. Indexes for new fields
-- ============================================================================

-- GIN index for expertise tag search
CREATE INDEX IF NOT EXISTS idx_lecturers_expertise_tags 
ON lecturers USING GIN (expertise_tags);

-- ============================================================================
-- 3. Update upsert function to include new fields
-- ============================================================================

CREATE OR REPLACE FUNCTION upsert_sinta_lecturer(
    p_sinta_id TEXT,
    p_name TEXT,
    p_nidn TEXT DEFAULT NULL,
    p_dept TEXT DEFAULT NULL,
    p_h_index_scopus INTEGER DEFAULT 0,
    p_h_index_gscholar INTEGER DEFAULT 0,
    p_publications JSONB DEFAULT '[]'::jsonb,
    p_expertise_tags TEXT[] DEFAULT '{}',
    p_researches JSONB DEFAULT '[]'::jsonb,
    p_services JSONB DEFAULT '[]'::jsonb,
    p_iprs JSONB DEFAULT '[]'::jsonb,
    p_books JSONB DEFAULT '[]'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_lecturer_id UUID;
BEGIN
    INSERT INTO lecturers (
        sinta_id, name, nidn, dept, 
        h_index_scopus, h_index_gscholar, sinta_publications,
        expertise_tags, sinta_researches, sinta_services, sinta_iprs, sinta_books,
        sinta_scraped_at
    )
    VALUES (
        p_sinta_id, p_name, p_nidn, p_dept,
        p_h_index_scopus, p_h_index_gscholar, p_publications,
        p_expertise_tags, p_researches, p_services, p_iprs, p_books,
        NOW()
    )
    ON CONFLICT (sinta_id) DO UPDATE SET
        name = EXCLUDED.name,
        nidn = COALESCE(EXCLUDED.nidn, lecturers.nidn),
        dept = COALESCE(EXCLUDED.dept, lecturers.dept),
        h_index_scopus = EXCLUDED.h_index_scopus,
        h_index_gscholar = EXCLUDED.h_index_gscholar,
        sinta_publications = EXCLUDED.sinta_publications,
        expertise_tags = EXCLUDED.expertise_tags,
        sinta_researches = EXCLUDED.sinta_researches,
        sinta_services = EXCLUDED.sinta_services,
        sinta_iprs = EXCLUDED.sinta_iprs,
        sinta_books = EXCLUDED.sinta_books,
        sinta_scraped_at = NOW(),
        last_updated = NOW()
    RETURNING id INTO v_lecturer_id;
    
    RETURN v_lecturer_id;
END;
$$;

-- ============================================================================
-- Verification
-- ============================================================================
-- SELECT column_name, data_type FROM information_schema.columns 
-- WHERE table_name = 'lecturers' ORDER BY ordinal_position;
