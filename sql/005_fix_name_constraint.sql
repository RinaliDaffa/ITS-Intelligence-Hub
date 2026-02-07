-- Fix: Drop unique constraint on name to allow duplicate names
-- (SINTA has multiple lecturers with the same name like SUPRAPTO)

-- Drop the unique constraint on name
ALTER TABLE lecturers DROP CONSTRAINT IF EXISTS lecturers_name_key;

-- Verify sinta_id is still unique (this is the correct unique identifier)
-- Already exists from 003_sinta_integration.sql
