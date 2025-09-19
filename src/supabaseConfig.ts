// supabaseConfig.js
export const supabaseUrl = "https://qjhxkdiasyfvykkqjwsu.supabase.co";
export const supabaseAnonKey = 
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." +
  "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFqa" +
  "HhrZGlhc3lmdnlra3Fqd3N1Iiwicm9sZSI6Im" +
  "Fub24iLCJpYXQiOjE3NTgwMzgzMzIsImV4cCI" +
  "6MjA3MzYxNDMzMn0.1m3tVEQnli_35sSXMC7b" +
  "coGo04derANMuRe2VCTcyL8";

import { createClient } from '@supabase/supabase-js';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

