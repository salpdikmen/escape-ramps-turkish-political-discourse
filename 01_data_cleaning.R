# ============================================================
# 01_data_cleaning.R
# AKP Parliamentary Speeches — Data Cleaning & Filtering
# ============================================================
# Starting corpus: 286,045 AKP speeches (ParlaMint-TR)
# After filtering (50+ words): 29,860 speeches retained
# ============================================================

library(readr)
library(dplyr)
library(stringr)
library(quanteda)

# --- Load Data ---
# Place the raw CSV file in the data/raw/ directory before running
akp_speeches <- read_csv("data/raw/akp_konusmalar.csv")

# Preview the data
head(akp_speeches)
str(akp_speeches)

# --- Step 1: Word Count Filtering ---
# Calculate word count per speech
akp_speeches <- akp_speeches %>%
  mutate(word_count = str_count(text, "\\w+"))

# Filter: keep speeches with 50+ words (remove very short procedural utterances)
# Note: Upper limit (500 words) was tested but removed — longer speeches retained
akp_filtered <- akp_speeches %>%
  filter(word_count >= 50)

cat("Original speech count:", nrow(akp_speeches), "\n")
cat("After filtering (50+ words):", nrow(akp_filtered), "\n\n")

# Optional: Inspect very short speeches (removed)
short_speeches <- akp_speeches %>%
  filter(word_count < 50) %>%
  sample_n(min(10, nrow(.))) %>%
  select(text, word_count)

# Optional: Inspect very long speeches
long_speeches <- akp_speeches %>%
  filter(word_count > 500) %>%
  sample_n(min(10, nrow(.))) %>%
  select(text, word_count)

# --- Save Filtered Dataset ---
readr::write_csv(akp_filtered, "data/processed/akp_speeches_50words_plus.csv")
cat("Filtered dataset saved to: data/processed/akp_speeches_50words_plus.csv\n")

# --- Save a 1000-speech Sample for Manual Inspection ---
# Used to identify procedural keywords for the next step
inspection_sample <- akp_filtered %>%
  sample_n(min(1000, nrow(.)))

readr::write_csv(inspection_sample, "data/processed/inspection_sample_1000.csv")
cat("1000-speech inspection sample saved to: data/processed/inspection_sample_1000.csv\n")
