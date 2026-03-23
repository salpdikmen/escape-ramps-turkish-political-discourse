# ============================================================
# 06_validation.R
# Dictionary Validation — Stratified Sample Preparation
# ============================================================
# Produces stratified validation samples for manual coding.
# 500 sentences manually coded by the author for the
# Strong dictionary (250 nostalgic / 250 non-nostalgic).
#
# Results presented in increments of 100 to confirm the
# plateau effect in classification metrics.
#
# Final performance (Strong Dictionary, n=500):
#   Accuracy:  87.6%
#   Precision: 88.4%
#   Recall:    87.0%
#   F1:        87.7%
# ============================================================

library(dplyr)
library(readr)
library(writexl)

# --- Load Scored Data ---
nostalgia_data <- read_csv("data/processed/nostalgia_scores_raw.csv")

# Preview
head(nostalgia_data)
colnames(nostalgia_data)

# --- Set Seed for Reproducibility ---
set.seed(42)

# --- Generate 10 Small Validation Sets (25 positive + 25 negative each) ---
# Used for incremental manual coding and early-stage reliability checks
for (i in 1:10) {
  validation_set <- bind_rows(
    sample_n(filter(nostalgia_data, nostalgia == 1), 25),
    sample_n(filter(nostalgia_data, nostalgia == 0), 25)
  ) %>%
    sample_frac(1) %>%       # Shuffle rows
    select(text, nostalgia)  # Only text and label for blind coding

  write_xlsx(
    validation_set,
    path = file.path("data/validation", paste0("validation_set_strong_", i, ".xlsx"))
  )
}

cat("10 small validation sets (25+25) saved to: data/validation/\n")

# --- Generate Main Validation Set (250 positive + 250 negative) ---
# This 500-sentence set was hand-coded by the author sentence by sentence.
# Results were recorded in increments of 100 to demonstrate metric stability.
validation_set_large <- bind_rows(
  sample_n(filter(nostalgia_data, nostalgia == 1), 250),
  sample_n(filter(nostalgia_data, nostalgia == 0), 250)
) %>%
  sample_frac(1) %>%
  select(text, nostalgia)

write_xlsx(
  validation_set_large,
  path = "data/validation/validation_set_strong_250.xlsx"
)

cat("Main validation set (250+250) saved to: data/validation/validation_set_strong_250.xlsx\n")

# ============================================================
# INCREMENTAL CONFUSION MATRIX RESULTS (Manually Coded)
# ============================================================
# Rows | TP  | TN  | FP | FN | Accuracy | Precision | Recall | F1
# 100  |  47 |  41 |  7 |  5 |  0.880   |   0.870   | 0.904  | 0.887
# 200  |  85 |  89 | 12 | 14 |  0.870   |   0.876   | 0.859  | 0.867
# 300  | 132 | 126 | 21 | 21 |  0.860   |   0.863   | 0.863  | 0.863
# 400  | 179 | 168 | 24 | 29 |  0.868   |   0.882   | 0.861  | 0.871
# 500  | 221 | 217 | 29 | 33 |  0.876   |   0.884   | 0.870  | 0.877
#
# Metrics plateau from row 200 onwards, confirming sample sufficiency.
# ============================================================
