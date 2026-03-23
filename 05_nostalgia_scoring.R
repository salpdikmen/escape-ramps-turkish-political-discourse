# ============================================================
# 05_nostalgia_scoring.R
# Nostalgia Score Calculation (Müller & Proksch 2024 Method)
# ============================================================
# Applies the strong nostalgia dictionary to the lemmatised
# corpus to produce a normalised nostalgia score per speech.
#
# Normalisation formula (Müller & Proksch 2024):
#   nostalgia_score = (nostalgia_hits / word_count) * 1000
#
# This yields a comparable rate per 1,000 words across
# speeches of varying lengths.
# ============================================================

rm(list = ls())

library(readr)
library(readxl)
library(stringr)
library(dplyr)
library(quanteda)
library(lubridate)
library(writexl)
library(ggplot2)

# --- Load Dictionary ---
# The strong nostalgia dictionary (26 collocations)
# Produced in step 04_ngram_dictionary.R
colloc_df  <- read_excel("data/dictionaries/strong_dictionary_26.xlsx")
dict_terms <- str_replace_all(colloc_df$collocation, " ", "_")
nostalgia_dict <- dictionary(list(nostalgia = dict_terms))

# --- Load Tokenised Corpus ---
data <- read_csv("data/processed/akp_speeches_tokenised.csv")

# Remove empty rows
data <- data %>%
  filter(!is.na(lemma_text_cleaned) & str_trim(lemma_text_cleaned) != "")

# --- Build Corpus ---
corp <- corpus(data, text_field = "lemma_text_cleaned", docid_field = "text_ID")

# --- Tokenise (1–3 grams) ---
toks <- tokens(corp, what = "word", remove_punct = TRUE, remove_numbers = TRUE)

# Build 2-gram and 3-gram token objects
toks_2grams <- tokens_ngrams(toks, n = 2, concatenator = "_")
dfm_2gram   <- dfm(toks_2grams)

toks_3grams <- tokens_ngrams(toks, n = 3, concatenator = "_")
dfm_3gram   <- dfm(toks_3grams)

# --- Align DFMs ---
# Both DFMs must share the same documents before combining
common_docs       <- intersect(docnames(dfm_2gram), docnames(dfm_3gram))
dfm_2gram_aligned <- dfm_subset(dfm_2gram, docnames(dfm_2gram) %in% common_docs)[common_docs, ]
dfm_3gram_aligned <- dfm_subset(dfm_3gram, docnames(dfm_3gram) %in% common_docs)[common_docs, ]

# Combine 2-gram and 3-gram DFMs
dfm_combined <- cbind(dfm_2gram_aligned, dfm_3gram_aligned)

# --- Dictionary Lookup ---
dfm_nostalgia <- dfm_lookup(dfm_combined, dictionary = nostalgia_dict)
nostalgia_counts <- convert(dfm_nostalgia, to = "data.frame")

# --- Merge with Original Data ---
merged_df <- left_join(data, nostalgia_counts, by = c("text_ID" = "doc_id"))

# --- Save Raw Scores ---
write_csv(merged_df, "data/processed/nostalgia_scores_raw.csv")
cat("Raw nostalgia scores saved to: data/processed/nostalgia_scores_raw.csv\n")

# --- Normalised Nostalgia Score (Müller & Proksch 2024) ---
merged_df <- merged_df %>%
  mutate(nostalgia_score = (nostalgia / word_count) * 1000)

# Inspect
head(merged_df %>% select(text_ID, word_count, nostalgia, nostalgia_score))

# --- Quarterly Aggregation ---
quarterly_nostalgia <- merged_df %>%
  mutate(
    year    = as.numeric(format(Date, "%Y")),
    quarter = paste0(year, "-Q", ceiling(as.numeric(format(Date, "%m")) / 3))
  ) %>%
  group_by(quarter) %>%
  summarise(
    avg_nostalgia_score  = mean(nostalgia_score, na.rm = TRUE),
    median_nostalgia_score = median(nostalgia_score, na.rm = TRUE),
    total_speeches       = n(),
    total_nostalgia_hits = sum(nostalgia, na.rm = TRUE),
    avg_speech_length    = mean(word_count, na.rm = TRUE)
  ) %>%
  arrange(quarter)

# Save quarterly summary
write_xlsx(quarterly_nostalgia, "data/processed/quarterly_nostalgia.xlsx")
cat("Quarterly nostalgia summary saved to: data/processed/quarterly_nostalgia.xlsx\n")

# --- Trend Plot ---
quarterly_nostalgia <- quarterly_nostalgia %>%
  mutate(
    year_num    = as.numeric(substr(quarter, 1, 4)),
    quarter_num = as.numeric(substr(quarter, 7, 7)),
    date_plot   = as.Date(paste0(year_num, "-", (quarter_num - 1) * 3 + 1, "-01"))
  )

ggplot(quarterly_nostalgia, aes(x = date_plot, y = avg_nostalgia_score)) +
  geom_line(colour = "#2c3e50", linewidth = 0.8) +
  geom_point(colour = "#2c3e50", size = 2) +
  labs(
    title    = "AKP Nostalgic Discourse — Quarterly Trend (2011–2022)",
    subtitle = "Strong Dictionary (26 collocations) | Normalised per 1,000 words",
    x        = "Quarter",
    y        = "Average Nostalgia Score"
  ) +
  theme_minimal() +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

ggsave("outputs/figures/nostalgia_quarterly_trend.png", width = 12, height = 6, dpi = 150)
cat("Trend plot saved to: outputs/figures/nostalgia_quarterly_trend.png\n")
