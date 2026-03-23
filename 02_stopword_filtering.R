# ============================================================
# 02_stopword_filtering.R
# Procedural Keyword Filtering
# ============================================================
# Removes speeches containing procedural/parliamentary
# boilerplate language that carries no semantic content.
# Final corpus after this step: 29,860 speeches
# ============================================================

library(readr)
library(dplyr)
library(stringr)

# --- Load Data ---
# Output from 01_data_cleaning.R
akp_filtered <- read_csv("data/processed/akp_speeches_50words_plus.csv")

cat("Loaded speech count:", nrow(akp_filtered), "\n\n")

# --- Step 1: Author-defined Procedural Keywords ---
# Keywords identified through manual inspection of 1,000 speeches
# These are parliamentary procedural phrases with no nostalgic content
procedural_keywords <- c(
  "yoklama",                            # Roll call
  "dakika süre veriyorum",              # "I give X minutes speaking time"
  "kabul edenler, kabul etmeyenler",    # "Those in favour, those against"
  "iç tüzük",                           # Rules of procedure
  "birleşiminin ve oturum",             # Session and sitting references
  "önerge geri çekilmiştir",            # "Motion has been withdrawn"
  "esas numaralı kanun teklifi",        # Bill reference numbers
  "süreniz beş dakika",                 # "Your time is five minutes"
  "buyurun",                            # "Please proceed" (floor yielding)
  "kararnamelerde",                     # In decrees
  "komisyon önergeye katılıyor mu ?",   # "Does the committee support the motion?"
  "gerekçeyi okutuyorum",               # "I am reading the justification"
  "oylarınıza sunuyorum",               # "I submit to your vote"
  "soru-cevap(birleşik halde)",         # Q&A combined format
  "tasarı metninin çıkarılmasını",      # Removal of bill text
  "ant içmemiş"                         # "Has not taken the oath"
)

# --- Step 2: Extended Procedural Keyword List ---
# Combined list: author-defined + general parliamentary formulae
procedural_keywords_full <- unique(c(
  procedural_keywords,
  # Voting and decision formulae
  "kabul edildi", "oy birliği", "yüksek oy", "birleşimi kapatıyorum",
  "tutanağa geçmiştir", "genel kurul", "gündem maddesi",
  "kanun teklifi", "kanun tasarısı", "önerge", "başkanlık divanı",
  "sözlü soru", "yazılı soru", "reddedenler", "ret oyu", "kabul oyu",
  "görüşmelere geçiyoruz", "kanunlaşmıştır", "reddedilmiştir",
  "kabul edilmiştir", "komisyon", "oylamaya sunuyorum", "tasarı",
  "rapor", "tutanak", "parlamento", "meclis genel kurulu",
  "kürsü", "birleşime ara veriyorum"
))

# Build regex pattern (case-insensitive)
procedural_pattern <- paste0("(?i)\\b(", paste(procedural_keywords_full, collapse = "|"), ")\\b")

# --- Apply Filtering ---
akp_clean <- akp_filtered %>%
  filter(!str_detect(text, procedural_pattern))

cat("After procedural keyword filtering:", nrow(akp_clean), "speeches retained\n\n")

# --- Save Output ---
readr::write_csv(akp_clean, "data/processed/akp_speeches_clean.csv")
cat("Clean dataset saved to: data/processed/akp_speeches_clean.csv\n")
