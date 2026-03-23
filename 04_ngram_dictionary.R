# ============================================================
# 04_ngram_dictionary.R
# N-gram Collocation Extraction & Nostalgia Dictionary Construction
# ============================================================
# Builds the nostalgia collocation dictionary from the
# lemmatised and cleaned corpus using Quanteda.
#
# Three dictionaries were constructed through iterative refinement:
#   - Weak   (283 collocations): Lambda ≥ 3, Z ≥ 3, count ≥ 5
#   - Medium  (76 collocations): Lambda ≥ 4, Z ≥ 4, count ≥ 8
#   - Strong  (26 collocations): Lambda ≥ 5, Z ≥ 5, count ≥ 10
#
# The Strong dictionary was selected for final analysis.
# ============================================================

library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(readr)
library(dplyr)
library(stopwords)
library(writexl)

# --- Load Lemmatised Data ---
data <- read_csv("data/processed/akp_speeches_lemmatised.csv")

print("First rows of lemmatised data:")
print(head(data))
print("Column names:")
print(names(data))

# --- Build Corpus ---
corp <- corpus(data, text_field = "lemma_text", docid_field = "text_ID")

# --- Define Stop Words ---
# Turkish stop words from the stopwords-iso source
tr_stopwords_iso <- stopwords::stopwords("tr", source = "stopwords-iso")

# Additional domain-specific stop words (semantically void in parliamentary context)
additional_stopwords <- c(
  "olmak", "gelmek", "etmek", "yapmak", "demek", "bilmek", "bulmak", "geçmek",
  "kalmak", "bakmak", "durmak", "görmek", "çıkmak", "girmek", "almak", "vermek",
  "koymak", "sürmek", "yaşamak", "ulaşmak", "göstermek", "çözmek", "tartışmak",
  "beklemek", "bulunmak", "yürütmek", "çalışmak", "istemek", "gerek", "meclis",
  "millet", "ülke", "başkan", "saygıdeğer", "sayın", "konuşmak", "kurtulmak",
  "yön", "halk", "konu", "adres", "durum", "soru", "siyasi", "parti", "genel",
  "kurum", "değerli", "arkadaş", "birlik", "dünya", "tarih", "hukuk", "demokrasi",
  "yeni", "sistem", "döne", "gün", "ayrı", "büyük", "küçük", "kendi", "bütün",
  "tüm", "hep", "hiç", "birçok", "pekçok", "bazı", "çok", "daha", "iyi", "kötü",
  "önce", "sonra", "aşağı", "yukarı", "sol", "sağ", "alt", "üst", "iç", "dış",
  "artık", "zaten", "hal", "halihazırda", "şimdi", "gibi", "ile", "için", "yani",
  "oysa", "ancak", "çünkü", "hatta", "böyle", "öyle", "şöyle", "tek", "iki", "üç",
  "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "yüz", "bin", "milyon",
  "milyar", "trilyon", "yıl", "âmâ", "bugün", "önemli", "zaman", "teşekkür", "karşı"
)

# Note: Nostalgia-sensitive terms deliberately excluded from stop word list
# (e.g., "kriz", "anayasa", "ekonomik") to avoid inadvertently removing context

# Extended stop words: economic, political, procedural terms
# (excluded to avoid semantic noise in nostalgia scoring)
nostalgia_sensitive <- c(
  "kriz", "anayasa", "mahkeme", "yasa", "hukuk", "demokratik", "politik", "ekonomik",
  "geliştirmek", "kalkınmak", "yatırım", "gelir", "ücret", "başarı", "devlet",
  "bugün", "şimdi", "artık", "önem", "özellikle", "gerçekten"
)

extended_stopwords <- setdiff(c(
  # Economic/financial terms
  "bütçe", "ekonomi", "yatırım", "proje", "personel", "kredi", "faiz",
  "ticaret", "ihracat", "üretim", "sanayi", "gelir", "ücret",
  "vergi", "borç", "lira", "tl", "fiyat", "piyasa", "banka", "ihale",
  "hesap", "oran", "rakam", "sayı", "toplam", "artmak", "yükselmek",
  "büyümek", "geliştirmek", "artırmak", "üretmek", "istihdam", "sektör",
  "kur", "yatır", "harcamak", "ödemek", "kazanmak",
  # Political/institutional terms
  "milletvekili", "bakan", "başbakan", "cumhurbaşkanı", "parti̇", "grup",
  "kurul", "komisyon", "vekil", "grub", "belediye", "muhalefet",
  "iktidar", "politika", "siyasî", "siyaset", "seçi", "oy", "chp",
  # Legal/procedural terms
  "kânun", "kanu", "madde", "düzenlemek", "uygulamak", "karar",
  "mahkeme", "yargı", "ceza", "suç", "hüküm", "teklif", "önerge",
  "belirlemek", "düzenleme", "işlem", "faaliyet", "süreç", "usul", "yetki",
  "sorumluluk", "idare", "yönet", "hakk",
  "savcı", "polis", "soruşturmak", "kovuşturmak",
  # Generic/spatial terms
  "bura", "ora", "aynı", "ön", "son", "önce", "sonra",
  "parti", "ifade", "şekil", "söz", "konu", "anlam", "durum",
  "nokta", "husus", "mesele", "taraf", "kapsam", "çerçeve", "içeri", "yönelik",
  "ilişkin", "sonuç", "yaklaşım", "olay", "seçim", "aday", "görüş", "görüşme",
  "netice", "öneri", "dilmek", "getirmek", "vermek", "almak", "sunmak",
  "yapmak", "etmek", "açmak", "gitmek", "olmak", "başlamak", "anlatmak",
  "söylemek", "kalkmak", "çıkmak", "girmek", "tutmak", "taşımak", "çalışmak",
  "istemek", "kullanmak", "yürütmek", "devam",
  # Formal address terms
  "saygı", "sayın", "değerli", "selamlamak", "alkışlamak", "kutlamak", "vesile",
  "tabiî", "bey", "il", "ilçe"
), nostalgia_sensitive)

# Combine all stop words
tr_stopwords_final <- unique(c(tr_stopwords_iso, additional_stopwords, extended_stopwords))

# --- Tokenise ---
toks <- tokens(corp, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(pattern = tr_stopwords_final, valuetype = "fixed")

cat("Total tokens after cleaning:", sum(ntoken(toks)), "\n")

# Save cleaned token representation to corpus
data$lemma_text_cleaned <- sapply(toks, function(x) paste(x, collapse = " "))
write_csv(data, "data/processed/akp_speeches_tokenised.csv")
cat("Tokenised dataset saved to: data/processed/akp_speeches_tokenised.csv\n")

# --- Top 500 Features (Diagnostic) ---
dfm_for_freq <- dfm(toks)
top_features <- textstat_frequency(dfm_for_freq, n = 500)
print("Top 500 most frequent tokens after cleaning:")
print(top_features)

# ============================================================
# N-GRAM COLLOCATION EXTRACTION
# ============================================================

# 2-gram collocations (min count = 5)
colloc_2 <- textstat_collocations(toks, size = 2, min_count = 5)
colloc_2$size <- 2

# 3-gram collocations (min count = 3)
colloc_3 <- textstat_collocations(toks, size = 3, min_count = 3)
colloc_3$size <- 3

# 4-gram collocations (min count = 2)
colloc_4 <- textstat_collocations(toks, size = 4, min_count = 2)
colloc_4$size <- 4

# Select relevant columns
colloc_2_simple <- colloc_2[, c("collocation", "count", "lambda", "z", "size")]
colloc_3_simple <- colloc_3[, c("collocation", "count", "lambda", "z", "size")]
colloc_4_simple <- colloc_4[, c("collocation", "count", "lambda", "z", "size")]

# Combine all n-grams
all_collocs <- rbind(colloc_2_simple, colloc_3_simple, colloc_4_simple)

# Save full collocation list for manual dictionary review
write_xlsx(all_collocs, path = "data/dictionaries/collocation_ngrams_full.xlsx")
cat("Full collocation list saved to: data/dictionaries/collocation_ngrams_full.xlsx\n")

# ============================================================
# DICTIONARY FILTERING THRESHOLDS
# ============================================================
# After manual review of the full collocation list,
# three dictionaries were produced at different thresholds:

# Weak Dictionary   (283 collocations): Lambda >= 3, Z >= 3, count >= 5
# Medium Dictionary  (76 collocations): Lambda >= 4, Z >= 4, count >= 8
# Strong Dictionary  (26 collocations): Lambda >= 5, Z >= 5, count >= 10  <-- Used in final analysis

# Apply Strong threshold as example:
strong_dict <- all_collocs %>%
  filter(lambda >= 5, z >= 5, count >= 10)

cat("Strong dictionary: ", nrow(strong_dict), "collocations\n")
write_xlsx(strong_dict, path = "data/dictionaries/strong_dictionary_26.xlsx")
cat("Strong dictionary saved to: data/dictionaries/strong_dictionary_26.xlsx\n")
