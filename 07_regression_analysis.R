# ============================================================
# 07_regression_analysis.R
# Regression Models, Lag Analysis & Structural Break Testing
# ============================================================
# Tests two core hypotheses:
#   H1: Economic deterioration → increase in nostalgic rhetoric
#   H2: Structural shift in nostalgic discourse post-2017
#
# Models:
#   - Baseline OLS regression (all three dictionaries)
#   - VIF multicollinearity diagnostics
#   - 2017 interaction models
#   - Lag regression (0–4 quarters / 0–12 months)
#   - Cross-correlation analysis
#   - (Optional) Granger causality tests
#
# Key result: Inflation at lag 4 (~12 months) is the strongest
# predictor of nostalgic rhetoric (β=0.032, p<0.001, R²=42.2%)
# ============================================================

library(readxl)
library(tidyverse)
library(broom)
library(corrplot)
library(stargazer)
library(ggplot2)
library(lubridate)
library(writexl)

# Create output directory if it does not exist
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables",  showWarnings = FALSE, recursive = TRUE)

# --- Load Economic + Nostalgia Data ---
# The main regression table merges quarterly nostalgia scores
# with economic indicators (TÜİK + CBRT EVDS)
economic_data <- read_excel("data/economic/ANA_REGRESYON_TABLOSU.xlsx", sheet = "3_dic")

# Rename columns (standardise)
names(economic_data) <- c("Period", "Inflation", "Consumer_Confidence", "TRY_USD",
                           "Unemployment", "Nostalgia_Strong", "Nostalgia_Medium", "Nostalgia_Weak")

# Convert to numeric
economic_data <- economic_data %>%
  mutate(across(c(Inflation, Consumer_Confidence, TRY_USD,
                  Unemployment, Nostalgia_Strong, Nostalgia_Medium, Nostalgia_Weak), as.numeric))

# Parse date from period string (e.g. "2015 Q3")
economic_data$Year    <- as.numeric(substr(economic_data$Period, 1, 4))
economic_data$Quarter <- as.numeric(substr(economic_data$Period, 7, 7))
economic_data$Date    <- as.Date(paste0(economic_data$Year, "-",
                                         (economic_data$Quarter - 1) * 3 + 1, "-01"))

# Post-2017 dummy variable (constitutional referendum)
economic_data$Post_2017 <- ifelse(economic_data$Year >= 2017, 1, 0)

head(economic_data)

# ============================================================
# 1. CORRELATION ANALYSIS
# ============================================================

cor_vars <- economic_data[, c("Inflation", "Consumer_Confidence", "TRY_USD",
                               "Unemployment", "Nostalgia_Strong", "Nostalgia_Medium", "Nostalgia_Weak")]

correlation_matrix <- cor(cor_vars, use = "complete.obs")
print("Correlation Matrix:")
print(round(correlation_matrix, 3))

# Save correlation matrix
write.csv(round(correlation_matrix, 3),
          file = "outputs/tables/correlation_matrix.csv",
          row.names = TRUE)

# Correlation plot
png("outputs/figures/correlation_plot.png", width = 800, height = 600)
corrplot(correlation_matrix, method = "color", type = "upper",
         order = "hclust", tl.cex = 0.8, tl.col = "black")
dev.off()
cat("Correlation plot saved.\n")

# ============================================================
# 2. BASELINE OLS REGRESSION MODELS
# ============================================================

# Model 1: Strong Dictionary
model1_strong <- lm(Nostalgia_Strong ~ Inflation + Consumer_Confidence + TRY_USD + Unemployment,
                    data = economic_data)

# Model 2: Medium Dictionary
model2_medium <- lm(Nostalgia_Medium ~ Inflation + Consumer_Confidence + TRY_USD + Unemployment,
                    data = economic_data)

# Model 3: Weak Dictionary
model3_weak <- lm(Nostalgia_Weak ~ Inflation + Consumer_Confidence + TRY_USD + Unemployment,
                  data = economic_data)

print("=== MODEL 1: STRONG DICTIONARY ===")
summary(model1_strong)

print("=== MODEL 2: MEDIUM DICTIONARY ===")
summary(model2_medium)

print("=== MODEL 3: WEAK DICTIONARY ===")
summary(model3_weak)

# Save model summaries
sink("outputs/tables/model1_strong_summary.txt"); print(summary(model1_strong)); sink()
sink("outputs/tables/model2_medium_summary.txt"); print(summary(model2_medium)); sink()
sink("outputs/tables/model3_weak_summary.txt");   print(summary(model3_weak));   sink()

# ============================================================
# 2.5 MULTICOLLINEARITY DIAGNOSTICS (VIF)
# ============================================================

library(car)

vif_strong <- vif(model1_strong)
vif_medium <- vif(model2_medium)
vif_weak   <- vif(model3_weak)

vif_results <- data.frame(
  Variable   = names(vif_strong),
  VIF_Strong = as.numeric(vif_strong),
  VIF_Medium = as.numeric(vif_medium),
  VIF_Weak   = as.numeric(vif_weak)
)

print("=== VIF RESULTS ===")
print(vif_results)
print("Interpretation: VIF < 5 = low, 5-10 = moderate, > 10 = high multicollinearity")

write.csv(vif_results, "outputs/tables/vif_analysis.csv", row.names = FALSE)

# ============================================================
# 3. 2017 STRUCTURAL BREAK — INTERACTION MODELS
# ============================================================

model4_strong_2017 <- lm(Nostalgia_Strong ~ Inflation * Post_2017 +
                           Consumer_Confidence * Post_2017 +
                           TRY_USD * Post_2017 +
                           Unemployment * Post_2017, data = economic_data)

model4_medium_2017 <- lm(Nostalgia_Medium ~ Inflation * Post_2017 +
                           Consumer_Confidence * Post_2017 +
                           TRY_USD * Post_2017 +
                           Unemployment * Post_2017, data = economic_data)

model4_weak_2017 <- lm(Nostalgia_Weak ~ Inflation * Post_2017 +
                         Consumer_Confidence * Post_2017 +
                         TRY_USD * Post_2017 +
                         Unemployment * Post_2017, data = economic_data)

print("=== MODEL 4: 2017 INTERACTION (STRONG) ===")
summary(model4_strong_2017)

sink("outputs/tables/model4_strong_2017_summary.txt"); print(summary(model4_strong_2017)); sink()
sink("outputs/tables/model4_medium_2017_summary.txt"); print(summary(model4_medium_2017)); sink()
sink("outputs/tables/model4_weak_2017_summary.txt");   print(summary(model4_weak_2017));   sink()

# ============================================================
# 4. LAG REGRESSION ANALYSIS (0–12 MONTHS)
# ============================================================

# Build lag variables (each lag = 1 quarter = ~3 months)
economic_data <- economic_data %>%
  arrange(Year, Quarter) %>%
  mutate(
    Inflation_lag1    = lag(Inflation, 1),     # 3 months
    Inflation_lag2    = lag(Inflation, 2),     # 6 months
    Inflation_lag3    = lag(Inflation, 3),     # 9 months
    Inflation_lag4    = lag(Inflation, 4),     # 12 months
    TRY_USD_lag1      = lag(TRY_USD, 1),
    TRY_USD_lag2      = lag(TRY_USD, 2),
    TRY_USD_lag3      = lag(TRY_USD, 3),
    TRY_USD_lag4      = lag(TRY_USD, 4),
    Unemployment_lag1 = lag(Unemployment, 1),
    Unemployment_lag2 = lag(Unemployment, 2),
    Unemployment_lag3 = lag(Unemployment, 3),
    Unemployment_lag4 = lag(Unemployment, 4)
  )

# Cross-correlation plots
png("outputs/figures/ccf_inflation.png", width = 800, height = 600)
ccf_inf <- ccf(economic_data$Inflation, economic_data$Nostalgia_Strong,
               lag.max = 8, na.action = na.pass,
               main = "Inflation–Nostalgia Cross-Correlation",
               ylab = "Cross-correlation", xlab = "Lag (quarters)")
dev.off()

png("outputs/figures/ccf_tryusd.png", width = 800, height = 600)
ccf_try <- ccf(economic_data$TRY_USD, economic_data$Nostalgia_Strong,
               lag.max = 8, na.action = na.pass,
               main = "TRY/USD–Nostalgia Cross-Correlation",
               ylab = "Cross-correlation", xlab = "Lag (quarters)")
dev.off()

png("outputs/figures/ccf_unemployment.png", width = 800, height = 600)
ccf_une <- ccf(economic_data$Unemployment, economic_data$Nostalgia_Strong,
               lag.max = 8, na.action = na.pass,
               main = "Unemployment–Nostalgia Cross-Correlation",
               ylab = "Cross-correlation", xlab = "Lag (quarters)")
dev.off()

# Report strongest lags
cat("Strongest lag — Inflation:   Lag", ccf_inf$lag[which.max(abs(ccf_inf$acf))],
    "quarters\n")
cat("Strongest lag — TRY/USD:    Lag", ccf_try$lag[which.max(abs(ccf_try$acf))],
    "quarters\n")
cat("Strongest lag — Unemployment: Lag", ccf_une$lag[which.max(abs(ccf_une$acf))],
    "quarters\n")

# Short lag model (3–6 months)
model5a_strong <- lm(Nostalgia_Strong ~ Inflation + Inflation_lag1 + Inflation_lag2 +
                       TRY_USD + TRY_USD_lag1 + TRY_USD_lag2 +
                       Unemployment + Unemployment_lag1 + Unemployment_lag2,
                     data = economic_data)

# Long lag model (3–12 months)
model5b_strong <- lm(Nostalgia_Strong ~ Inflation + Inflation_lag1 + Inflation_lag2 +
                       Inflation_lag3 + Inflation_lag4 +
                       TRY_USD + TRY_USD_lag1 + TRY_USD_lag2 +
                       TRY_USD_lag3 + TRY_USD_lag4 +
                       Unemployment + Unemployment_lag1 + Unemployment_lag2 +
                       Unemployment_lag3 + Unemployment_lag4,
                     data = economic_data)

print("=== MODEL 5A: SHORT LAG (3–6 MONTHS) — STRONG DICTIONARY ===")
summary(model5a_strong)

print("=== MODEL 5B: LONG LAG (3–12 MONTHS) — STRONG DICTIONARY ===")
summary(model5b_strong)

sink("outputs/tables/model5a_short_lag_strong.txt"); print(summary(model5a_strong)); sink()
sink("outputs/tables/model5b_long_lag_strong.txt");  print(summary(model5b_strong)); sink()

# Lag model comparison (R²)
lag_comparison <- data.frame(
  Model         = c("Short_Lag_Strong", "Long_Lag_Strong"),
  R_Squared     = c(summary(model5a_strong)$r.squared, summary(model5b_strong)$r.squared),
  Adj_R_Squared = c(summary(model5a_strong)$adj.r.squared, summary(model5b_strong)$adj.r.squared),
  AIC           = c(AIC(model5a_strong), AIC(model5b_strong))
)

print("=== LAG MODEL PERFORMANCE COMPARISON ===")
print(lag_comparison)
write.csv(lag_comparison, "outputs/tables/lag_model_comparison.csv", row.names = FALSE)

# ============================================================
# 5. OVERALL MODEL COMPARISON TABLE
# ============================================================

tidy_m1 <- tidy(model1_strong) %>% mutate(Model = "Strong_Dictionary")
tidy_m2 <- tidy(model2_medium) %>% mutate(Model = "Medium_Dictionary")
tidy_m3 <- tidy(model3_weak)   %>% mutate(Model = "Weak_Dictionary")

all_models <- bind_rows(tidy_m1, tidy_m2, tidy_m3)

print("=== ALL MODEL COEFFICIENTS ===")
print(all_models %>%
        filter(term != "(Intercept)") %>%
        select(Model, term, estimate, p.value) %>%
        arrange(term, Model))

r_squared_table <- data.frame(
  Model         = c("Strong_Dictionary", "Medium_Dictionary", "Weak_Dictionary"),
  R_Squared     = c(summary(model1_strong)$r.squared,
                    summary(model2_medium)$r.squared,
                    summary(model3_weak)$r.squared),
  Adj_R_Squared = c(summary(model1_strong)$adj.r.squared,
                    summary(model2_medium)$adj.r.squared,
                    summary(model3_weak)$adj.r.squared)
)

print("=== R-SQUARED COMPARISON ===")
print(r_squared_table)

write.csv(all_models %>%
            filter(term != "(Intercept)") %>%
            select(Model, term, estimate, p.value) %>%
            arrange(term, Model),
          "outputs/tables/all_models_comparison.csv", row.names = FALSE)

write.csv(r_squared_table, "outputs/tables/model_r_squared.csv", row.names = FALSE)

# ============================================================
# 6. HYPOTHESIS TEST RESULTS
# ============================================================

# H1: Economic deterioration → increase in nostalgic rhetoric
print("=== HYPOTHESIS 1 RESULTS ===")
print("Inflation coefficients (positive = H1 supported):")
print(all_models %>% filter(term == "Inflation") %>% select(Model, estimate, p.value))

# H2: Structural shift post-2017
print("=== HYPOTHESIS 2 RESULTS ===")
print("Post-2017 interaction coefficients:")
print(tidy(model4_strong_2017) %>%
        filter(grepl("Post_2017", term)) %>%
        select(term, estimate, p.value))

# Optional: Granger causality tests (requires lmtest package)
tryCatch({
  library(lmtest)
  print("=== GRANGER CAUSALITY TESTS ===")
  granger_inf <- grangertest(Nostalgia_Strong ~ Inflation,    order = 4, data = economic_data)
  granger_try <- grangertest(Nostalgia_Strong ~ TRY_USD,      order = 4, data = economic_data)
  granger_une <- grangertest(Nostalgia_Strong ~ Unemployment, order = 4, data = economic_data)
  print("Inflation → Nostalgia:"); print(granger_inf)
  print("TRY/USD → Nostalgia:");   print(granger_try)
  print("Unemployment → Nostalgia:"); print(granger_une)

  granger_df <- bind_rows(
    tidy(granger_inf) %>% mutate(Variable = "Inflation"),
    tidy(granger_try) %>% mutate(Variable = "TRY_USD"),
    tidy(granger_une) %>% mutate(Variable = "Unemployment")
  )
  write.csv(granger_df, "outputs/tables/granger_test_results.csv", row.names = FALSE)
}, error = function(e) {
  message("lmtest not installed — Granger tests skipped. Run: install.packages('lmtest')")
})

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All model summaries saved to: outputs/tables/\n")
cat("All figures saved to:         outputs/figures/\n")
