# ============================================================
# 03_lemmatisation.py
# Turkish Lemmatisation using Zemberek NLP
# ============================================================
# Lemmatises the cleaned AKP speech corpus using the
# Zemberek Turkish NLP library via JPype (Java bridge).
# Success rate: ~94.3% of tokens successfully lemmatised.
#
# Prerequisites:
#   - Java JDK 21 installed
#   - Zemberek JAR: https://github.com/ahmetaa/zemberek-nlp
#   - pip install jpype1 pandas
#
# Usage:
#   Update JAR_PATH and JVM_PATH below to match your system,
#   then run: python 03_lemmatisation.py
# ============================================================

import pandas as pd
import jpype
import jpype.imports
from jpype.types import *
import re

# --- Configuration ---
# Update these paths to match your local setup
INPUT_PATH  = "data/processed/akp_speeches_clean.csv"
OUTPUT_PATH = "data/processed/akp_speeches_lemmatised.csv"

# Path to Zemberek JAR file (download from GitHub releases)
JAR_PATH = "zemberek-full.jar"

# Path to your JVM (update to match your OS and Java installation)
# macOS example:  "/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/lib/server/libjvm.dylib"
# Linux example:  "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so"
# Windows example: "C:/Program Files/Java/jdk-21/bin/server/jvm.dll"
JVM_PATH = "/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/lib/server/libjvm.dylib"

# --- Start JVM ---
if not jpype.isJVMStarted():
    try:
        jpype.startJVM(JVM_PATH, classpath=[JAR_PATH])
        print("JVM started successfully.")
    except Exception as e:
        print(f"JVM startup error: {e}")
        exit()

# --- Load Zemberek Classes ---
try:
    TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
    TurkishSentenceExtractor = jpype.JClass("zemberek.tokenization.TurkishSentenceExtractor")
    morphology = TurkishMorphology.createWithDefaults()
    sentence_extractor = TurkishSentenceExtractor.DEFAULT
    print("Zemberek classes loaded successfully.")
except Exception as e:
    print(f"Zemberek loading error: {e}")
    exit()

# --- Lemmatisation Function ---
def lemmatize_text(text):
    """
    Lemmatises a Turkish text string using Zemberek morphological analysis.
    Falls back to the original token if no analysis is found.
    Returns a space-joined string of lemmas.
    """
    lemmatized_words = []
    try:
        if not text or not str(text).strip():
            return ""

        # Clean text: keep Turkish characters, letters, digits
        cleaned_text = re.sub(r'[^a-zA-Z0-9çÇğĞıİöÖşŞüÜ\s]', '', str(text)).lower()
        words = cleaned_text.split()

        for word in words:
            if not word:
                continue
            try:
                analyses = morphology.analyze(word)
                results = analyses.getAnalysisResults()

                if not results:
                    # No analysis found: keep original token
                    lemmatized_words.append(word)
                else:
                    first_analysis = results[0]
                    dict_item = first_analysis.getDictionaryItem()

                    if dict_item is not None:
                        lemma = str(dict_item.lemma).strip()
                        lemmatized_words.append(lemma if lemma else word)
                    else:
                        lemmatized_words.append(word)

            except Exception:
                lemmatized_words.append(word)

        return " ".join(lemmatized_words)

    except Exception as e:
        print(f"Lemmatisation error: {e}")
        return text

# --- Load Data ---
try:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8").fillna("")
    print(f"CSV loaded. Total rows: {len(df)}")
except Exception as e:
    print(f"CSV read error: {e}")
    exit()

# --- Test on First 5 Rows ---
print("Testing lemmatisation on first 5 rows...")
df_test = df.head(5).copy()
df_test["lemma_text"] = df_test["text"].apply(lambda x: lemmatize_text(str(x)))

print("\nSample output:")
for i, row in df_test.iterrows():
    print(f"\nRow {i+1}:")
    print(f"  Original : {row['text'][:120]}...")
    print(f"  Lemmatised: {row['lemma_text'][:120]}...")
    print("-" * 50)

# --- Process Full Dataset ---
confirm = input("\nProcess full dataset? (y/n): ").strip().lower()
if confirm == 'y':
    print("Processing full dataset (this may take a while)...")
    df["lemma_text"] = df["text"].apply(lambda x: lemmatize_text(str(x)))
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Full lemmatised dataset saved to: {OUTPUT_PATH}")
else:
    print("Full processing skipped.")

# --- Shutdown JVM ---
jpype.shutdownJVM()
print("Done.")
