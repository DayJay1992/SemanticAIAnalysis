import pandas as pd
import numpy as np
import nltk
import re
import os

# 📁 Datei-Pfade
stilometrie_path = "stilometrie_werte.xlsx"
text_path = "mein_testtext.txt"
output_path = "begruendung_klassifikation.txt"

# 📥 Eingabetext lesen
with open(text_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

# 🔍 Text bereinigen
def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

text = clean_text(raw_text)

# 🧠 Stilometrie-Funktion
def extract_stylometric_features(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    num_sentences = len(sentences) if sentences else 1
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    unique_lemmas = len(set(words))
    ttr_lemma = unique_lemmas / num_words if num_words > 0 else 0
    ttr_token = len(set(words)) / num_words if num_words > 0 else 0

    return {
        "NumSentences": num_sentences,
        "NumWords": num_words,
        "AvgSentenceLength": round(avg_sentence_length, 2),
        "UniqueLemmas": unique_lemmas,
        "TTR_Lemma_based": round(ttr_lemma, 3),
        "TTR_Token_based": round(ttr_token, 3),
        "AvgWordLength": round(avg_word_length, 2)
    }

# 📊 Stilometrische Daten aus Excel laden
stilometrie_df = pd.read_excel(stilometrie_path)
stilometrie_df = stilometrie_df[stilometrie_df["NumSentences"].notna()]  # Sicherheit

# 🔢 Nur numerische Mittelwerte berechnen
relevante_spalten = [
    "NumSentences", "NumWords", "AvgSentenceLength",
    "UniqueLemmas", "TTR_Lemma_based", "TTR_Token_based", "AvgWordLength"
]
global_means = stilometrie_df[relevante_spalten].mean()

# 🧠 Neue Textanalyse
features = extract_stylometric_features(text)

# 📄 Analysebericht schreiben
with open(output_path, "w", encoding="utf-8") as out:
    out.write("📄 Klassifikationsanalyse – Begründung\n")
    out.write("=" * 50 + "\n\n")

    out.write("📝 Eingabetext: mein_testtext.txt\n\n")
    out.write("🔍 Stilometrische Merkmale:\n")
    for key, value in features.items():
        global_val = global_means.get(key, 0)
        delta = round(value - global_val, 2)
        deviation = f"({'+' if delta > 0 else ''}{delta} gegenüber Ø {global_val:.2f})"
        out.write(f"- {key}: {value} {deviation}\n")

    # 📊 Bewertungshinweise
    out.write("\n📊 Bewertungshinweise:\n")
    if features["TTR_Lemma_based"] < global_means["TTR_Lemma_based"] - 0.05:
        out.write("→ Niedrige TTR – Hinweis auf maschinellen Stil.\n")
    if features["AvgSentenceLength"] > global_means["AvgSentenceLength"] + 3:
        out.write("→ Lange Satzstruktur – oft in KI-Texten.\n")
    if features["UniqueLemmas"] < global_means["UniqueLemmas"] - 200:
        out.write("→ Geringe Wortvielfalt – typisch für KI.\n")
    if features["TTR_Token_based"] < global_means["TTR_Token_based"] - 0.05:
        out.write("→ Auch auf Token-Ebene geringe Vielfalt.\n")
    else:
        out.write("→ Stilometrische Merkmale im Rahmen menschlicher Varianz.\n")

print(f"✅ Analyse gespeichert in: {output_path}")