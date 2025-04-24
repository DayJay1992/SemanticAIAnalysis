
import pandas as pd
import json
import re
import os
from collections import defaultdict, Counter
import spacy
from tqdm import tqdm

# 📦 Sprachmodell laden
nlp = spacy.load("de_core_news_lg")

# 📁 Dateipfade
kategorien_path = "kategorie_vergleich.xlsx"
text_path = "mein_text.txt"  # zu analysierender Text
output_txt_path = "begruendung_analyse.txt"  # Ergebnisbericht

# 🔧 Fallback-Text falls Datei nicht vorhanden
dummy_text = """
In dieser Untersuchung wird deutlich, dass zentrale Argumentationsmuster sowie relevante Strukturen im Text häufig eine bedeutende Rolle spielen.
"""

# 🔍 Text bereinigen
def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

# 🧠 Adjektive extrahieren
def extract_adjectives(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"ADJ","ADV"}
        and token.has_vector
        and not token.is_stop
        and token.is_alpha
    ]

# 📥 Lade Text
if os.path.exists(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
else:
    raw_text = dummy_text

# 📊 Lade Kategorietabelle
df = pd.read_excel(kategorien_path)
df["Modell_Typ"] = df["Model"] + " – " + df["TextType"]

# 🔢 Mittelwerte berechnen
df_base = df.drop(columns=["Model", "TextType", "Modell_Typ"])
global_means = df_base.mean()

# 🔍 Profile für Mensch vs KI
human_profile = df[df["Model"] == "HumanText"].drop(columns=["Model", "TextType", "Modell_Typ"]).mean()
ki_profile = df[df["Model"] != "HumanText"].drop(columns=["Model", "TextType", "Modell_Typ"]).mean()

# 📦 Vorbereitung: alle bekannten Kategorien mit Vektor
known_categories = list(global_means.index)
category_vectors = {cat: nlp(cat) for cat in known_categories if nlp(cat).has_vector}

# 🧠 Text analysieren
text = clean_text(raw_text)
adjectives = extract_adjectives(text)
lemma_vectors = {lemma: nlp(lemma) for lemma in set(adjectives) if nlp(lemma).has_vector}

# 🔗 Kategorisiere Adjektive
assignment = []
for lemma, vec in lemma_vectors.items():
    best_score = 0.0
    best_match = None
    for cat, cat_vec in category_vectors.items():
        sim = vec.similarity(cat_vec)
        if sim > best_score:
            best_score = sim
            best_match = cat
    if best_match:
        assignment.append(best_match)

# 📊 Häufigkeit der Kategorien
assigned_counts = Counter(assignment)

# 📋 Abweichung + Klassifikation
deviation_report = []
ki_score = 0
human_score = 0

for cat, count in assigned_counts.items():
    human_avg = human_profile.get(cat, 0)
    ki_avg = ki_profile.get(cat, 0)

    if human_avg == 0 and ki_avg == 0:
        continue

    dist_to_human = abs(count - human_avg)
    dist_to_ki = abs(count - ki_avg)

    if dist_to_human < dist_to_ki:
        tendenz = "✅ eher menschlich"
        human_score += abs(count - global_means.get(cat, 0))
    elif dist_to_ki < dist_to_human:
        tendenz = "🤖 eher KI-typisch"
        ki_score += abs(count - global_means.get(cat, 0))
    else:
        tendenz = "❓ unklar"

    global_avg = global_means.get(cat, 0)
    diff = count - global_avg
    percent = (diff / global_avg) * 100 if global_avg else 0

    deviation_report.append((cat, count, round(global_avg, 2), round(diff), round(percent, 1), tendenz))

# 🔽 Sortieren nach Abweichung
deviation_report.sort(key=lambda x: -abs(x[4]))

# 📊 Konfidenzscore berechnen
total = ki_score + human_score
if total > 0:
    ki_confidence = ki_score / total
    human_confidence = human_score / total
else:
    ki_confidence = human_confidence = 0.5

# 💾 Speichere als Textdatei
with open(output_txt_path, "w", encoding="utf-8") as out:
    out.write("📊 Analyse der semantischen Adjektiv-Kategorien\n")
    out.write("=" * 60 + "\n\n")
    out.write(f"Verwendete Adjektive im Text: {len(adjectives)}\n")
    out.write(f"Verglichene Kategorien: {len(known_categories)}\n\n")
    out.write(f"🤖 KI-Score: {ki_score:.2f}\n")
    out.write(f"✅ Mensch-Score: {human_score:.2f}\n")
    out.write(f"\n📈 KI-Wahrscheinlichkeit: {ki_confidence:.2%}\n")
    out.write(f"📉 Mensch-Wahrscheinlichkeit: {human_confidence:.2%}\n\n")
    out.write("Top auffällige Kategorien mit Klassifikation:\n\n")

    for cat, count, avg, diff, pct, tendenz in deviation_report[:30]:
        trend = "↑" if pct > 0 else "↓"
        out.write(f"- {cat}: {count} (⌀ {avg}) {trend} {diff:+} ({pct:+.1f}%) → {tendenz}\n")

output_txt_path