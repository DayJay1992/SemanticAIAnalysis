import spacy
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Lade deutsches Sprachmodell mit Vektoren
nlp = spacy.load("de_core_news_lg")

# POS-Tags für Inhaltswörter
CONTENT_POS = ["NOUN", "ADJ", "VERB", "ADV", "NOUN, ADJ, VERB, ADV", "ADJ, ADV", "NOUN, VERB"]
#CONTENT_POS = ["ADJ, ADV", "NOUN, VERB"]
for kat in CONTENT_POS:
    if kat == "NOUN":
        outputfolder = "Kategorisierungen_N"
    elif kat == "ADJ":
        outputfolder = "Kategorisierungen_A"
    elif kat == "VERB":
        outputfolder = "Kategorisierungen_Verb"
    elif kat == "ADV":
        outputfolder = "Kategorisierungen_Adv"
    elif kat == "NOUN, ADJ, VERB, ADV":
        outputfolder = "Kategorisierungen_Alle"
    elif kat == "ADJ, ADV":
        outputfolder = "Kategorisierungen_ADJADV"
    elif kat == "NOUN, VERB":
        outputfolder = "Kategorisierungen_NOUNVERB"

    print(f"Outputfolder for kat {kat}: /{outputfolder}")

    

# Funktion zur Lemma-Extraktion aus einem Text
    def extract_content_lemmas(text):
        doc = nlp(text)
        return [
            token.lemma_.lower() for token in doc
            if token.pos_ in kat
            and token.has_vector
            and not token.is_stop
            and token.is_alpha
        ]

    # Funktion zur Erstellung globaler Kategorien aus allen Lemmata
    def build_global_categories(lemmas, similarity_threshold=0.7):
        clusters = defaultdict(list)
        category_vectors = {}

        for lemma in sorted(lemmas):
            token = nlp(lemma)
            if not token.has_vector:
                continue

            best_match = None
            best_score = 0.0

            for category in clusters:
                if category not in category_vectors:
                    category_doc = nlp(category)
                    if category_doc.has_vector:
                        category_vectors[category] = category_doc
                    else:
                        continue

                sim = token.similarity(category_vectors[category])
                if sim > best_score and sim >= similarity_threshold:
                    best_score = sim
                    best_match = category

            if best_match:
                clusters[best_match].append(lemma)
            else:
                clusters[lemma].append(lemma)
                category_vectors[lemma] = token

        return clusters

    # Funktion zur Zuweisung von Lemmata zu Kategorien
    def assign_lemmas_to_categories(lemmas, clusters):
        # Cache für einmalige NLP-Berechnung
        lemma_vec_cache = {lemma: nlp(lemma) for lemma in set(lemmas) if nlp(lemma).has_vector}
        category_vectors = {cat: nlp(cat) for cat in clusters if nlp(cat).has_vector}
        assignment = []

        for lemma in tqdm(lemmas, desc=f"🔍 Kategorisiere {len(lemmas)} Lemmata", leave=False):
            token = lemma_vec_cache.get(lemma)
            if not token:
                continue
            best_match = None
            best_score = 0.0

            for category, vec in category_vectors.items():
                sim = token.similarity(vec)
                if sim > best_score:
                    best_score = sim
                    best_match = category

            if best_match:
                assignment.append(best_match)

        return Counter(assignment)

    def create_plot():
        # 🔹 Excel-Datei einlesen
        df = pd.read_excel(f"{outputfolder}/kategorie_vergleich.xlsx")

        # 🔹 Erstelle eine neue Spalte mit „Modell_Texttyp“ als Gruppierungseinheit
        df["Kombi"] = df["Model"] + " – " + df["TextType"]

        # 🔹 Kategorien-Spalten extrahieren (alle außer Meta-Spalten)
        kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType", "Kombi"]]

        # 🔹 Gesamtvorkommen jeder Kategorie berechnen
        gesamt = df[kategorie_spalten].sum().sort_values(ascending=False)

        # 🔹 Top 10 Kategorien auswählen
        top_10_kategorien = list(gesamt.head(10).index)

        # 🔹 DataFrame umformen für Plot
        df_plot = df[["Kombi"] + top_10_kategorien].set_index("Kombi").T

        # 🔹 Plot erstellen
        plt.figure(figsize=(14, 6))
        df_plot.plot(kind="bar", figsize=(16, 8), width=0.85)

        plt.title("Top 10 semantische Kategorien – Häufigkeit pro Modell/Texttyp")
        plt.ylabel("Häufigkeit")
        plt.xlabel("Kategorie")
        plt.xticks(rotation=45)
        plt.legend(title="Modell – Texttyp", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # 🔹 Speichern
        plt.savefig("kategorien_top10_vergleich.png")
        print("Plot erstellt!")

    # Lade JSON-Texte
    with open("/home/dennis/Nextcloud/Documents/Linguistik/ForschungKI/Wortfeldanalyse/texte.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Schritt 1: Sammle globale Lemmata aus allen Texten (Mensch + Modelle)
    global_lemmas = set()

    for obj in data:
        if "humanText" in obj:
            global_lemmas.update(extract_content_lemmas(obj["humanText"]))
        for val in obj.values():
            if isinstance(val, dict):
                global_lemmas.update(extract_content_lemmas(val.get("TextA", "")))
                global_lemmas.update(extract_content_lemmas(val.get("TextB", "")))

    # Schritt 2: Erstelle globale semantische Kategorien aus allen Lemmata
    print("🔍 Kategorisiere globale Lemmata ...")
    categories = build_global_categories(global_lemmas)
    print(f"✅ {len(categories)} Kategorien erstellt.")

    # Schritt 3: Analysiere pro Textsatz (Mensch oder Modell)
    results = []

    for obj in tqdm(data, desc="📄 Verarbeite JSON-Objekte"):
        # Analyse des menschlichen Textes
        if "humanText" in obj:
            lemmas = extract_content_lemmas(obj["humanText"])
            counts = assign_lemmas_to_categories(lemmas, categories)
            results.append({"Model": "HumanText", "TextType": "Original", **counts})

        # Analyse aller KI-generierten Texte
        for model_name, val in tqdm(obj.items(), desc="🔄 Modelle pro Textsatz", leave=False):
            if isinstance(val, dict):
                for text_type in ["TextA", "TextB"]:
                    text = val.get(text_type, "")
                    if text.strip():
                        lemmas = extract_content_lemmas(text)
                        counts = assign_lemmas_to_categories(lemmas, categories)
                        results.append({"Model": model_name, "TextType": text_type, **counts})

    # Schritt 4: Exportiere die Resultate als Excel-Tabelle mit Kategoriehäufigkeiten
    df = pd.DataFrame(results).fillna(0)

    # Gruppiere nach Modell und Texttyp, summiere alle Werte auf
    df = df.groupby(["Model", "TextType"], as_index=False).sum()
    kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType"]]

    # Top 100 Kategorien nach Gesamthäufigkeit bestimmen
    top_cols = df[kategorie_spalten].sum().sort_values(ascending=False).head(100).index.tolist()

    # DataFrame auf Top-Kategorien beschränken
    cols = ["Model", "TextType"] + top_cols
    df = df[cols]

    output_path = f"{outputfolder}/kategorie_vergleich.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Top 100 Kategorien gespeichert in {outputfolder}/kategorie_vergleich.xlsx")

    #Extra Sheet, das die Häufigkeit jeder Kategorie innerhalb eines Modells berechnet und sortiert.
    df_long = df.melt(id_vars=["Model", "TextType"], var_name="Kategorie", value_name="Häufigkeit")

    # Nur Einträge mit Häufigkeit > 0
    df_long = df_long[df_long["Häufigkeit"] > 0]

    # Für jede Gruppe: Top 10 Kategorien behalten
    top10_per_group = (
        df_long.sort_values(["Model", "TextType", "Häufigkeit"], ascending=[True, True, False])
        .groupby(["Model", "TextType"])
        .head(10)
    )

    # 📌 In neues Sheet der gleichen Excel-Datei exportieren
    with pd.ExcelWriter(f"{outputfolder}/kategorie_vergleich.xlsx", mode="a", engine="openpyxl") as writer:
        top10_per_group.to_excel(writer, sheet_name="Top10_ProModell", index=False)

    # Zusätzlich: speichere alle Cluster (Kategorien und ihre Lemmata) als Textdatei
    with open(f"{outputfolder}/globale_kategorien.txt", "w", encoding="utf-8") as f:
        f.write(f"Gesamtzahl der Kategorien: {len(categories)}")
        for cat, words in sorted(categories.items()):
            f.write(f"Kategorie: {cat} ({len(words)} Lemmata)")
            f.write(", ".join(sorted(words)) + "")

    print("📂 Kategorien gespeichert in: {outputfolder}/globale_kategorien.txt")

    create_plot()
