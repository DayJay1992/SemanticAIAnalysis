import spacy
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from spacy.tokens import Doc



# Lade deutsches Transformer-Modell
nlp = spacy.load("de_dep_news_trf")

# POS-Kombinationen (Universal POS Tags)
CONTENT_POS = ["ADJ, ADV", "NOUN, VERB"]

# Lade JSON-Texte
with open("corpus/texte.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for kat_str in CONTENT_POS:
    kat = [k.strip() for k in kat_str.split(",")]
    outputfolder = f"Kategorisierungen_{kat_str.replace(', ', '')}"
    os.makedirs(outputfolder, exist_ok=True)

    print(f"\n📂 Verwende POS-Kombination: {kat} → Outputfolder: {outputfolder}")

    def extract_content_lemmas_with_vectors(text):
        doc = nlp(text)

        # Transformer-Ausgaben direkt abrufen
        trf_data = doc._.trf_data
        last_hidden = trf_data.last_hidden_layer_state  # shape: [n_wordpieces, hidden_dim]
        align = trf_data.wordpieces.align  # Liste von Listen mit WordPiece-Indizes pro Token

        vectors = []
        lemmas = []

        for i, token in enumerate(doc):
            if token.pos_ in kat and not token.is_stop and token.is_alpha:
                wp_indices = align[i]
                if not wp_indices:
                    continue
                vec = np.mean([last_hidden[j] for j in wp_indices], axis=0)
                vectors.append(vec)
                lemmas.append(token.lemma_.lower())

        return lemmas, vectors

    def build_global_categories(lemmas, vectors, similarity_threshold=0.7):
        clusters = defaultdict(list)
        category_vectors = {}

        for lemma, vector in zip(lemmas, vectors):
            best_match = None
            best_score = -1.0
            for category, cat_vec in category_vectors.items():
                sim = cosine_similarity(vector, cat_vec)
                if sim > best_score and sim >= similarity_threshold:
                    best_score = sim
                    best_match = category
            if best_match:
                clusters[best_match].append(lemma)
            else:
                clusters[lemma].append(lemma)
                category_vectors[lemma] = vector

        return clusters, category_vectors

    def cosine_similarity(v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def assign_lemmas_to_categories(lemmas, vectors, category_vectors):
        assignment = []
        for lemma, vec in zip(lemmas, vectors):
            best_match = None
            best_score = -1.0
            for category, cat_vec in category_vectors.items():
                sim = cosine_similarity(vec, cat_vec)
                if sim > best_score:
                    best_score = sim
                    best_match = category
            if best_match:
                assignment.append(best_match)
        return Counter(assignment)

    def create_plot():
        df = pd.read_excel(f"{outputfolder}/kategorie_vergleich.xlsx")
        df["Kombi"] = df["Model"] + " – " + df["TextType"]
        kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType", "Kombi"]]
        if not kategorie_spalten:
            print("⚠️ Keine Kategorien zum Plotten vorhanden.")
            return
        gesamt = df[kategorie_spalten].sum().sort_values(ascending=False)
        top_10_kategorien = list(gesamt.head(10).index)
        df_plot = df[["Kombi"] + top_10_kategorien].set_index("Kombi").T
        plt.figure(figsize=(14, 6))
        df_plot.plot(kind="bar", figsize=(16, 8), width=0.85)
        plt.title("Top 10 semantische Kategorien – Häufigkeit pro Modell/Texttyp")
        plt.ylabel("Häufigkeit")
        plt.xlabel("Kategorie")
        plt.xticks(rotation=45)
        plt.legend(title="Modell – Texttyp", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{outputfolder}/kategorien_top10_vergleich.png")
        print("📊 Plot erstellt!")

    global_lemmas = []
    global_vectors = []
    for obj in data:
        if "humanText" in obj:
            lemmas, vectors = extract_content_lemmas_with_vectors(obj["humanText"])
            global_lemmas.extend(lemmas)
            global_vectors.extend(vectors)
        for val in obj.values():
            if isinstance(val, dict):
                for text in [val.get("TextA", ""), val.get("TextB", "")]:
                    lemmas, vectors = extract_content_lemmas_with_vectors(text)
                    global_lemmas.extend(lemmas)
                    global_vectors.extend(vectors)

    print("🔍 Kategorisiere globale Lemmata ...")
    categories, category_vectors = build_global_categories(global_lemmas, global_vectors)
    print(f"✅ {len(categories)} Kategorien erstellt.")

    results = []
    for obj in tqdm(data, desc="📄 Verarbeite JSON-Objekte"):
        if "humanText" in obj:
            lemmas, vectors = extract_content_lemmas_with_vectors(obj["humanText"])
            counts = assign_lemmas_to_categories(lemmas, vectors, category_vectors)
            results.append({"Model": "HumanText", "TextType": "Original", **counts})
        for model_name, val in tqdm(obj.items(), desc="🔄 Modelle pro Textsatz", leave=False):
            if isinstance(val, dict):
                for text_type in ["TextA", "TextB"]:
                    text = val.get(text_type, "")
                    if text.strip():
                        lemmas, vectors = extract_content_lemmas_with_vectors(text)
                        counts = assign_lemmas_to_categories(lemmas, vectors, category_vectors)
                        results.append({"Model": model_name, "TextType": text_type, **counts})

    df = pd.DataFrame(results).fillna(0)
    df = df.groupby(["Model", "TextType"], as_index=False).sum()
    kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType"]]
    top_cols = df[kategorie_spalten].sum().sort_values(ascending=False).head(100).index.tolist()
    df = df[["Model", "TextType"] + top_cols]

    output_path = f"{outputfolder}/kategorie_vergleich.xlsx"
    df.to_excel(output_path, index=False)
    print(f"💾 Top 100 Kategorien gespeichert in {output_path}")

    df_long = df.melt(id_vars=["Model", "TextType"], var_name="Kategorie", value_name="Häufigkeit")
    df_long = df_long[df_long["Häufigkeit"] > 0]
    top10_per_group = (
        df_long.sort_values(["Model", "TextType", "Häufigkeit"], ascending=[True, True, False])
        .groupby(["Model", "TextType"])
        .head(10)
    )
    with pd.ExcelWriter(output_path, mode="a", engine="openpyxl") as writer:
        top10_per_group.to_excel(writer, sheet_name="Top10_ProModell", index=False)

    with open(f"{outputfolder}/globale_kategorien.txt", "w", encoding="utf-8") as f:
        f.write(f"Gesamtzahl der Kategorien: {len(categories)}\n")
        for cat, words in sorted(categories.items()):
            f.write(f"Kategorie: {cat} ({len(words)} Lemmata)\n")
            f.write(", ".join(sorted(words)) + "\n")

    print(f"📂 Kategorien gespeichert in: {outputfolder}/globale_kategorien.txt")
    create_plot()