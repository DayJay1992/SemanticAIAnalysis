import spacy
import json
import os
import re
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Loading German model
nlp = spacy.load("de_core_news_lg")

# POS-Tags
CONTENT_POS = ["NOUN", "ADJ", "VERB", "ADV", "NOUN, ADJ, VERB, ADV", "ADJ, ADV", "NOUN, VERB"]
#CONTENT_POS = ["ADJ, ADV", "NOUN, VERB"]

#Determining output folder
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

    

# Function for lemma extraction
    def extract_content_lemmas(text):
        doc = nlp(text)
        return [
            token.lemma_.lower() for token in doc
            if token.pos_ in kat
            and token.has_vector
            and not token.is_stop
            and token.is_alpha
        ]

    # Create global categories based on a similarity of at least 0.7
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

    # Decide for each lemma to which category it belongs
    def assign_lemmas_to_categories(lemmas, clusters):
        # Cache
        lemma_vec_cache = {lemma: nlp(lemma) for lemma in set(lemmas) if nlp(lemma).has_vector}
        category_vectors = {cat: nlp(cat) for cat in clusters if nlp(cat).has_vector}
        assignment = []

        for lemma in tqdm(lemmas, desc=f"🔍 Categorize {len(lemmas)} lemmas", leave=False):
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
        # 🔹 Read excel
        df = pd.read_excel(f"{outputfolder}/kategorie_vergleich.xlsx")

        #Create a new column with Modell_Texttyp as a unit
        # 🔹 Erstelle eine neue Spalte mit „Modell_Texttyp“ als Gruppierungseinheit
        df["Kombi"] = df["Model"] + " – " + df["TextType"]

        # 🔹 Extract Category-Columns (minus meta columns)
        kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType", "Kombi"]]

        # 🔹 Calculate occurences of each category type
        gesamt = df[kategorie_spalten].sum().sort_values(ascending=False)

        # 🔹 select top 10 categories
        top_10_kategorien = list(gesamt.head(10).index)

        # 🔹 Reframe dataframe for plot
        df_plot = df[["Kombi"] + top_10_kategorien].set_index("Kombi").T

        # 🔹 Create plot
        plt.figure(figsize=(14, 6))
        df_plot.plot(kind="bar", figsize=(16, 8), width=0.85)

        plt.title("Top 10 semantische Kategorien – Häufigkeit pro Modell/Texttyp")
        plt.ylabel("Häufigkeit")
        plt.xlabel("Kategorie")
        plt.xticks(rotation=45)
        plt.legend(title="Modell – Texttyp", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # 🔹 save
        plt.savefig("kategorien_top10_vergleich.png")
        print("Plot erstellt!")

    # Load JSON-File
    with open("/home/dennis/Nextcloud/Documents/Linguistik/ForschungKI/Wortfeldanalyse/texte.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 1: Collect global lemmas for each model + human
    global_lemmas = set()

    for obj in data:
        if "humanText" in obj:
            global_lemmas.update(extract_content_lemmas(obj["humanText"]))
        for val in obj.values():
            if isinstance(val, dict):
                global_lemmas.update(extract_content_lemmas(val.get("TextA", "")))
                global_lemmas.update(extract_content_lemmas(val.get("TextB", "")))

    # step 2: create global categories
    print("🔍 Kategorisiere globale Lemmata ...")
    categories = build_global_categories(global_lemmas)
    print(f"✅ {len(categories)} Kategorien erstellt.")

    # step 3: Analyze each text
    results = []

    for obj in tqdm(data, desc="📄 Verarbeite JSON-Objekte"):
        # Analyze human text
        if "humanText" in obj:
            lemmas = extract_content_lemmas(obj["humanText"])
            counts = assign_lemmas_to_categories(lemmas, categories)
            results.append({"Model": "HumanText", "TextType": "Original", **counts})

        # Analyze AI-texts
        for model_name, val in tqdm(obj.items(), desc="🔄 Modelle pro Textsatz", leave=False):
            if isinstance(val, dict):
                for text_type in ["TextA", "TextB"]:
                    text = val.get(text_type, "")
                    if text.strip():
                        lemmas = extract_content_lemmas(text)
                        counts = assign_lemmas_to_categories(lemmas, categories)
                        results.append({"Model": model_name, "TextType": text_type, **counts})

    # Step 4: Export results as excel
    df = pd.DataFrame(results).fillna(0)

    # Group by model and texttype and add all occurences
    df = df.groupby(["Model", "TextType"], as_index=False).sum()
    kategorie_spalten = [col for col in df.columns if col not in ["Model", "TextType"]]

    # Select top 100 category by occurence
    top_cols = df[kategorie_spalten].sum().sort_values(ascending=False).head(100).index.tolist()

    # Limit dataframe to top 100 only (excel might crash otherwise)
    cols = ["Model", "TextType"] + top_cols
    df = df[cols]

    output_path = f"{outputfolder}/kategorie_vergleich.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Top 100 Kategorien gespeichert in {outputfolder}/kategorie_vergleich.xlsx")

    #Extra Sheet that shows the top occurences of categories for each model type
    df_long = df.melt(id_vars=["Model", "TextType"], var_name="Kategorie", value_name="Häufigkeit")

    # Only occurences > 0
    df_long = df_long[df_long["Häufigkeit"] > 0]

    # For each group select top 10
    top10_per_group = (
        df_long.sort_values(["Model", "TextType", "Häufigkeit"], ascending=[True, True, False])
        .groupby(["Model", "TextType"])
        .head(10)
    )

    # Export in a new sheet of the same excel-file
    with pd.ExcelWriter(f"{outputfolder}/kategorie_vergleich.xlsx", mode="a", engine="openpyxl") as writer:
        top10_per_group.to_excel(writer, sheet_name="Top10_ProModell", index=False)

    # Save ALL categroies and their occurences in a .txt file (because Excel is just top 100)
    with open(f"{outputfolder}/globale_kategorien.txt", "w", encoding="utf-8") as f:
        f.write(f"Gesamtzahl der Kategorien: {len(categories)}")
        for cat, words in sorted(categories.items()):
            f.write(f"Kategorie: {cat} ({len(words)} Lemmata)")
            f.write(", ".join(sorted(words)) + "")

    print("📂 Kategorien gespeichert in: {outputfolder}/globale_kategorien.txt")

    create_plot()
