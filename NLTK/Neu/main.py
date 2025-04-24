import spacy
import numpy as np
import pandas as pd
import json
import re
from collections import Counter
from nltk.util import ngrams
import nltk
import os

# Wortarten zur Analyse
to_analyze = ["NOUN", "ADJ", "ADV", "VERB"]

# Lade spaCy-Modell
nlp = spacy.load("de_core_news_lg")

# Lade JSON-Datei
with open("../../Wortfeldanalyse/texte.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Bereinige Text
def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

# Vektor-Durchschnitt
def get_average_vector(words, model_name, text_type):
    print(f"Berechne Durchschnittsvektor für {model_name} ({text_type})")
    word_vectors = [nlp(word).vector for word in words if nlp(word).has_vector]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros((nlp.vocab.vectors_length,))

# Kosinus-Ähnlichkeit
def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Stilometrie-Features mit externer Filterfunktion
def get_stylometric_features(text, model, text_type, lemma_filter_func, label_suffix=""):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    num_sentences = len(sentences) if sentences else 1
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    filtered_lemmas = lemma_filter_func(text)
    unique_lemmas = len(set(filtered_lemmas))
    num_lemmas = len(filtered_lemmas)
    unique_tokens = len(set(words))
    lemma_ttr = unique_lemmas / num_lemmas if num_lemmas > 0 else 0
    token_ttr = unique_tokens / num_words if num_words > 0 else 0

    output_dir = "unique_lemmata_output"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{model}_{text_type}{label_suffix}_stylometry.txt"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {model}\n")
        file.write(f"Texttype: {text_type}\n")
        file.write(f"Num_Lemmas: {num_lemmas}\n")
        file.write(f"Unique Lemmas ({unique_lemmas}):\n")
        file.write(", ".join(sorted(set(filtered_lemmas))) + "\n")
        file.write("\nStylometric Features:\n")
        file.write(f"NumSentences: {num_sentences}\n")
        file.write(f"AvgSentenceLength: {round(avg_sentence_length, 2)}\n")
        file.write(f"TTR_Lemma_based: {round(lemma_ttr, 3)}\n")
        file.write(f"TTR_Token_Based: {round(token_ttr, 3)}\n")
        file.write(f"NumWords: {num_words}\n")
        file.write(f"AvgWordLength: {round(avg_word_length, 2)}\n")

    print(f"✅ Stylometric features gespeichert in: {file_path}")

    return {
        "NumSentences": num_sentences,
        "NumWords": num_words,
        "AvgSentenceLength": round(avg_sentence_length, 2),
        "Num_Lemmas": {num_lemmas},
        "UniqueLemmas": unique_lemmas,
        "TTR_Lemma_based": round(lemma_ttr, 3),
        "TTR_Token_based": round(token_ttr, 3),
        "AvgWordLength": round(avg_word_length, 2)
    }

# n-Gramme extrahieren
def get_top_ngrams(text, n=2, top_k=30):
    tokens = [token.text.lower() for token in nlp(text) if token.is_alpha]
    ngram_counts = Counter(ngrams(tokens, n))
    return ngram_counts.most_common(top_k)

# JSON-Daten vorbereiten
human_texts = []
model_texts = {}

for obj in data:
    humantext = clean_text(obj.get("humanText", ""))
    if humantext:
        human_texts.append(humantext)

    for model_name, content in obj.items():
        if isinstance(content, dict):
            for text_type in ["TextA", "TextB"]:
                text = clean_text(content.get(text_type, ""))
                if text:
                    if model_name not in model_texts:
                        model_texts[model_name] = {"TextA": [], "TextB": []}
                    model_texts[model_name][text_type].append(text)

human_text_combined = " ".join(human_texts)

# Analyse-Funktion
def run_analysis(wortarten, output_filename):
    print(f"\n🔍 Starte Analyse für: {wortarten if isinstance(wortarten, list) else [wortarten]}")
    if isinstance(wortarten, list):
        label_suffix = "_gesamt"
    else:
        label_suffix = f"_{wortarten}"

    def get_filtered_lemmas_dynamic(text):
        doc = nlp(text)
        filtered_lemmas = [
            token.lemma_.lower() for token in doc
            if (token.pos_ in wortarten if isinstance(wortarten, list) else token.pos_ == wortarten)
            and not token.is_punct and not token.is_digit and not token.is_stop
            and not re.match(r"^\d+[a-zA-Z]$", token.text)
            and not re.match(r"^\d{2,4}ff$", token.text)
            and not re.match(r"^\d{2,4}–\d{2,4}$", token.text)
            and not re.match(r"^\(\d+\)$", token.text)
            and not re.match(r"^\d+\.\)$", token.text)
            and not re.match(r"^\(\d+[a-zA-Z]?\)$", token.text)
            and not re.match(r"^(vgl|al)\.$", token.text, re.IGNORECASE)
            and not re.match(r"^\(i{1,3}v?|v?i{0,3}\)$", token.text, re.IGNORECASE)
            and not re.match(r"^\d+(\.\d+)*\.$", token.text)
        ]
        return filtered_lemmas

    results = []
    similarity_results = []
    ngram_results = []

    # Human-Text analysieren
    human_lemmas = get_filtered_lemmas_dynamic(human_text_combined)
    human_vector = get_average_vector(human_lemmas, "HumanText", "Original")
    human_features = get_stylometric_features(human_text_combined, "HumanText", "Original", get_filtered_lemmas_dynamic, label_suffix)
    lemma_freq = Counter(human_lemmas).most_common(30)
    bigrams = get_top_ngrams(human_text_combined, n=2)
    trigrams = get_top_ngrams(human_text_combined, n=3)
    quadrigrams = get_top_ngrams(human_text_combined, n=4)

    results.append({"Model": "HumanText", "TextType": "Original", **human_features})
    for word, freq in lemma_freq:
        results.append({"Model": "HumanText", "TextType": "Original", "TopLemma": word, "Frequency": freq})
    for ngram, freq in bigrams:
        ngram_results.append({"Model": "HumanText", "TextType": "Original", "Bigram": " ".join(ngram), "Frequency": freq})
    for ngram, freq in trigrams:
        ngram_results.append({"Model": "HumanText", "TextType": "Original", "Trigram": " ".join(ngram), "Frequency": freq})
    for ngram, freq in quadrigrams:
        ngram_results.append({"Model": "HumanText", "TextType": "Original", "Quadrigram": " ".join(ngram), "Frequency": freq})

    # Modelltexte analysieren
    for model, texts in model_texts.items():
        for text_type, text_list in texts.items():
            combined_text = " ".join(text_list)
            lemmas = get_filtered_lemmas_dynamic(combined_text)
            model_vector = get_average_vector(lemmas, model, text_type)
            similarity = cosine_similarity(human_vector, model_vector)
            features = get_stylometric_features(combined_text, model, text_type, get_filtered_lemmas_dynamic, label_suffix)
            lemma_freq = Counter(lemmas).most_common(100)
            bigrams = get_top_ngrams(combined_text, n=2)
            trigrams = get_top_ngrams(combined_text, n=3)
            quadrigrams = get_top_ngrams(combined_text, n=4)

            results.append({"Model": model, "TextType": text_type, "Similarity": round(similarity, 3), **features})
            similarity_results.append({"Model": model, "TextType": text_type, "Similarity": round(similarity, 3)})
            for word, freq in lemma_freq:
                results.append({"Model": model, "TextType": text_type, "TopLemma": word, "Frequency": freq})
            for ngram, freq in bigrams:
                ngram_results.append({"Model": model, "TextType": text_type, "Bigram": " ".join(ngram), "Frequency": freq})
            for ngram, freq in trigrams:
                ngram_results.append({"Model": model, "TextType": text_type, "Trigram": " ".join(ngram), "Frequency": freq})
            for ngram, freq in quadrigrams:
                ngram_results.append({"Model": model, "TextType": text_type, "Quadrigram": " ".join(ngram), "Frequency": freq})

    # Speichern
    df_similarity = pd.DataFrame(similarity_results)
    df_stylometry = pd.DataFrame(results)
    df_ngrams = pd.DataFrame(ngram_results)

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_similarity.to_excel(writer, sheet_name='Semantische Ähnlichkeit', index=False)
        df_stylometry.to_excel(writer, sheet_name='Stilometrie', index=False)
        df_ngrams.to_excel(writer, sheet_name='N-Gramme', index=False)

    print(f"✅ Analyse abgeschlossen und gespeichert unter: {output_filename}")

# Gesamtauswertung
run_analysis(to_analyze, "textanalyse_gesamt.xlsx")

# Einzelanalysen
for wortart in to_analyze:
    run_analysis(wortart, f"textanalyse_{wortart}.xlsx")
