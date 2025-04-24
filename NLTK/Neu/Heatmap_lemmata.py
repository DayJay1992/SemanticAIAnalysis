import pandas as pd
import plotly.express as px

# 📌 Excel laden – und spezifisch das Stilometrie-Sheet mit Lemmata

files = ["textanalyse_ADJ", "textanalyse_ADV","textanalyse_gesamt", "textanalyse_NOUN", "textanalyse_VERB" ]
for file in files:
    df = pd.read_excel(f"{file}.xlsx", sheet_name="Stilometrie")

    # 🔹 Filtere nur Lemma-Zeilen
    df = df[df["TopLemma"].notna() & df["Frequency"].notna()]
    df["Kombi"] = df["Model"] + " – " + df["TextType"]

    # 🔹 Pivot: Zeilen = Kombi, Spalten = Lemmata, Werte = Häufigkeit
    pivot_df = df.pivot_table(index="Kombi", columns="TopLemma", values="Frequency", aggfunc="sum").fillna(0)

    # 🔹 Optional: Beschränke auf Top 30 Lemmata
    top_30_lemmata = pivot_df.sum().sort_values(ascending=False).head(30).index
    pivot_df = pivot_df[top_30_lemmata]

    # 📊 Interaktive Heatmap
    fig = px.imshow(
        pivot_df,
        labels={"x": "Lemma", "y": "Modell – Texttyp", "color": "Frequenz"},
        color_continuous_scale="YlOrRd",
        height=800
    )

    fig.update_layout(
        title="🧠 Interaktive Heatmap: Top-Lemmata in Stilometrie",
        xaxis_tickangle=-45
    )

    # 🔄 Als HTML speichern & anzeigen
    fig.write_html(f"{file}-interaktive_stilometrie_heatmap.html")