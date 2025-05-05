import pandas as pd
import plotly.graph_objects as go

# 📌 Excel laden
pathes = [
    "Kategorisierungen_A", "Kategorisierungen_Adv", "Kategorisierungen_Alle",
    "Kategorisierungen_N", "Kategorisierungen_Verb",
    "Kategorisierungen_NOUNVERB", "Kategorisierungen_ADJADV"
]

for ordner in pathes:
    df = pd.read_excel(f"{ordner}/kategorie_vergleich.xlsx")
    
    # 🔹 Kombiniere Modell & Texttyp
    df["Kombi"] = df["Model"] + " – " + df["TextType"]
    df = df.set_index("Kombi")

    # 🔹 Entferne unnötige Spalten
    df = df.drop(columns=["Model", "TextType"])

    # 🔹 Sicherstellen, dass alle Werte numerisch sind
    df = df.fillna(0).astype(float)

    # 🔹 Textwerte (für Annotation) vorbereiten
    text = df.astype(int).astype(str).values.tolist()

    # 📊 Interaktive Heatmap mit Werten im Feld
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        text=text,
        texttemplate="%{text}",
        colorscale="YlOrRd",
        colorbar=dict(title="Häufigkeit")
    ))

    fig.update_layout(
        title="Interaktive Heatmap: Kategorienutzung pro Modell",
        xaxis_title="Kategorie",
        yaxis_title="Modell – Texttyp",
        xaxis_tickangle=-45,
        height=700
    )

    fig.write_html(f"{ordner}/interaktive_heatmap.html")