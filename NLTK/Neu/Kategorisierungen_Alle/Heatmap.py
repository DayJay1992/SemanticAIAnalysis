import pandas as pd
import plotly.express as px

# 📌 Excel laden
df = pd.read_excel("kategorie_vergleich.xlsx")

# 🔹 Kombiniere Modell & Texttyp
df["Kombi"] = df["Model"] + " – " + df["TextType"]
df = df.set_index("Kombi")

# 🔹 Entferne unnötige Spalten
df = df.drop(columns=["Model", "TextType"])

# 🔹 In "long format" bringen
df_long = df.reset_index().melt(id_vars="Kombi", var_name="Kategorie", value_name="Häufigkeit")

# 🔹 Nur Kategorien mit Häufigkeit > 0
df_long = df_long[df_long["Häufigkeit"] > 0]

# 📊 Interaktive Heatmap mit Scroll
fig = px.imshow(
    df.pivot_table(index="Kombi", values=df.columns, aggfunc="sum"),
    labels=dict(x="Kategorie", y="Modell – Texttyp", color="Häufigkeit"),
    color_continuous_scale="YlOrRd",
    height=700,
    aspect="auto"
)

fig.update_layout(title="Interaktive Heatmap: Kategorienutzung pro Modell", xaxis_tickangle=-45)
fig.write_html("interaktive_heatmap.html")
fig.show()
