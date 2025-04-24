import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Excel-Datei einlesen
df = pd.read_excel("kategorie_vergleich.xlsx")

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
plt.show()
