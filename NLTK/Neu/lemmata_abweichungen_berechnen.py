import pandas as pd
import os
import matplotlib.pyplot as plt

# 📁 Eingabe & Ausgabe
input_file = "analyse_output/Mappe1.xlsx"
output_folder = "analyse_output"
os.makedirs(output_folder, exist_ok=True)

# 📥 Lade Daten
df = pd.read_excel(input_file)
df["Modell_Typ"] = df["Model"] + " – " + df["TextType"]

# 📊 Pivot-Tabelle: Zeilen = Modell_Typ, Spalten = Lemma, Werte = Frequenz
pivot_df = df.pivot_table(index="Modell_Typ", columns="TopLemma", values="Frequency", aggfunc="sum").fillna(0)

# 🔢 Globale Mittelwerte je Lemma
global_means = pivot_df.mean()

# ➖ Abweichung berechnen
deviation_df = pivot_df - global_means

# 📄 TXT-Ausgabe mit Top 5 pro Modell
txt_output = os.path.join(output_folder, "modellbezogene_abweichungen.txt")

top_all = []  # Für späteren Gesamtplot

with open(txt_output, "w", encoding="utf-8") as f:
    f.write("🧠 Stärkste positiven Abweichungen je Modell:\n\n")
    
    for index, row in deviation_df.iterrows():
        top = row.sort_values(ascending=False).head(5)
        f.write(f"▶ {index}\n")
        for lemma, diff in top.items():
            f.write(f"  {lemma:<30} +{int(diff)}\n")
            top_all.append((index, lemma, diff))
        f.write("\n")

# 📊 Gesamtplot der 30 stärksten individuellen Ausschläge
top_all_sorted = sorted(top_all, key=lambda x: x[2], reverse=True)[:30]

labels = [f"{lemma} ({mod})" for mod, lemma, _ in top_all_sorted]
values = [diff for _, _, diff in top_all_sorted]

plt.figure(figsize=(12, 6))
plt.barh(labels, values, color="darkblue")
plt.title("Top 30 Lemmata mit stärkster Übernutzung pro Modell")
plt.xlabel("Abweichung vom globalen Mittelwert")
plt.gca().invert_yaxis()
plt.tight_layout()

plot_output = os.path.join(output_folder, "top_lemma_abweichungen_global.png")
plt.savefig(plot_output)

print("✅ Alles erledigt!")
print(f"📄 Abweichungen gespeichert in: {txt_output}")
print(f"📊 Plot gespeichert in: {plot_output}")