import pandas as pd
import os
import matplotlib.pyplot as plt

# 📁 Dateipfade
doc = "kategorie_vergleich.xlsx"
pathes = ["Kategorisierungen_A", "Kategorisierungen_Adv", "Kategorisierungen_Alle", "Kategorisierungen_N", "Kategorisierungen_Verb", "Kategorisierungen_NOUNVERB", "Kategorisierungen_ADJADV"]

for dir in pathes:
    file_path = os.path.join(dir, doc)
    output_filename = "abweichungen_kategorien.txt"
    output_path = os.path.join(dir, output_filename)
    plot_filename = "abweichungen_kategorien_plot.png"
    plot_path = os.path.join(dir, plot_filename)

    print(f"📂 Öffne Datei: {file_path}")

    # 📥 Excel-Datei laden
    df = pd.read_excel(file_path)

    # 🔧 Kombiniere Modell + Texttyp
    df["Modell_Typ"] = df["Model"] + " – " + df["TextType"]
    df_features = df.drop(columns=["Model", "TextType"])

    # 📊 Berechne globale Mittelwerte
    global_means = df_features.drop(columns="Modell_Typ").mean()

    # ➖ Abweichungen berechnen
    df_base = df_features.set_index("Modell_Typ")
    df_absolute = df_base - global_means
    df_relative = ((df_base - global_means) / global_means) * 100

    # 🔀 In Long-Form bringen
    abs_stack = df_absolute.stack().reset_index()
    abs_stack.columns = ["Modell_Typ", "Kategorie", "Abweichung_Absolut"]

    rel_stack = df_relative.stack().reset_index()
    rel_stack.columns = ["Modell_Typ", "Kategorie", "Abweichung_Prozent"]

    # 🔗 Zusammenführen
    df_combined = pd.merge(abs_stack, rel_stack, on=["Modell_Typ", "Kategorie"])

    # 🔝 Top 30 positive & negative Abweichungen
    top30_pos = df_combined.sort_values("Abweichung_Absolut", ascending=False).head(30)
    top30_neg = df_combined.sort_values("Abweichung_Absolut").head(30)
    top_combined = pd.concat([top30_neg, top30_pos])

    # 📄 Textdatei schreiben
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Top 30 Überrepräsentationen:\n\n")
        for _, row in top30_pos.iterrows():
            file.write(f"{row['Modell_Typ']} – {row['Kategorie']}\n")
            file.write(f"  ➤ Abweichung absolut: +{row['Abweichung_Absolut']:.2f}\n")
            file.write(f"  ➤ Abweichung prozentual: +{row['Abweichung_Prozent']:.2f}%\n\n")

        file.write("\nTop 30 Unterrepräsentationen:\n\n")
        for _, row in top30_neg.iterrows():
            file.write(f"{row['Modell_Typ']} – {row['Kategorie']}\n")
            file.write(f"  ➤ Abweichung absolut: {row['Abweichung_Absolut']:.2f}\n")
            file.write(f"  ➤ Abweichung prozentual: {row['Abweichung_Prozent']:.2f}%\n\n")

    print("✅ Abweichungen wurden gespeichert unter:", output_path)

    # 📊 Balkendiagramm erstellen
    labels = top_combined["Modell_Typ"] + " – " + top_combined["Kategorie"]
    values = top_combined["Abweichung_Absolut"]
    colors = ["crimson" if v < 0 else "steelblue" for v in values]

    plt.figure(figsize=(12, 12))
    plt.barh(labels, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Abweichung zur Durchschnittsnutzung")
    plt.title("Top 30 Über- und Unterrepräsentierte Kategorien")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("📊 Diagramm gespeichert unter:", plot_path)
