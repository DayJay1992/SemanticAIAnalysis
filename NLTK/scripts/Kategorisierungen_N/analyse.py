import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your Excel file (adjust filename and sheet name if needed)
df = pd.read_excel("kategorie_table.csv")

# Convert columns to string to avoid any parsing issues
df["Model"] = df["Model"].astype(str)
df["TextType"] = df["TextType"].astype(str)

# Run ANOVA for "Kapitel"
model_kapitel = ols("kapitel ~ C(Model) + C(TextType) + C(Model):C(TextType)", data=df).fit()
anova_kapitel = sm.stats.anova_lm(model_kapitel, typ=2)
print("\n📘 ANOVA for 'Kapitel':\n", anova_kapitel)

# Run ANOVA for "Abschnitt"
model_abschnitt = ols("abschnitt ~ C(Model) + C(TextType) + C(Model):C(TextType)", data=df).fit()
anova_abschnitt = sm.stats.anova_lm(model_abschnitt, typ=2)
print("\n📗 ANOVA for 'Abschnitt':\n", anova_abschnitt)

# Optional: plot boxplots
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="TextType", y="kapitel", hue="Model")
plt.title("Kapitel by Model and TextType")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="TextType", y="abschnitt", hue="Model")
plt.title("Abschnitt by Model and TextType")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()