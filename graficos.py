import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("banana_quality.csv")

sns.kdeplot(df, x="Size", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()

sns.kdeplot(df, x="Ripeness", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()

sns.kdeplot(df, x="Acidity", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()

sns.kdeplot(df, x="Sweetness", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()

sns.kdeplot(df, x="HarvestTime", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()


sns.kdeplot(df, x="Softness", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()

sns.kdeplot(df, x="Weight", hue="Quality",  fill=True, common_norm=False, alpha=.5, linewidth=0)
plt.show()
