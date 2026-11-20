
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

input_path = Path("reports/feature_importance.csv")
output_path = Path("reports/feature_importance.png")

print("Loading feature importance data...")

df = pd.read_csv(input_path)

# انتخاب 15 ویژگی مهم
top_features = df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Network Feature")
plt.title("Top 15 Important Features - Random Forest")
plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nChart saved to: {output_path}")
