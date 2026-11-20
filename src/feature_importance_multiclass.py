
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

model_path = Path("models/random_forest_multiclass.joblib")
output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading trained Random Forest model...")
model = joblib.load(model_path)

print("Extracting feature importance...")

feature_names = model.feature_names_in_
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

top_20 = importance_df.head(20)

print("\nTop 20 Important Features")
print("=" * 55)
print(top_20.to_string(
    index=False,
    formatters={"Importance": "{:.6f}".format}
))

csv_path = output_dir / "random_forest_feature_importance.csv"
top_20.to_csv(csv_path, index=False)

print(f"\nFeature importance saved to: {csv_path}")

print("\nCreating feature importance chart...")

plot_df = top_20.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))
plt.barh(plot_df["Feature"], plot_df["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Network Feature")
plt.title("Top 20 Feature Importances - Random Forest")
plt.tight_layout()

chart_path = output_dir / "random_forest_top20_features.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart saved to: {chart_path}")
print("\nFeature importance analysis completed.")
