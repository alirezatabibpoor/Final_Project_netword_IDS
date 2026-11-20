import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

model_path = Path("models/decision_tree_baseline.joblib")
output_csv = Path("reports/decision_tree_feature_importance.csv")
output_png = Path("reports/decision_tree_top20_features.png")

print("Loading Decision Tree model...")
model = joblib.load(model_path)

print("Extracting feature importance...")
importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

output_csv.parent.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(output_csv, index=False)

top20 = importance_df.head(20).sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))
plt.barh(top20["Feature"], top20["Importance"])
plt.xlabel("Feature Importance")
plt.title("Decision Tree - Top 20 Important Features")
plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()

print("\nTop 20 Important Features:")
print(
    importance_df.head(20).to_string(
        index=False,
        formatters={"Importance": "{:.6f}".format}
    )
)

print(f"\nCSV saved to: {output_csv}")
print(f"Chart saved to: {output_png}")
print("\nFeature importance analysis completed.")