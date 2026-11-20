
import pandas as pd
import joblib

from pathlib import Path

model_path = Path("models/random_forest.joblib")
train_path = Path("data/processed/ddos_train.csv")
report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

print("Loading Random Forest model...")

model = joblib.load(model_path)

print("Loading feature names...")

# فقط نام ستون‌ها را لازم داریم
columns = pd.read_csv(train_path, nrows=0).columns
feature_names = [
    column for column in columns
    if column != "Label"
]

# استخراج اهمیت ویژگی‌ها از مدل
importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "Rank",
    range(1, len(importance_df) + 1)
)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(
    index=False,
    formatters={"Importance": "{:.6f}".format}
))

output_path = report_dir / "feature_importance.csv"
importance_df.to_csv(output_path, index=False)

print(f"\nFull feature importance saved to: {output_path}")
