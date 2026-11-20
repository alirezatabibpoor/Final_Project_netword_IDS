
import time
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

test_path = Path("data/processed/multiclass_test.csv")

models = {
    "Decision Tree": Path("models/decision_tree_baseline.joblib"),
    "Random Forest": Path("models/random_forest_multiclass.joblib"),
}

output_path = Path("data/processed/multiclass_model_comparison.csv")

print("Loading test dataset...")
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Label", "Source_File"])
y_test = test_df["Label"]

results = []

for model_name, model_path in models.items():
    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro_Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Weighted_F1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Prediction_Time_Seconds": prediction_time,
    })

comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 85)
print(comparison.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.6f}".format,
        "Macro_Precision": "{:.6f}".format,
        "Macro_Recall": "{:.6f}".format,
        "Macro_F1": "{:.6f}".format,
        "Weighted_F1": "{:.6f}".format,
        "Prediction_Time_Seconds": "{:.3f}".format,
    }
))

output_path.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
print("\nComparison completed.")
