
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import confusion_matrix

test_path = Path("data/processed/multiclass_test.csv")

models = {
    "Decision Tree": Path("models/decision_tree_baseline.joblib"),
    "Random Forest": Path("models/random_forest_multiclass.joblib"),
}

output_path = Path("data/processed/multiclass_error_analysis.csv")

print("Loading test dataset...")
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Label", "Source_File"])
y_test = test_df["Label"]

results = []
class_names = sorted(y_test.unique())

for model_name, model_path in models.items():
    print(f"\nAnalyzing {model_name}...")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_names
    )

    for i, class_name in enumerate(class_names):
        actual_count = int(matrix[i, :].sum())
        correct_count = int(matrix[i, i])
        missed_count = actual_count - correct_count

        # Other classes incorrectly predicted as this class.
        false_positives = int(matrix[:, i].sum() - matrix[i, i])

        results.append({
            "Model": model_name,
            "Class": class_name,
            "Actual_Samples": actual_count,
            "Correct_Predictions": correct_count,
            "Missed_Samples": missed_count,
            "False_Positive_Predictions": false_positives,
            "Recall": (
                correct_count / actual_count
                if actual_count else 0
            ),
            "Precision": (
                correct_count / (correct_count + false_positives)
                if correct_count + false_positives else 0
            ),
        })

    # Overall errors for the model.
    total_errors = int((y_test != y_pred).sum())
    print(f"Total incorrect predictions: {total_errors:,}")

    benign_mask = y_test == "BENIGN"
    benign_false_alarms = int(
        ((y_pred != "BENIGN") & benign_mask).sum()
    )
    print(
        "BENIGN samples incorrectly flagged as attacks: "
        f"{benign_false_alarms:,}"
    )

results_df = pd.DataFrame(results)

output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print("\nError Analysis Summary")
print("=" * 80)

for model_name in models:
    subset = results_df[results_df["Model"] == model_name].copy()
    subset = subset.sort_values(
        "False_Positive_Predictions",
        ascending=False
    )

    print(f"\n{model_name}")
    print(
        subset[
            [
                "Class",
                "Actual_Samples",
                "Missed_Samples",
                "False_Positive_Predictions",
                "Recall",
                "Precision",
            ]
        ].to_string(
            index=False,
            formatters={
                "Recall": "{:.4f}".format,
                "Precision": "{:.4f}".format,
            }
        )
    )

print(f"\nResults saved to: {output_path}")
print("\nError analysis completed.")
