
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

test_path = Path("data/processed/ddos_test.csv")
models_dir = Path("models")
reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

print("Loading test data...")
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]

model_files = {
    "Decision Tree": models_dir / "decision_tree.joblib",
    "Random Forest": models_dir / "random_forest.joblib",
}

results = []

for model_name, model_path in model_files.items():
    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "DDoS Precision": precision_score(
            y_test, predictions, pos_label="DDoS"
        ),
        "DDoS Recall": recall_score(
            y_test, predictions, pos_label="DDoS"
        ),
        "DDoS F1-score": f1_score(
            y_test, predictions, pos_label="DDoS"
        ),
        "False Alarms": int(
            confusion_matrix(
                y_test, predictions, labels=["BENIGN", "DDoS"]
            )[0, 1]
        ),
        "Missed Attacks": int(
            confusion_matrix(
                y_test, predictions, labels=["BENIGN", "DDoS"]
            )[1, 0]
        ),
    })

comparison = pd.DataFrame(results)

print("\nModel Comparison:")
print(comparison.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.6f}".format,
        "DDoS Precision": "{:.6f}".format,
        "DDoS Recall": "{:.6f}".format,
        "DDoS F1-score": "{:.6f}".format,
    }
))

output_path = reports_dir / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
