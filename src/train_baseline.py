
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

train_file = Path("data/processed/multiclass_train.csv")
test_file = Path("data/processed/multiclass_test.csv")

model_file = Path("models/decision_tree_baseline.joblib")
report_file = Path("data/processed/decision_tree_report.txt")

print("Loading training and testing data...")

train_df = pd.read_csv(train_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

feature_columns = [
    column
    for column in train_df.columns
    if column not in {"Label", "Source_File"}
]

X_train = train_df[feature_columns]
y_train = train_df["Label"]

X_test = test_df[feature_columns]
y_test = test_df["Label"]

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Number of features: {len(feature_columns)}")

model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42,
    class_weight="balanced",
)

print("\nTraining Decision Tree...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")

print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)

labels = model.classes_

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print("Class order:", list(labels))
print(matrix)

# Save model and evaluation results
model_file.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_file)

with open(report_file, "w", encoding="utf-8") as file:
    file.write("Decision Tree Baseline Evaluation\n")
    file.write("=" * 50 + "\n")
    file.write(f"Training samples: {len(X_train):,}\n")
    file.write(f"Testing samples: {len(X_test):,}\n")
    file.write(f"Number of features: {len(feature_columns)}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n")
    file.write(f"Accuracy: {accuracy:.6f}\n")
    file.write(f"Macro F1-score: {macro_f1:.6f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(f"Class order: {list(labels)}\n")
    file.write(str(matrix))

print(f"\nModel saved to: {model_file}")
print(f"Report saved to: {report_file}")
