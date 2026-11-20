
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

train_path = Path("data/processed/ddos_train.csv")
test_path = Path("data/processed/ddos_test.csv")

print("Loading training and testing datasets...")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]

print("Training Decision Tree...")

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=20
)

model.fit(X_train, y_train)

print("Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["BENIGN", "DDoS"]
)

print("\nDecision Tree Results")
print("-" * 45)
print(f"Accuracy: {accuracy:.6f}")

print("\nClassification Report:")
print(report)

print("Confusion Matrix (BENIGN, DDoS):")
print(matrix)

# ذخیره مدل
model_path = Path("models/decision_tree.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)
dump(model, model_path)

# ذخیره گزارش
report_path = Path("reports/decision_tree_report.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Decision Tree Results\n")
    file.write("-" * 45 + "\n")
    file.write(f"Accuracy: {accuracy:.6f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix (BENIGN, DDoS):\n")
    file.write(str(matrix))
    file.write("\n")

print(f"\nModel saved to: {model_path}")
print(f"Report saved to: {report_path}")
