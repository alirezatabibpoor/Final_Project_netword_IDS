
import time
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

train_path = Path("data/processed/multiclass_train.csv")
test_path = Path("data/processed/multiclass_test.csv")

model_path = Path("models/random_forest_multiclass.joblib")
report_path = Path("data/processed/random_forest_multiclass_report.txt")

print("Loading training and testing datasets...")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Keep source information out of the model features.
X_train = train_df.drop(columns=["Label", "Source_File"])
y_train = train_df["Label"]

X_test = test_df.drop(columns=["Label", "Source_File"])
y_test = test_df["Label"]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {y_train.nunique()}")

print("\nInitializing Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    verbose=1
)

print("\nTraining Random Forest...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")

print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

class_names = sorted(y_test.unique())

report = classification_report(
    y_test,
    y_pred,
    labels=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

print("\nRandom Forest Multiclass Results")
print("=" * 50)
print(f"Accuracy:    {accuracy:.6f}")
print(f"Macro F1:    {macro_f1:.6f}")
print(f"Weighted F1: {weighted_f1:.6f}")
print(f"Training time: {training_time:.2f} seconds")

print("\nClassification Report:")
print(report)

print("Class order in confusion matrix:")
print(class_names)
print("\nConfusion Matrix:")
print(matrix)

model_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Random Forest Multiclass Results\n")
    file.write("=" * 50 + "\n")
    file.write(f"Accuracy: {accuracy:.6f}\n")
    file.write(f"Macro F1: {macro_f1:.6f}\n")
    file.write(f"Weighted F1: {weighted_f1:.6f}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nClass order in confusion matrix:\n")
    file.write(str(class_names))
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(matrix))
    file.write("\n")

print(f"\nModel saved to: {model_path}")
print(f"Report saved to: {report_path}")
print("\nRandom Forest experiment completed.")
