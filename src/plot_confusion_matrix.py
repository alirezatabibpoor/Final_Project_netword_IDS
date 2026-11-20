import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# File paths
test_path = Path("data/processed/multiclass_test.csv")
model_path = Path("models/decision_tree_baseline.joblib")
output_path = Path("reports/decision_tree_confusion_matrix.png")

print("Loading test dataset...")
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Label", "Source_File"])
y_test = test_df["Label"]

print("Loading Decision Tree model...")
model = joblib.load(model_path)

print("Generating predictions...")
y_pred = model.predict(X_test)

class_names = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

print("\nConfusion Matrix:")
print(cm)

print("\nCreating heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format=","
)

ax.set_title("Decision Tree - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nConfusion matrix saved to: {output_path}")
print("Plot completed successfully.")


# Create normalized confusion matrix
print("\nCreating normalized confusion matrix...")

cm_normalized = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names,
    normalize="true"
)

fig, ax = plt.subplots(figsize=(14, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format=".1%"
)

ax.set_title("Decision Tree - Normalized Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.tight_layout()

normalized_path = Path(
    "reports/decision_tree_confusion_matrix_normalized.png"
)

plt.savefig(normalized_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Normalized confusion matrix saved to: {normalized_path}")
print("Normalized plot completed successfully.")