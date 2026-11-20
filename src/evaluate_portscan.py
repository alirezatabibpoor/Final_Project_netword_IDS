
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data_path = Path("data/processed/portscan_clean.csv")
model_path = Path("models/random_forest.joblib")

print("Loading PortScan dataset and trained model...")

df = pd.read_csv(data_path)
model = load(model_path)

# همان ویژگی‌هایی که مدل هنگام آموزش استفاده کرده است
feature_columns = list(model.feature_names_in_)

X_test = df[feature_columns]
y_actual = df["Label"].str.strip()

# پیش‌بینی مدل: BENIGN یا DDoS
y_pred = model.predict(X_test)

# برای این آزمایش، هر دو DDoS پیش‌بینی‌شده و PortScan واقعی
# در گروه کلی «حمله» قرار می‌گیرند.
# این تبدیل فقط برای ارزیابی دودویی است.
y_actual_binary = y_actual.map({
    "BENIGN": "BENIGN",
    "PortScan": "ATTACK"
})

y_pred_binary = pd.Series(y_pred).map({
    "BENIGN": "BENIGN",
    "DDoS": "ATTACK"
})

print("\nBinary detection results")
print("-" * 50)

print(f"Total samples: {len(df):,}")
print(f"Accuracy: {accuracy_score(y_actual_binary, y_pred_binary):.6f}")

print("\nClassification report:")
print(classification_report(
    y_actual_binary,
    y_pred_binary,
    labels=["BENIGN", "ATTACK"],
    target_names=["BENIGN", "PortScan (attack)"],
    digits=4,
    zero_division=0
))

print("Confusion matrix:")
print("Rows = actual [BENIGN, PortScan]")
print("Columns = predicted [BENIGN, ATTACK]")
print(confusion_matrix(
    y_actual_binary,
    y_pred_binary,
    labels=["BENIGN", "ATTACK"]
))

print("\nPredicted labels from the original model:")
print(pd.Series(y_pred).value_counts())

print("\nEvaluation completed.")
