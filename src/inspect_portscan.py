
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path(
    "data/raw/MachineLearningCSV/"
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

print("Analyzing the complete PortScan dataset...")
print("-" * 55)

label_counts = {}
total_rows = 0
missing_values = {}
infinite_values = {}

for chunk in pd.read_csv(file_path, chunksize=50_000, low_memory=False):
    # استانداردسازی نام ستون‌ها و برچسب‌ها
    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    total_rows += len(chunk)

    # شمارش برچسب‌ها
    counts = chunk["Label"].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    # بررسی مقادیر گمشده و بی‌نهایت در ویژگی‌های عددی
    features = chunk.drop(columns=["Label"])

    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")

        missing_count = int(values.isna().sum())
        infinite_count = int(np.isinf(values).sum())

        if missing_count:
            missing_values[column] = (
                missing_values.get(column, 0) + missing_count
            )

        if infinite_count:
            infinite_values[column] = (
                infinite_values.get(column, 0) + infinite_count
            )

print(f"\nTotal rows: {total_rows:,}")

print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    percentage = count / total_rows * 100
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nColumns with missing or non-numeric values:")
if missing_values:
    for column, count in sorted(missing_values.items()):
        print(f"{column}: {count:,}")
else:
    print("None")

print("\nColumns with infinite values:")
if infinite_values:
    for column, count in sorted(infinite_values.items()):
        print(f"{column}: {count:,}")
else:
    print("None")

print("\nAnalysis completed.")
