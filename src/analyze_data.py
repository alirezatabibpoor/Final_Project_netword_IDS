
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

print("Analyzing the DDoS dataset...")
print("-" * 50)

label_counts = {}
missing_counts = None
infinity_counts = None
total_rows = 0

for chunk in pd.read_csv(file_path, chunksize=100_000):
    # حذف فاصله‌های اضافی از نام ستون‌ها
    chunk.columns = chunk.columns.str.strip()

    # حذف فاصله‌های اضافی از برچسب‌ها
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    total_rows += len(chunk)

    counts = chunk["Label"].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    # شمارش مقادیر خالی
    missing = chunk.isnull().sum()
    if missing_counts is None:
        missing_counts = missing
    else:
        missing_counts = missing_counts.add(missing, fill_value=0)

    # شمارش مقادیر بی‌نهایت در ستون‌های عددی
    numeric_data = chunk.select_dtypes(include=[np.number])
    infinities = np.isinf(numeric_data).sum()

    if infinity_counts is None:
        infinity_counts = infinities
    else:
        infinity_counts = infinity_counts.add(infinities, fill_value=0)

print(f"\nTotal rows: {total_rows:,}")

print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    percentage = count / total_rows * 100
    print(f"{label}: {count:,} ({percentage:.2f}%)")

print("\nColumns with missing values:")
found_missing = False
for column, count in missing_counts.items():
    if count > 0:
        print(f"{column}: {int(count):,}")
        found_missing = True

if not found_missing:
    print("No missing values found.")

print("\nColumns with infinite values:")
found_infinity = False
for column, count in infinity_counts.items():
    if count > 0:
        print(f"{column}: {int(count):,}")
        found_infinity = True

if not found_infinity:
    print("No infinite values found.")

print("\nAnalysis completed.")
