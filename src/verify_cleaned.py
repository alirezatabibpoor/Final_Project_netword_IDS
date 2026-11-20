
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path("data/processed/ddos_clean.csv")

print("Verifying cleaned dataset...")
print("-" * 50)

total_rows = 0
label_counts = {}
missing_counts = {}
infinity_counts = {}

for chunk in pd.read_csv(file_path, chunksize=100_000):
    total_rows += len(chunk)

    # بررسی توزیع برچسب‌ها
    for label, count in chunk["Label"].value_counts().items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    # شمارش مقادیر خالی
    for column, count in chunk.isnull().sum().items():
        missing_counts[column] = (
            missing_counts.get(column, 0) + int(count)
        )

    # شمارش مقادیر بی‌نهایت در ستون‌های عددی
    numeric_data = chunk.select_dtypes(include=[np.number])
    for column, count in np.isinf(numeric_data).sum().items():
        infinity_counts[column] = (
            infinity_counts.get(column, 0) + int(count)
        )

print(f"\nTotal rows: {total_rows:,}")
print("\nLabel distribution:")

for label, count in sorted(label_counts.items()):
    percentage = count / total_rows * 100
    print(f"{label}: {count:,} ({percentage:.2f}%)")

remaining_missing = {
    column: count
    for column, count in missing_counts.items()
    if count > 0
}

remaining_infinity = {
    column: count
    for column, count in infinity_counts.items()
    if count > 0
}

print("\nRemaining missing values:")
print(remaining_missing if remaining_missing else "None")

print("\nRemaining infinite values:")
print(remaining_infinity if remaining_infinity else "None")

if not remaining_missing and not remaining_infinity:
    print("\nVerification passed!")
else:
    print("\nWarning: Some invalid values remain.")
