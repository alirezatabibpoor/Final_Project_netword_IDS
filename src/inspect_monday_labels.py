import pandas as pd
from pathlib import Path

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
)

print("Inspecting Monday dataset labels...")

label_counts = {}

for chunk in pd.read_csv(
    file_path,
    usecols=lambda column: column.strip().lower() == "label",
    chunksize=100_000,
    low_memory=False
):
    label_column = chunk.columns[0]
    labels = chunk[label_column].astype(str).str.strip()
    counts = labels.value_counts()

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

print("\nLabel distribution:")
for label, count in sorted(
    label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label}: {count:,}")

print("\nTotal rows:", f"{sum(label_counts.values()):,}")
print("\nInspection completed.")