
from pathlib import Path
from collections import Counter
import pandas as pd

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

label_counts = Counter()
total_rows = 0

print("Inspecting WebAttacks dataset labels...")

for chunk in pd.read_csv(file_path, chunksize=50_000):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError(
            f"Label column not found. Columns: {chunk.columns.tolist()}"
        )

    labels = chunk["Label"].fillna("MISSING").astype(str).str.strip()
    label_counts.update(labels.value_counts().to_dict())
    total_rows += len(chunk)

print("\nLabel distribution:")
for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print(f"\nTotal rows: {total_rows:,}")
print("\nInspection completed.")
