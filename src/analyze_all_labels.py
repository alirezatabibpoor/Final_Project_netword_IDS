
import pandas as pd
from pathlib import Path

data_dir = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE"
)

csv_files = sorted(data_dir.glob("*.csv"))

print("Analyzing labels across all CICIDS2017 files...")
print("=" * 65)

for file_path in csv_files:
    label_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=50_000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        labels = chunk["Label"].astype(str).str.strip()

        total_rows += len(chunk)

        counts = labels.value_counts()
        for label, count in counts.items():
            label_counts[label] = (
                label_counts.get(label, 0) + int(count)
            )

    print(f"\nFile: {file_path.name}")
    print(f"Total rows: {total_rows:,}")

    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count:,}")

print("\nAnalysis completed.")
