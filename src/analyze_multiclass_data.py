
from pathlib import Path
from collections import Counter
import pandas as pd

data_dir = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE"
)

selected_files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

label_counts = Counter()

print("Analyzing selected CICIDS2017 files...")
print("=" * 75)

for filename in selected_files:
    file_path = data_dir / filename
    file_counts = Counter()

    print(f"\nFile: {filename}")

    # Read the file in chunks, then normalize column names.
    for chunk in pd.read_csv(
        file_path,
        chunksize=100_000,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(
                f"Label column not found in {filename}. "
                f"Columns: {chunk.columns.tolist()}"
            )

        labels = chunk["Label"].astype("string").str.strip()
        counts = labels.value_counts(dropna=False)

        for label, count in counts.items():
            label_name = (
                "<MISSING>"
                if pd.isna(label)
                else str(label)
            )
            file_counts[label_name] += int(count)
            label_counts[label_name] += int(count)

    for label, count in file_counts.most_common():
        print(f"  {label}: {count:,}")

print("\n" + "=" * 75)
print("Combined label distribution")
print("=" * 75)

for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print("\nAnalysis completed.")
