
from pathlib import Path
from collections import Counter
import pandas as pd

data_dir = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE"
)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "multiclass_selected_raw.csv"

selected_files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

selected_labels = {
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "FTP-Patator",
    "SSH-Patator",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
}

if output_file.exists():
    output_file.unlink()

label_counts = Counter()
total_saved = 0
first_chunk = True

print("Preparing selected multiclass dataset...")
print("=" * 65)

for filename in selected_files:
    file_path = data_dir / filename
    print(f"\nProcessing: {filename}")

    file_saved = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=50_000,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        # Normalize column names before accessing Label.
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(
                f"Label column not found in {filename}"
            )

        chunk["Label"] = (
            chunk["Label"]
            .astype("string")
            .str.strip()
        )

        # Keep only the selected classes.
        filtered = chunk[
            chunk["Label"].isin(selected_labels)
        ].copy()

        if filtered.empty:
            continue

        # Preserve the source file for later evaluation.
        filtered["Source_File"] = filename

        filtered.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8",
        )

        first_chunk = False

        counts = filtered["Label"].value_counts()

        for label, count in counts.items():
            label_counts[str(label)] += int(count)

        file_saved += len(filtered)
        total_saved += len(filtered)

    print(f"Selected rows: {file_saved:,}")

print("\n" + "=" * 65)
print("Dataset preparation completed.")
print(f"Output file: {output_file}")
print(f"Total selected rows: {total_saved:,}")

print("\nFinal label distribution:")
for label, count in label_counts.most_common():
    print(f"{label}: {count:,}")

print("\nTotal classes:", len(label_counts))
