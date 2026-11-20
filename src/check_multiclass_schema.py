
from pathlib import Path
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

reference_columns = None

print("Checking selected CICIDS2017 files...")
print("=" * 65)

for filename in selected_files:
    file_path = data_dir / filename

    print(f"\nFile: {filename}")

    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    # فقط نام ستون‌ها و پنج ردیف نمونه را می‌خوانیم
    df = pd.read_csv(
        file_path,
        nrows=5,
        low_memory=False
    )

    # یکسان‌سازی فاصله‌های ابتدا و انتهای نام ستون‌ها
    df.columns = df.columns.str.strip()

    print(f"Columns: {len(df.columns)}")
    print(f"Label column exists: {'Label' in df.columns}")

    feature_columns = [
        column for column in df.columns
        if column != "Label"
    ]

    if reference_columns is None:
        reference_columns = feature_columns
        print("Selected as reference schema.")
    else:
        missing = sorted(
            set(reference_columns) - set(feature_columns)
        )
        extra = sorted(
            set(feature_columns) - set(reference_columns)
        )

        print(f"Missing feature columns: {missing}")
        print(f"Extra feature columns: {extra}")

print("\nSchema check completed.")
