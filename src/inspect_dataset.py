
from pathlib import Path

data_dir = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE"
)

csv_files = sorted(data_dir.glob("*.csv"))

print("CICIDS2017 CSV files")
print("=" * 60)

if not csv_files:
    print("No CSV files found. Check the dataset path.")
else:
    for index, file_path in enumerate(csv_files, start=1):
        size_mb = file_path.stat().st_size / (1024 * 1024)

        print(
            f"{index:2}. {file_path.name} "
            f"({size_mb:.2f} MB)"
        )

    print("-" * 60)
    print(f"Total CSV files: {len(csv_files)}")
