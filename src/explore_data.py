
import pandas as pd
from pathlib import Path

file_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

print("Reading a small sample of the dataset...")

df = pd.read_csv(file_path, nrows=5)

print("\nDataset columns:")
for column in df.columns:
    print(repr(column))

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nSample shape:", df.shape)

label_columns = [
    column for column in df.columns
    if column.strip().lower() == "label"
]

if label_columns:
    print("\nLabels in this sample:")
    print(df[label_columns[0]].value_counts())
else:
    print("\nWARNING: Label column was not found.")
