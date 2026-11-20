
import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path(
    "data/raw/MachineLearningCSV/"
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

output_path = Path("data/processed/portscan_clean.csv")

print("Preparing PortScan dataset...")
print("-" * 50)

# خواندن داده‌ها
df = pd.read_csv(input_path, low_memory=False)

# یکسان‌سازی نام ستون‌ها و برچسب‌ها
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

original_rows = len(df)

# تبدیل ویژگی‌ها به مقادیر عددی
feature_columns = [col for col in df.columns if col != "Label"]

for column in feature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# تبدیل مقادیر بی‌نهایت به مقدار گمشده
df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# حذف ردیف‌هایی که ویژگی نامعتبر یا گمشده دارند
df = df.dropna(subset=feature_columns)

# حذف نمونه‌های کاملاً تکراری از نظر ویژگی‌ها
# برچسب نیز حفظ می‌شود و ردیف‌های دارای برچسب متفاوت
# صرفاً به دلیل یکسان بودن ویژگی‌ها حذف نمی‌شوند.
df = df.drop_duplicates()

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nPreparation completed.")
print(f"Original rows: {original_rows:,}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Rows removed: {original_rows - len(df):,}")

print("\nLabel distribution:")
print(df["Label"].value_counts())

print(f"\nSaved to: {output_path}")
