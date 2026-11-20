
import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

output_path = Path("data/processed/ddos_clean.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("Starting dataset cleaning...")
print("-" * 50)

total_rows = 0
removed_rows = 0
saved_rows = 0
first_chunk = True

# فایل خروجی قبلی را حذف می‌کنیم تا داده‌ها تکراری نشوند.
if output_path.exists():
    output_path.unlink()

for chunk in pd.read_csv(input_path, chunksize=100_000):
    # استانداردسازی نام ستون‌ها و برچسب‌ها
    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    total_rows += len(chunk)

    # تبدیل مقادیر مثبت و منفی بی‌نهایت به مقدار خالی
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    # حذف ردیف‌هایی که حداقل یک مقدار خالی دارند
    before = len(chunk)
    chunk = chunk.dropna()
    removed_rows += before - len(chunk)

    saved_rows += len(chunk)

    # ذخیره تدریجی داده‌های پاک‌سازی‌شده
    chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print(f"Original rows: {total_rows:,}")
print(f"Removed rows: {removed_rows:,}")
print(f"Saved rows: {saved_rows:,}")
print(f"Output file: {output_path}")

print("\nCleaning completed successfully.")
