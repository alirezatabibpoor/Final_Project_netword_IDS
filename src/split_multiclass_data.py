
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

input_file = Path(
    "data/processed/multiclass_deduplicated.csv"
)

train_file = Path(
    "data/processed/multiclass_train.csv"
)

test_file = Path(
    "data/processed/multiclass_test.csv"
)

print("Loading deduplicated dataset...")

df = pd.read_csv(input_file, low_memory=False)

print(f"Total rows: {len(df):,}")
print(f"Total classes: {df['Label'].nunique()}")

# Features and target
X = df.drop(columns=["Label", "Source_File"])
y = df["Label"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preserve source-file information for later analysis
source_train = df.loc[X_train.index, "Source_File"]
source_test = df.loc[X_test.index, "Source_File"]

train_df = X_train.copy()
train_df["Label"] = y_train
train_df["Source_File"] = source_train

test_df = X_test.copy()
test_df["Label"] = y_test
test_df["Source_File"] = source_test

# Save datasets
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print("\nSplit completed")
print("=" * 60)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(
    f"Training percentage: "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    f"Testing percentage: "
    f"{len(test_df) / len(df) * 100:.2f}%"
)

print("\nTraining class distribution:")
print(train_df["Label"].value_counts().to_string())

print("\nTesting class distribution:")
print(test_df["Label"].value_counts().to_string())

print("\nSaved files:")
print(train_file)
print(test_file)
