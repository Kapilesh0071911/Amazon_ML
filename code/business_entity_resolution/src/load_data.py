import pandas as pd

# Project paths
TRAIN = "dataset/train"

# Load training datasets
train_s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t"
)

truth = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    keep_default_na=False
)

# Display basic information
print("\n===== SOURCE 1 =====")
print(train_s1.head())
print("Shape:", train_s1.shape)

print("\n===== SOURCE 2 =====")
print(train_s2.head())
print("Shape:", train_s2.shape)

print("\n===== SOURCE 3 =====")
print(train_s3.head())
print("Shape:", train_s3.shape)

print("\n===== GROUND TRUTH =====")
print(truth.head())
print("Shape:", truth.shape)