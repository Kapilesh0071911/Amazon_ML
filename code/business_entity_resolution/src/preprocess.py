import os
import re
import pandas as pd
from unidecode import unidecode


TRAIN_DIR = "dataset/train"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = unidecode(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def preprocess_file(input_file, output_file):

    print(f"\nReading: {input_file}")

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print(f"Rows: {len(df):,}")

    df["name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_file}")


# Source 1
preprocess_file(
    f"{TRAIN_DIR}/train_source1.tsv",
    f"{OUTPUT_DIR}/train_source1_processed.tsv"
)

# Source 2
preprocess_file(
    f"{TRAIN_DIR}/train_source2.tsv",
    f"{OUTPUT_DIR}/train_source2_processed.tsv"
)

# Source 3
preprocess_file(
    f"{TRAIN_DIR}/train_source3.tsv",
    f"{OUTPUT_DIR}/train_source3_processed.tsv"
)

print("\n================================")
print("PREPROCESSING COMPLETE")
print("================================")