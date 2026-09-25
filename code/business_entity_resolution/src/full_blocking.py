import os
import re
import pandas as pd
import numpy as np

from rapidfuzz import process, fuzz


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/processed"
OUTPUT_DIR = "output"

S1_FILE = os.path.join(
    DATA_DIR,
    "train_source1_processed.tsv"
)

S2_FILE = os.path.join(
    DATA_DIR,
    "train_source2_processed.tsv"
)

S3_FILE = os.path.join(
    DATA_DIR,
    "train_source3_processed.tsv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

TOP_K = 30

S1_CHUNK_SIZE = 5000

# First N characters used for blocking
PREFIX_LENGTH = 3

# Maximum number of candidates retained
MAX_BLOCK_CANDIDATES = 500


# ============================================================
# HELPER
# ============================================================

def get_prefix(value):

    if pd.isna(value):
        return ""

    value = str(value)

    # Keep only alphanumeric characters
    value = re.sub(
        r"[^a-z0-9]",
        "",
        value.lower()
    )

    return value[:PREFIX_LENGTH]


# ============================================================
# CHECK FILES
# ============================================================

print("Checking input files...")

for file_path in [
    S1_FILE,
    S2_FILE,
    S3_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    print(
        f"Found: {file_path}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    os.remove(
        OUTPUT_FILE
    )

    print(
        f"Removed old output: {OUTPUT_FILE}"
    )


# ============================================================
# BUILD CANDIDATE INDEX
# ============================================================

print("\nBuilding candidate index...")

candidate_index = {}


# ------------------------------------------------------------
# Read S2
# ------------------------------------------------------------

print("\nIndexing Source 2...")

s2_reader = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized"
    ],
    chunksize=100_000
)

s2_count = 0

for chunk in s2_reader:

    chunk[
        "name_normalized"
    ] = (
        chunk[
            "name_normalized"
        ]
        .fillna("")
        .astype(str)
    )

    for row in chunk.itertuples(
        index=False
    ):

        entity_id = row.entity_id
        name = row.name_normalized

        prefix = get_prefix(
            name
        )

        if not prefix:
            continue

        if prefix not in candidate_index:

            candidate_index[
                prefix
            ] = []

        candidate_index[
            prefix
        ].append(
            (
                entity_id,
                name,
                "S2"
            )
        )

    s2_count += len(chunk)

    print(
        f"  Indexed S2: "
        f"{s2_count:,}"
    )


# ------------------------------------------------------------
# Read S3
# ------------------------------------------------------------

print("\nIndexing Source 3...")

s3_reader = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized"
    ],
    chunksize=100_000
)

s3_count = 0

for chunk in s3_reader:

    chunk[
        "name_normalized"
    ] = (
        chunk[
            "name_normalized"
        ]
        .fillna("")
        .astype(str)
    )

    for row in chunk.itertuples(
        index=False
    ):

        entity_id = row.entity_id
        name = row.name_normalized

        prefix = get_prefix(
            name
        )

        if not prefix:
            continue

        if prefix not in candidate_index:

            candidate_index[
                prefix
            ] = []

        candidate_index[
            prefix
        ].append(
            (
                entity_id,
                name,
                "S3"
            )
        )

    s3_count += len(chunk)

    print(
        f"  Indexed S3: "
        f"{s3_count:,}"
    )


print(
    "\nCandidate index built."
)

print(
    "Number of prefix buckets:",
    len(candidate_index)
)


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print(
    "\nLoading Source 1..."
)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

s1[
    "name_normalized"
] = (
    s1[
        "name_normalized"
    ]
    .fillna("")
    .astype(str)
)

print(
    "Source 1 rows:",
    f"{len(s1):,}"
)


# ============================================================
# PROCESS S1 IN CHUNKS
# ============================================================

total_output = 0

for start in range(
    0,
    len(s1),
    S1_CHUNK_SIZE
):

    end = min(
        start + S1_CHUNK_SIZE,
        len(s1)
    )

    s1_chunk = s1.iloc[
        start:end
    ]

    print(
        "\n========================================"
    )

    print(
        f"Processing S1 rows "
        f"{start:,} - {end:,}"
    )

    print(
        "========================================"
    )

    output_rows = []


    for row in s1_chunk.itertuples(
        index=False
    ):

        s1_id = row.entity_id
        s1_name = row.name_normalized

        prefix = get_prefix(
            s1_name
        )

        if not prefix:
            continue


        # ----------------------------------------------------
        # Retrieve only matching prefix bucket
        # ----------------------------------------------------

        candidates = candidate_index.get(
            prefix,
            []
        )


        if not candidates:
            continue


        # ----------------------------------------------------
        # Limit very large buckets
        # ----------------------------------------------------

        if len(candidates) > MAX_BLOCK_CANDIDATES:

            candidates = candidates[
                :MAX_BLOCK_CANDIDATES
            ]


        # ----------------------------------------------------
        # Extract candidate names
        # ----------------------------------------------------

        candidate_names = [
            item[1]
            for item in candidates
        ]


        # ----------------------------------------------------
        # Fuzzy matching inside the block
        # ----------------------------------------------------

        matches = process.extract(
            s1_name,
            candidate_names,
            scorer=fuzz.ratio,
            limit=TOP_K
        )


        for match in matches:

            matched_name = match[0]
            score = match[1]
            candidate_position = match[2]

            candidate_id = candidates[
                candidate_position
            ][0]

            candidate_source = candidates[
                candidate_position
            ][2]


            output_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": candidate_source,
                    "similarity": score / 100.0,
                    "rank": 0
                }
            )


    # ========================================================
    # RANK CANDIDATES
    # ========================================================

    output_df = pd.DataFrame(
        output_rows
    )


    if not output_df.empty:

        output_df = (
            output_df
            .sort_values(
                [
                    "source1_entity_id",
                    "similarity"
                ],
                ascending=[
                    True,
                    False
                ]
            )
        )

        output_df[
            "rank"
        ] = (
            output_df
            .groupby(
                "source1_entity_id"
            )
            .cumcount()
            + 1
        )


        output_df = output_df[
            output_df["rank"] <= TOP_K
        ]


        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        output_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=not os.path.exists(
                OUTPUT_FILE
            )
        )


        total_output += len(
            output_df
        )


    print(
        f"Written candidate pairs: "
        f"{len(output_df):,}"
    )

    print(
        f"Total candidate pairs so far: "
        f"{total_output:,}"
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "BLOCKING COMPLETED"
)

print(
    "========================================"
)

print(
    f"Candidate pairs: {total_output:,}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)