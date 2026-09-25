import os
import re
import pandas as pd


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

# Process Source 1 in chunks
S1_CHUNK_SIZE = 10000

# Read S2/S3 in chunks
CANDIDATE_CHUNK_SIZE = 100000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_value(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


# ============================================================
# BLOCKING KEYS
# ============================================================

def get_block_keys(name, address):

    name = normalize_value(name)
    address = normalize_value(address)

    keys = set()

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    if len(name) >= 3:
        keys.add(
            "NF3_" + name[:3]
        )

    if len(name) >= 4:
        keys.add(
            "NF4_" + name[:4]
        )

    if len(name) >= 3:
        keys.add(
            "NL3_" + name[-3:]
        )

    # --------------------------------------------------------
    # ADDRESS BLOCKS
    # --------------------------------------------------------

    if len(address) >= 3:
        keys.add(
            "AF3_" + address[:3]
        )

    if len(address) >= 4:
        keys.add(
            "AF4_" + address[:4]
        )

    if len(address) >= 3:
        keys.add(
            "AL3_" + address[-3:]
        )

    return keys


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
            f"Missing input file: {file_path}"
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
# BUILD BLOCK INDEX
# ============================================================

print("\nBuilding block index...")


block_index = {}


def index_source(
    file_path,
    source_name
):

    total_rows = 0
    chunk_number = 0

    print(
        f"\nIndexing {source_name}..."
    )

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "name_normalized",
            "address_normalized"
        ],
        chunksize=CANDIDATE_CHUNK_SIZE
    ):

        chunk_number += 1

        chunk[
            "name_normalized"
        ] = (
            chunk[
                "name_normalized"
            ]
            .fillna("")
            .astype(str)
        )

        chunk[
            "address_normalized"
        ] = (
            chunk[
                "address_normalized"
            ]
            .fillna("")
            .astype(str)
        )


        for row in chunk.itertuples(
            index=False
        ):

            keys = get_block_keys(
                row.name_normalized,
                row.address_normalized
            )

            for key in keys:

                if key not in block_index:

                    block_index[key] = []

                block_index[key].append(
                    (
                        row.entity_id,
                        source_name
                    )
                )


        total_rows += len(chunk)

        print(
            f"  {source_name} chunk "
            f"{chunk_number} processed "
            f"({len(chunk):,} rows)"
        )


    print(
        f"{source_name} complete: "
        f"{total_rows:,} rows"
    )


# ============================================================
# INDEX S2
# ============================================================

index_source(
    S2_FILE,
    "S2"
)


# ============================================================
# INDEX S3
# ============================================================

index_source(
    S3_FILE,
    "S3"
)


print(
    "\n========================================"
)

print(
    "BLOCK INDEX READY"
)

print(
    "Total block keys:",
    f"{len(block_index):,}"
)

print(
    "========================================"
)


# ============================================================
# PROCESS SOURCE 1
# ============================================================

print(
    "\nLoading Source 1..."
)


total_s1_rows = sum(
    1
    for _ in open(
        S1_FILE,
        encoding="utf-8"
    )
) - 1


print(
    "Source 1 rows:",
    f"{total_s1_rows:,}"
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


# ============================================================
# OUTPUT HEADER
# ============================================================

header_written = False

total_pairs = 0
processed_rows = 0


# ============================================================
# STREAM SOURCE 1
# ============================================================

for s1_chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized",
        "address_normalized"
    ],
    chunksize=S1_CHUNK_SIZE
):

    s1_chunk[
        "name_normalized"
    ] = (
        s1_chunk[
            "name_normalized"
        ]
        .fillna("")
        .astype(str)
    )

    s1_chunk[
        "address_normalized"
    ] = (
        s1_chunk[
            "address_normalized"
        ]
        .fillna("")
        .astype(str)
    )


    candidate_rows = []


    # ========================================================
    # GENERATE CANDIDATES
    # ========================================================

    for row in s1_chunk.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        keys = get_block_keys(
            row.name_normalized,
            row.address_normalized
        )


        candidate_ids = set()


        for key in keys:

            for (
                candidate_id,
                source
            ) in block_index.get(
                key,
                []
            ):

                candidate_ids.add(
                    (
                        candidate_id,
                        source
                    )
                )


        for (
            candidate_id,
            source
        ) in candidate_ids:

            candidate_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": source
                }
            )


    # ========================================================
    # WRITE CHUNK
    # ========================================================

    if candidate_rows:

        output_df = pd.DataFrame(
            candidate_rows
        ).drop_duplicates()


        output_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=not header_written,
            
        )


        header_written = True

        total_pairs += len(
            output_df
        )


    processed_rows += len(
        s1_chunk
    )


    print(
        f"Processed S1 records: "
        f"{processed_rows:,} / "
        f"{total_s1_rows:,}"
        f" | Candidate pairs: "
        f"{total_pairs:,}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n========================================"
)

print(
    "FULL BLOCKING COMPLETED"
)

print(
    "========================================"
)

print(
    "Source 1 records:",
    f"{total_s1_rows:,}"
)

print(
    "Candidate pairs:",
    f"{total_pairs:,}"
)

print(
    "Output file:",
    OUTPUT_FILE
)