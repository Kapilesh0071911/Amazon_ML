import pandas as pd
import re

from rapidfuzz import process, fuzz


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/processed"

S1_FILE = f"{DATA_DIR}/train_source1_processed.tsv"
S2_FILE = f"{DATA_DIR}/train_source2_processed.tsv"
S3_FILE = f"{DATA_DIR}/train_source3_processed.tsv"

TRUTH_FILE = "dataset/train/train_ground_truth.tsv"

EVAL_SIZE = 1000
TOP_K = 30


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
# MULTIPLE BLOCKING KEYS
# ============================================================

def get_block_keys(name, address):

    name = normalize_value(name)
    address = normalize_value(address)

    keys = set()

    # ----------------------------
    # NAME BLOCKS
    # ----------------------------

    if len(name) >= 3:
        keys.add("NF3_" + name[:3])

    if len(name) >= 4:
        keys.add("NF4_" + name[:4])

    if len(name) >= 3:
        keys.add("NL3_" + name[-3:])

    # ----------------------------
    # ADDRESS BLOCKS
    # ----------------------------

    if len(address) >= 3:
        keys.add("AF3_" + address[:3])

    if len(address) >= 4:
        keys.add("AF4_" + address[:4])

    if len(address) >= 3:
        keys.add("AL3_" + address[-3:])

    return keys


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1 sample...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    nrows=EVAL_SIZE
)

s1["name_normalized"] = (
    s1["name_normalized"]
    .fillna("")
    .astype(str)
)

s1["address_normalized"] = (
    s1["address_normalized"]
    .fillna("")
    .astype(str)
)

print(
    "S1 records:",
    len(s1)
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

truth = pd.read_csv(
    TRUTH_FILE,
    sep="\t",
    dtype=str
)

truth_lookup = dict(
    zip(
        truth["source1_entity_id"],
        truth["matched_entity_ids"]
    )
)


# ============================================================
# GET TRUE MATCH IDS
# ============================================================

true_match_ids = set()

for s1_id in s1["entity_id"]:

    matched = truth_lookup.get(
        s1_id,
        ""
    )

    if pd.notna(matched) and str(matched).strip():

        true_match_ids.update(
            str(matched).split(",")
        )


print(
    "True match IDs:",
    len(true_match_ids)
)


# ============================================================
# LOAD TRUE MATCHES FROM SOURCE 2
# ============================================================

print("\nLoading true matches...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str
)

s2 = s2[
    s2["entity_id"].isin(
        true_match_ids
    )
][
    [
        "entity_id",
        "name_normalized",
        "address_normalized"
    ]
]

print(
    "True S2 matches:",
    len(s2)
)


# ============================================================
# LOAD TRUE MATCHES FROM SOURCE 3
# ============================================================

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str
)

s3 = s3[
    s3["entity_id"].isin(
        true_match_ids
    )
][
    [
        "entity_id",
        "name_normalized",
        "address_normalized"
    ]
]

print(
    "True S3 matches:",
    len(s3)
)


# ============================================================
# ADD RANDOM CANDIDATES
# ============================================================

print("\nAdding random candidates...")

random_s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    skiprows=lambda x: x > 0 and x % 10000 != 0,
    nrows=20000
)[
    [
        "entity_id",
        "name_normalized",
        "address_normalized"
    ]
]

random_s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    skiprows=lambda x: x > 0 and x % 10000 != 0,
    nrows=20000
)[
    [
        "entity_id",
        "name_normalized",
        "address_normalized"
    ]
]

random_candidates = pd.concat(
    [
        random_s2,
        random_s3
    ],
    ignore_index=True
)

random_candidates = random_candidates[
    ~random_candidates["entity_id"].isin(
        true_match_ids
    )
]


# ============================================================
# FINAL TEST CANDIDATES
# ============================================================

candidate_sample = pd.concat(
    [
        s2,
        s3,
        random_candidates
    ],
    ignore_index=True
).drop_duplicates(
    subset=["entity_id"]
)

print(
    "Total candidates:",
    len(candidate_sample)
)


# ============================================================
# BUILD MULTI-BLOCK INDEX
# ============================================================

print("\nBuilding name + address block index...")

block_index = {}

for row in candidate_sample.itertuples(
    index=False
):

    entity_id = row.entity_id

    keys = get_block_keys(
        row.name_normalized,
        row.address_normalized
    )

    for key in keys:

        if key not in block_index:

            block_index[key] = []

        block_index[key].append(
            (
                entity_id,
                row.name_normalized,
                row.address_normalized
            )
        )


print(
    "Number of blocks:",
    len(block_index)
)


# ============================================================
# EVALUATE
# ============================================================

hits = 0
evaluated = 0

for row in s1.itertuples(
    index=False
):

    s1_id = row.entity_id

    true_string = truth_lookup.get(
        s1_id,
        ""
    )

    if pd.isna(true_string) or not str(true_string).strip():
        continue

    true_ids = set(
        str(true_string).split(",")
    )


    # --------------------------------------------------------
    # Retrieve candidates using name + address blocks
    # --------------------------------------------------------

    candidates_dict = {}

    keys = get_block_keys(
        row.name_normalized,
        row.address_normalized
    )

    for key in keys:

        for entity_id, name, address in block_index.get(
            key,
            []
        ):

            candidates_dict[
                entity_id
            ] = (
                name,
                address
            )


    candidates = list(
        candidates_dict.items()
    )

    if not candidates:

        evaluated += 1
        continue


    # --------------------------------------------------------
    # Fuzzy name matching
    # --------------------------------------------------------

    candidate_names = [
        item[1][0]
        for item in candidates
    ]

    matches = process.extract(
        row.name_normalized,
        candidate_names,
        scorer=fuzz.ratio,
        limit=TOP_K
    )


    generated_ids = set()

    for match in matches:

        position = match[2]

        generated_ids.add(
            candidates[position][0]
        )


    if true_ids.intersection(
        generated_ids
    ):

        hits += 1

    evaluated += 1


# ============================================================
# RESULT
# ============================================================

recall = (
    hits / evaluated
    if evaluated
    else 0
)


print(
    "\n========================================"
)

print(
    "NAME + ADDRESS BLOCKING RECALL"
)

print(
    "========================================"
)

print(
    "S1 records evaluated:",
    evaluated
)

print(
    "Records with true match in top 30:",
    hits
)

print(
    "Recall@30:",
    round(
        recall * 100,
        2
    ),
    "%"
)