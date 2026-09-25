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

def normalize_name(value):

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

def get_block_keys(name):

    name = normalize_name(name)

    if not name:
        return set()

    keys = set()

    # First 3 characters
    if len(name) >= 3:
        keys.add(
            "F3_" + name[:3]
        )

    # First 4 characters
    if len(name) >= 4:
        keys.add(
            "F4_" + name[:4]
        )

    # Last 3 characters
    if len(name) >= 3:
        keys.add(
            "L3_" + name[-3:]
        )

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
# LOAD TRUE MATCHES
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
        "name_normalized"
    ]
]

print(
    "True S2 matches:",
    len(s2)
)


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
        "name_normalized"
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
        "name_normalized"
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
        "name_normalized"
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

print("\nBuilding multi-block index...")

block_index = {}

for row in candidate_sample.itertuples(
    index=False
):

    entity_id = row.entity_id
    name = row.name_normalized

    keys = get_block_keys(name)

    for key in keys:

        if key not in block_index:

            block_index[key] = []

        block_index[key].append(
            (
                entity_id,
                name
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
    s1_name = row.name_normalized

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
    # Get candidates from ALL blocking keys
    # --------------------------------------------------------

    candidates_dict = {}

    keys = get_block_keys(
        s1_name
    )

    for key in keys:

        for entity_id, name in block_index.get(
            key,
            []
        ):

            candidates_dict[
                entity_id
            ] = name


    candidates = list(
        candidates_dict.items()
    )


    if not candidates:

        evaluated += 1
        continue


    # --------------------------------------------------------
    # Fuzzy matching
    # --------------------------------------------------------

    candidate_names = [
        item[1]
        for item in candidates
    ]

    matches = process.extract(
        s1_name,
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
    "MULTI-BLOCK RECALL"
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