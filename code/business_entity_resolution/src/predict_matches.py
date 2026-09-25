import os
import time
import heapq
import joblib
import pandas as pd
import numpy as np

from rapidfuzz.fuzz import ratio, token_set_ratio


# ============================================================
# CONFIG
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

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "entity_resolution_model.joblib"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)


# ============================================================
# TEST SETTINGS
# ============================================================

# START WITH 1000
TEST_MODE = True

TEST_ROWS = 1000


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

CHUNK_SIZE = 100000

# Maximum candidates after cheap blocking
MAX_CANDIDATES = 30

# Maximum candidates from each individual block
MAX_PER_BLOCK = 20

# Cheap fuzzy shortlist size
SHORTLIST_SIZE = 15

# Matching threshold
MATCH_THRESHOLD = 0.50


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FAST ENTITY RESOLUTION PREDICTION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    S1_FILE,
    S2_FILE,
    S3_FILE,
    MODEL_FILE
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    print("Found:", path)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

package = joblib.load(
    MODEL_FILE
)

if isinstance(package, dict):

    model = package["model"]

    feature_columns = package.get(
        "features",
        [
            "name_ratio",
            "name_token_ratio",
            "address_ratio",
            "address_token_ratio",
            "country_match",
            "exact_name",
            "exact_address",
            "name_length_difference",
            "address_length_difference"
        ]
    )

else:

    model = package

    feature_columns = [
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_match",
        "exact_name",
        "exact_address",
        "name_length_difference",
        "address_length_difference"
    ]

print("Model loaded successfully.")


# ============================================================
# NORMALIZATION
# ============================================================

def clean(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ============================================================
# BLOCK KEY GENERATION
# ============================================================

def block_keys(name, address):

    keys = set()

    name = clean(name)
    address = clean(address)

    # --------------------------------------------------------
    # NAME PREFIX
    # --------------------------------------------------------

    if len(name) >= 2:

        keys.add(
            "N2:" + name[:2]
        )

    if len(name) >= 3:

        keys.add(
            "N3:" + name[:3]
        )

    if len(name) >= 4:

        keys.add(
            "N4:" + name[:4]
        )


    # --------------------------------------------------------
    # NAME SUFFIX
    # --------------------------------------------------------

    if len(name) >= 3:

        keys.add(
            "NX3:" + name[-3:]
        )


    # --------------------------------------------------------
    # ADDRESS PREFIX
    # --------------------------------------------------------

    if len(address) >= 2:

        keys.add(
            "A2:" + address[:2]
        )

    if len(address) >= 3:

        keys.add(
            "A3:" + address[:3]
        )

    if len(address) >= 4:

        keys.add(
            "A4:" + address[:4]
        )


    # --------------------------------------------------------
    # COMBINED BLOCK
    # --------------------------------------------------------

    if (
        len(name) >= 3
        and
        len(address) >= 3
    ):

        keys.add(
            "NA:"
            + name[:3]
            + ":"
            + address[:3]
        )


    return keys


# ============================================================
# DATA STRUCTURES
# ============================================================

block_index = {}

records = {}


# ============================================================
# INDEX SOURCE
# ============================================================

def index_source(
    file_path,
    source
):

    print(
        f"\nIndexing {source}..."
    )

    total = 0
    chunk_no = 0

    usecols = [
        "entity_id",
        "name_normalized",
        "address_normalized",
        "country"
    ]

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        chunk_no += 1

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = str(
                row.entity_id
            )

            name = clean(
                row.name_normalized
            )

            address = clean(
                row.address_normalized
            )

            country = clean(
                row.country
            )

            key = (
                source,
                entity_id
            )

            records[key] = (
                name,
                address,
                country
            )

            for block in block_keys(
                name,
                address
            ):

                bucket = block_index.setdefault(
                    block,
                    []
                )

                # Prevent extremely large buckets
                if len(bucket) < 1000:

                    bucket.append(
                        key
                    )


        total += len(chunk)

        print(
            f"  {source} chunk {chunk_no}: "
            f"{len(chunk):,} rows"
        )


    print(
        f"{source} complete: "
        f"{total:,} rows"
    )


# ============================================================
# BUILD INDEX
# ============================================================

print("\n" + "=" * 70)
print("BUILDING FAST BLOCK INDEX")
print("=" * 70)

index_source(
    S2_FILE,
    "S2"
)

index_source(
    S3_FILE,
    "S3"
)


print("\n" + "=" * 70)
print("INDEX READY")
print("=" * 70)

print(
    "Block keys:",
    f"{len(block_index):,}"
)

print(
    "Indexed records:",
    f"{len(records):,}"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(
    name,
    address
):

    candidates = set()

    keys = block_keys(
        name,
        address
    )

    # --------------------------------------------------------
    # Prioritize stronger blocks
    # --------------------------------------------------------

    priority = []

    for key in keys:

        if key.startswith("NA:"):
            priority.append((0, key))

        elif key.startswith("N4:"):
            priority.append((1, key))

        elif key.startswith("A4:"):
            priority.append((2, key))

        elif key.startswith("N3:"):
            priority.append((3, key))

        elif key.startswith("A3:"):
            priority.append((4, key))

        else:
            priority.append((5, key))


    priority.sort()


    # --------------------------------------------------------
    # Pull only limited candidates from each block
    # --------------------------------------------------------

    for _, key in priority:

        bucket = block_index.get(
            key,
            []
        )

        if not bucket:
            continue


        # Do not process giant buckets
        if len(bucket) > MAX_PER_BLOCK:

            selected = bucket[
                :MAX_PER_BLOCK
            ]

        else:

            selected = bucket


        for candidate in selected:

            candidates.add(
                candidate
            )


        if len(candidates) >= MAX_CANDIDATES:

            break


    return list(
        candidates
    )[:MAX_CANDIDATES]


# ============================================================
# FAST PRE-SCORE
# ============================================================

def cheap_score(
    name1,
    address1,
    name2,
    address2
):

    # Name is more important
    name_score = ratio(
        name1,
        name2
    )

    address_score = ratio(
        address1,
        address2
    )

    return (
        0.65 * name_score
        +
        0.35 * address_score
    )


# ============================================================
# FEATURES
# ============================================================

def make_features(
    s1_name,
    s1_address,
    s1_country,
    candidate
):

    c_name = candidate[0]
    c_address = candidate[1]
    c_country = candidate[2]


    name_ratio = (
        ratio(
            s1_name,
            c_name
        )
        / 100.0
    )


    name_token_ratio = (
        token_set_ratio(
            s1_name,
            c_name
        )
        / 100.0
    )


    address_ratio = (
        ratio(
            s1_address,
            c_address
        )
        / 100.0
    )


    address_token_ratio = (
        token_set_ratio(
            s1_address,
            c_address
        )
        / 100.0
    )


    country_match = int(
        bool(s1_country)
        and
        bool(c_country)
        and
        s1_country == c_country
    )


    exact_name = int(
        bool(s1_name)
        and
        s1_name == c_name
    )


    exact_address = int(
        bool(s1_address)
        and
        s1_address == c_address
    )


    name_length_difference = abs(
        len(s1_name)
        -
        len(c_name)
    )


    address_length_difference = abs(
        len(s1_address)
        -
        len(c_address)
    )


    return [
        name_ratio,
        name_token_ratio,
        address_ratio,
        address_token_ratio,
        country_match,
        exact_name,
        exact_address,
        name_length_difference,
        address_length_difference
    ]


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(row):

    s1_id = str(
        row.entity_id
    )

    s1_name = clean(
        row.name_normalized
    )

    s1_address = clean(
        row.address_normalized
    )

    s1_country = clean(
        row.country
    )


    # --------------------------------------------------------
    # BLOCKING
    # --------------------------------------------------------

    candidates = get_candidates(
        s1_name,
        s1_address
    )


    if not candidates:

        return (
            s1_id,
            "",
            0.0,
            0
        )


    # --------------------------------------------------------
    # CHEAP PRE-SCORING
    # --------------------------------------------------------

    scored = []


    for candidate_key in candidates:

        candidate = records[
            candidate_key
        ]

        score = cheap_score(
            s1_name,
            s1_address,
            candidate[0],
            candidate[1]
        )

        scored.append(
            (
                score,
                candidate_key
            )
        )


    # --------------------------------------------------------
    # KEEP ONLY TOP FEW
    # --------------------------------------------------------

    if len(scored) > SHORTLIST_SIZE:

        scored = heapq.nlargest(
            SHORTLIST_SIZE,
            scored,
            key=lambda x: x[0]
        )


    # --------------------------------------------------------
    # RANDOM FOREST FEATURES
    # --------------------------------------------------------

    candidate_keys = [
        x[1]
        for x in scored
    ]


    feature_rows = []


    for candidate_key in candidate_keys:

        candidate = records[
            candidate_key
        ]

        feature_rows.append(
            make_features(
                s1_name,
                s1_address,
                s1_country,
                candidate
            )
        )


    # --------------------------------------------------------
    # BATCH MODEL PREDICTION
    # --------------------------------------------------------

    X = pd.DataFrame(
        feature_rows,
        columns=feature_columns
    )


    probabilities = model.predict_proba(
        X
    )[:, 1]


    best_index = int(
        np.argmax(
            probabilities
        )
    )


    best_probability = float(
        probabilities[
            best_index
        ]
    )


    best_key = candidate_keys[
        best_index
    ]


    if (
        best_probability
        >=
        MATCH_THRESHOLD
    ):

        matched_id = best_key[1]

    else:

        matched_id = ""


    return (
        s1_id,
        matched_id,
        best_probability,
        len(candidates)
    )


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("STARTING PREDICTION")
print("=" * 70)

if TEST_MODE:

    print(
        f"TEST MODE: "
        f"{TEST_ROWS:,} records"
    )

else:

    print(
        "FULL MODE: ALL SOURCE-1 RECORDS"
    )


# Remove old result
if os.path.exists(
    OUTPUT_FILE
):

    os.remove(
        OUTPUT_FILE
    )


start_time = time.time()

processed = 0
matched = 0
total_candidates = 0

header_written = False


# ============================================================
# LOAD S1
# ============================================================

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized",
        "address_normalized",
        "country"
    ],
    chunksize=1000
):

    chunk = chunk.fillna("")


    # --------------------------------------------------------
    # TEST LIMIT
    # --------------------------------------------------------

    if TEST_MODE:

        remaining = (
            TEST_ROWS
            -
            processed
        )

        if remaining <= 0:
            break

        if len(chunk) > remaining:

            chunk = chunk.iloc[
                :remaining
            ]


    results = []


    # --------------------------------------------------------
    # PROCESS RECORDS
    # --------------------------------------------------------

    for row in chunk.itertuples(
        index=False
    ):

        result = process_record(
            row
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # WRITE RESULTS
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results,
        columns=[
            "source1_entity_id",
            "matched_entity_id",
            "match_probability",
            "candidate_count"
        ]
    )


    result_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="a",
        header=not header_written
    )


    header_written = True


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    processed += len(
        result_df
    )


    matched += int(
        (
            result_df[
                "matched_entity_id"
            ]
            .astype(str)
            .str.len()
            > 0
        ).sum()
    )


    total_candidates += int(
        result_df[
            "candidate_count"
        ].sum()
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    rate = (
        processed / elapsed
        if elapsed > 0
        else 0
    )


    if rate > 0:

        if TEST_MODE:

            remaining_records = (
                TEST_ROWS
                -
                processed
            )

        else:

            remaining_records = (
                2206821
                -
                processed
            )


        eta_seconds = (
            remaining_records
            /
            rate
        )

        eta_minutes = (
            eta_seconds
            /
            60
        )

    else:

        eta_minutes = 0


    print(
        f"Processed: "
        f"{processed:,}"
        +
        (
            f"/{TEST_ROWS:,}"
            if TEST_MODE
            else ""
        )
        +
        f" | Matched: {matched:,}"
        f" | Avg candidates: "
        f"{total_candidates / processed:.1f}"
        f" | Speed: "
        f"{rate:.1f}/sec"
        f" | ETA: "
        f"{eta_minutes:.1f} min"
    )


    # --------------------------------------------------------
    # TEST COMPLETE
    # --------------------------------------------------------

    if (
        TEST_MODE
        and
        processed >= TEST_ROWS
    ):

        break


# ============================================================
# FINAL
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(
    f"Processed: {processed:,}"
)

print(
    f"Matched: {matched:,}"
)

print(
    f"Average candidates: "
    f"{total_candidates / max(processed, 1):.1f}"
)

print(
    f"Runtime: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print("=" * 70)