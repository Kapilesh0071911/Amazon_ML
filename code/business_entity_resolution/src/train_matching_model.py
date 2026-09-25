import os
import random

import joblib
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/processed"
TRAIN_DIR = "dataset/train"
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

TRUTH_FILE = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "entity_resolution_model.joblib"
)

# Number of Source-1 records used for training
S1_SAMPLE_SIZE = 5000

# Number of random negative examples per positive
NEGATIVES_PER_POSITIVE = 2

RANDOM_SEED = 42


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# SAFE STRING
# ============================================================

def clean(value):

    if pd.isna(value):
        return ""

    return str(value)


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(
    s1_name,
    s1_address,
    s1_country,
    candidate_name,
    candidate_address,
    candidate_country
):

    s1_name = clean(s1_name)
    s1_address = clean(s1_address)
    s1_country = clean(s1_country)

    candidate_name = clean(candidate_name)
    candidate_address = clean(candidate_address)
    candidate_country = clean(candidate_country)

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_ratio = ratio(
        s1_name,
        candidate_name
    ) / 100.0

    name_token_ratio = token_set_ratio(
        s1_name,
        candidate_name
    ) / 100.0

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    address_ratio = ratio(
        s1_address,
        candidate_address
    ) / 100.0

    address_token_ratio = token_set_ratio(
        s1_address,
        candidate_address
    ) / 100.0

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = int(
        s1_country.strip().lower()
        ==
        candidate_country.strip().lower()
        and s1_country.strip() != ""
    )

    # --------------------------------------------------------
    # EXACT MATCHES
    # --------------------------------------------------------

    exact_name = int(
        s1_name.strip().lower()
        ==
        candidate_name.strip().lower()
        and s1_name.strip() != ""
    )

    exact_address = int(
        s1_address.strip().lower()
        ==
        candidate_address.strip().lower()
        and s1_address.strip() != ""
    )

    # --------------------------------------------------------
    # LENGTH DIFFERENCES
    # --------------------------------------------------------

    name_length_difference = abs(
        len(s1_name)
        -
        len(candidate_name)
    )

    address_length_difference = abs(
        len(s1_address)
        -
        len(candidate_address)
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
# LOAD SOURCE 1
# ============================================================

print("=" * 60)
print("LOADING SOURCE 1")
print("=" * 60)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

s1 = s1.fillna("")

print(
    f"Source 1 rows: {len(s1):,}"
)


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

if len(s1) > S1_SAMPLE_SIZE:

    s1_sample = s1.sample(
        n=S1_SAMPLE_SIZE,
        random_state=RANDOM_SEED
    ).copy()

else:

    s1_sample = s1.copy()


print(
    f"Training S1 sample: {len(s1_sample):,}"
)


# ============================================================
# CREATE FAST SOURCE 1 LOOKUP
# ============================================================

print(
    "\nBuilding Source 1 lookup..."
)

s1_lookup = {}

for row in s1_sample.itertuples(
    index=False
):

    s1_lookup[
        row.entity_id
    ] = {
        "name": row.name_normalized,
        "address": row.address_normalized,
        "country": row.country
    }


print(
    f"S1 lookup entries: {len(s1_lookup):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LOADING GROUND TRUTH"
)

print(
    "=" * 60
)

truth = pd.read_csv(
    TRUTH_FILE,
    sep="\t",
    dtype=str
)

truth = truth.fillna("")


# ============================================================
# CREATE TRUTH LOOKUP
# ============================================================

truth_lookup = dict(
    zip(
        truth["source1_entity_id"],
        truth["matched_entity_ids"]
    )
)


# ============================================================
# CREATE POSITIVE PAIRS
# ============================================================

print(
    "\nCreating positive pairs..."
)

positive_pairs = []

required_candidate_ids = set()


for s1_id in s1_lookup:

    matched = truth_lookup.get(
        s1_id,
        ""
    )

    matched = clean(
        matched
    ).strip()

    if not matched:
        continue

    match_ids = [
        x.strip()
        for x in matched.split(",")
        if x.strip()
    ]

    for candidate_id in match_ids:

        positive_pairs.append(
            (
                s1_id,
                candidate_id
            )
        )

        required_candidate_ids.add(
            candidate_id
        )


print(
    f"Positive pairs: {len(positive_pairs):,}"
)

print(
    f"Required candidate IDs: "
    f"{len(required_candidate_ids):,}"
)


# ============================================================
# LOAD SOURCE 2
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LOADING SOURCE 2"
)

print(
    "=" * 60
)


s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized",
        "address_normalized",
        "country"
    ]
)

s2 = s2.fillna("")


# Only keep IDs required by the ground truth
s2 = s2[
    s2["entity_id"].isin(
        required_candidate_ids
    )
]


print(
    f"Required S2 records loaded: "
    f"{len(s2):,}"
)


# ============================================================
# LOAD SOURCE 3
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LOADING SOURCE 3"
)

print(
    "=" * 60
)


s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "name_normalized",
        "address_normalized",
        "country"
    ]
)

s3 = s3.fillna("")


s3 = s3[
    s3["entity_id"].isin(
        required_candidate_ids
    )
]


print(
    f"Required S3 records loaded: "
    f"{len(s3):,}"
)


# ============================================================
# BUILD FAST CANDIDATE LOOKUP
# ============================================================

print(
    "\nBuilding candidate lookup..."
)


candidate_lookup = {}


for row in s2.itertuples(
    index=False
):

    candidate_lookup[
        row.entity_id
    ] = {
        "name": row.name_normalized,
        "address": row.address_normalized,
        "country": row.country
    }


for row in s3.itertuples(
    index=False
):

    candidate_lookup[
        row.entity_id
    ] = {
        "name": row.name_normalized,
        "address": row.address_normalized,
        "country": row.country
    }


print(
    f"Candidate lookup entries: "
    f"{len(candidate_lookup):,}"
)


# ============================================================
# BUILD POSITIVE TRAINING FEATURES
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "BUILDING POSITIVE TRAINING FEATURES"
)

print(
    "=" * 60
)


training_rows = []

missing_positive = 0


for index, (
    s1_id,
    candidate_id
) in enumerate(
    positive_pairs,
    start=1
):

    s1_record = s1_lookup.get(
        s1_id
    )

    candidate_record = candidate_lookup.get(
        candidate_id
    )

    if (
        s1_record is None
        or candidate_record is None
    ):

        missing_positive += 1
        continue


    features = calculate_features(
        s1_record["name"],
        s1_record["address"],
        s1_record["country"],

        candidate_record["name"],
        candidate_record["address"],
        candidate_record["country"]
    )


    training_rows.append(
        features + [1]
    )


    if index % 2000 == 0:

        print(
            f"Processed positive pairs: "
            f"{index:,} / "
            f"{len(positive_pairs):,}"
        )


print(
    f"Valid positive rows: "
    f"{len(training_rows):,}"
)

print(
    f"Missing positive pairs: "
    f"{missing_positive:,}"
)


# ============================================================
# NEGATIVE PAIRS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "BUILDING NEGATIVE TRAINING FEATURES"
)

print(
    "=" * 60
)


random.seed(
    RANDOM_SEED
)


candidate_ids = list(
    candidate_lookup.keys()
)


positive_set = set(
    positive_pairs
)


negative_count = 0


for index, (
    s1_id,
    positive_candidate_id
) in enumerate(
    positive_pairs,
    start=1
):

    s1_record = s1_lookup.get(
        s1_id
    )

    if s1_record is None:
        continue


    added = 0

    attempts = 0

    max_attempts = (
        NEGATIVES_PER_POSITIVE * 20
    )


    while (
        added < NEGATIVES_PER_POSITIVE
        and attempts < max_attempts
    ):

        attempts += 1


        random_candidate_id = random.choice(
            candidate_ids
        )


        # Never accidentally create a positive pair
        if (
            s1_id,
            random_candidate_id
        ) in positive_set:

            continue


        candidate_record = candidate_lookup.get(
            random_candidate_id
        )

        if candidate_record is None:
            continue


        features = calculate_features(
            s1_record["name"],
            s1_record["address"],
            s1_record["country"],

            candidate_record["name"],
            candidate_record["address"],
            candidate_record["country"]
        )


        training_rows.append(
            features + [0]
        )


        negative_count += 1
        added += 1


    if index % 2000 == 0:

        print(
            f"Processed negative examples for "
            f"{index:,} positive pairs"
        )


print(
    f"Negative rows created: "
    f"{negative_count:,}"
)


# ============================================================
# BUILD TRAINING DATAFRAME
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "BUILDING TRAINING DATASET"
)

print(
    "=" * 60
)


dataset = pd.DataFrame(
    training_rows,
    columns=FEATURE_COLUMNS + ["label"]
)


print(
    f"Training dataset shape: "
    f"{dataset.shape}"
)


print(
    "\nClass distribution:"
)

print(
    dataset["label"].value_counts()
)


# ============================================================
# VALIDATE DATASET
# ============================================================

if len(dataset) < 100:

    raise RuntimeError(
        "Training dataset is too small."
    )


if dataset["label"].nunique() < 2:

    raise RuntimeError(
        "Training dataset does not contain "
        "both positive and negative examples."
    )


# ============================================================
# SPLIT DATA
# ============================================================

print(
    "\nSplitting train/validation data..."
)


X = dataset[
    FEATURE_COLUMNS
]

y = dataset[
    "label"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)


print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Validation rows: {len(X_test):,}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING RANDOM FOREST"
)

print(
    "=" * 60
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


print(
    "Model training complete."
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MATCHING MODEL EVALUATION"
)

print(
    "=" * 60
)


print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


try:

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

except Exception as error:

    print(
        "ROC-AUC could not be calculated:",
        error
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "=" * 60
)


importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


model_package = {
    "model": model,
    "features": FEATURE_COLUMNS,
    "random_seed": RANDOM_SEED
}


joblib.dump(
    model_package,
    MODEL_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL TRAINING COMPLETE"
)

print(
    "=" * 60
)

print(
    f"Model saved: {MODEL_FILE}"
)

print(
    f"Training examples: {len(X_train):,}"
)

print(
    f"Validation examples: {len(X_test):,}"
)

print(
    "=" * 60
)