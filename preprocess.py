"""
preprocess.py — Spaceship Titanic Feature Engineering Pipeline

Produces (all written to output/):
  - output/train_processed.csv  (8693 rows x 39 cols, includes Transported)
  - output/test_processed.csv   (4277 rows x 38 cols, no Transported)

Run: python3 preprocess.py
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = 'output'

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
SPENDING_COLS     = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
NUMERICAL_COLS    = ['Age'] + SPENDING_COLS + ['Cabin_Num', 'Group_Size']
OHE_COLS          = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side', 'Age_Group']
BOOL_COLS         = ['CryoSleep', 'VIP']
DERIVED_BOOL_COLS = ['Is_Alone', 'Has_Spent', 'Is_Child']
# Columns that exist before Phase 7 and may have nulls requiring imputation.
# Age_Group is derived in Phase 7 and is never null, so it's excluded here.
CATEGORICAL_IMPUTE = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side'] + BOOL_COLS
COLS_TO_DROP      = ['Cabin', 'Name']


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — Load
# ─────────────────────────────────────────────────────────────────
def load_data():
    """Load raw CSV files. Returns (train_df, test_df)."""
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    print(f"[1/8] Loaded: train {train.shape}, test {test.shape}")
    return train, test


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — Feature Extraction
# ─────────────────────────────────────────────────────────────────
def extract_features(df):
    """
    Pure string parsing — no statistics required.
    - Cabin  'B/0/P' → Deck='B', Num=0.0, Side='P'
    - PassengerId '0003_02' → Group='0003'
    """
    # Cabin: 'DECK/NUM/SIDE'
    cabin_parts = df['Cabin'].str.split('/', expand=True)
    df['Cabin_Deck'] = cabin_parts[0]                                      # str
    df['Cabin_Num']  = pd.to_numeric(cabin_parts[1], errors='coerce')      # float, NaN-safe; cast to int after imputation
    df['Cabin_Side'] = cabin_parts[2]                                      # str

    # Group: first segment of PassengerId
    df['Group_ID']   = df['PassengerId'].str.split('_').str[0]
    df['Group_Size'] = df.groupby('Group_ID')['Group_ID'].transform('count').astype(int)
    df['Is_Alone']   = (df['Group_Size'] == 1)

    return df


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — Conditional Imputation (CryoSleep → spending = 0)
# ─────────────────────────────────────────────────────────────────
def conditional_impute_cryo(df):
    """
    Passengers in cryosleep cannot spend money.
    Fill NaN spending to 0 where CryoSleep is confirmed True.
    NaN == True is False in pandas, so NaN CryoSleep rows are untouched.
    """
    cryo_mask = df['CryoSleep'] == True
    df.loc[cryo_mask, SPENDING_COLS] = df.loc[cryo_mask, SPENDING_COLS].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────
# PHASE 4 — Group-based Categorical Imputation
# ─────────────────────────────────────────────────────────────────
def group_impute_categorical(df):
    """
    Fill missing HomePlanet and Destination using the mode within each Group.
    Uses .transform() (not .apply()) to stay compatible with pandas >= 2.0.
    Groups with no known value pass through to global imputation.
    """
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('Group_ID')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if x.notna().any() else x
        )
    return df


# ─────────────────────────────────────────────────────────────────
# PHASE 5 — Compute Train-Only Statistics
# ─────────────────────────────────────────────────────────────────
def compute_train_stats(train):
    """
    All medians and modes are derived exclusively from the training set.
    Called AFTER phases 3+4 so the stats reflect the partially-cleaned
    distribution (cryo zeros and group fills already applied).
    """
    stats = {
        'medians': train[NUMERICAL_COLS].median(),
        'modes': {
            col: train[col].mode().iloc[0]
            for col in CATEGORICAL_IMPUTE
            if train[col].notna().any()
        }
    }
    print(f"[5/8] Train-only medians: {stats['medians'].to_dict()}")
    print(f"      Train-only modes:   {stats['modes']}")
    return stats


# ─────────────────────────────────────────────────────────────────
# PHASE 6 — Global Imputation (train stats applied to both datasets)
# ─────────────────────────────────────────────────────────────────
def global_impute(df, stats):
    """Fill remaining nulls using training-set medians / modes."""
    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(stats['medians'][col])

    for col in CATEGORICAL_IMPUTE:
        if col in stats['modes']:
            df[col] = df[col].fillna(stats['modes'][col])

    return df


# ─────────────────────────────────────────────────────────────────
# PHASE 7 — Derived Feature Engineering
# ─────────────────────────────────────────────────────────────────
def engineer_features(df):
    """
    Compute derived features that require fully-imputed inputs.
    Called AFTER global_impute so Age and spending columns are null-free.
    - Total_Spending / Has_Spent: expenditure aggregation
    - Is_Child / Age_Group: demographic binning
    - Cabin_Num cast to int (was float to survive NaN during imputation)
    """
    # Expenditure aggregation
    df['Total_Spending'] = df[SPENDING_COLS].sum(axis=1)
    df['Has_Spent']      = (df['Total_Spending'] > 0)

    # Demographic binning
    df['Is_Child']  = (df['Age'] <= 5)
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 5, 12, 18, 35, 55, 120],
        labels=['0-5', '6-12', '13-18', '19-35', '36-55', '55+'],
        include_lowest=True
    ).astype(str)   # str required for get_dummies

    # Cabin_Num: safe to cast to int now that nulls are filled
    df['Cabin_Num'] = df['Cabin_Num'].astype(int)

    return df


def analyze_new_features(train):
    """
    Print EDA insights for the two key new features.
    Called on train only (has Transported label).
    """
    transported_int = train['Transported'].astype(int)

    corr = train['Total_Spending'].corr(transported_int)
    print(f"\n[ANALYSIS] Corr(Total_Spending, Transported): {corr:.4f}")

    side_table = (
        train.groupby('Cabin_Side')['Transported']
        .value_counts(normalize=True)
        .mul(100).round(2)
        .rename('Rate (%)')
        .reset_index()
    )
    print("\n[ANALYSIS] Transported rate by Cabin_Side:")
    print(side_table.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# PHASE 8 — Encoding
# ─────────────────────────────────────────────────────────────────
def encode_features(train, test):
    """
    1. Cast CryoSleep and VIP to 0/1 int (via bool to handle object dtype).
    2. One-hot encode OHE_COLS using concat-then-split strategy so that
       train and test are guaranteed to share an identical column schema.
    """
    # Boolean → int  (chain .astype(bool) first; direct .astype(int) on object+NaN fails)
    for df in [train, test]:
        for col in BOOL_COLS + DERIVED_BOOL_COLS:
            df[col] = df[col].astype(bool).astype(int)

    # Concat for OHE — Transported column is NaN for all test rows
    combined = pd.concat([train, test], sort=False, ignore_index=True)

    combined = pd.get_dummies(
        combined,
        columns=OHE_COLS,
        drop_first=False,   # Keep all categories (best for tree models)
        dtype=int
    )

    # Split back: train rows have a Transported value; test rows have NaN
    train_enc = combined[combined['Transported'].notna()].copy()
    test_enc  = combined[combined['Transported'].isna()].copy()

    return train_enc, test_enc


# ─────────────────────────────────────────────────────────────────
# PHASE 8 — Export & Validate
# ─────────────────────────────────────────────────────────────────
def export_and_validate(train_enc, test_enc):
    """
    Drop raw source columns, reset indices, save CSVs, and assert quality.
    """
    train_out = train_enc.drop(columns=COLS_TO_DROP, errors='ignore')
    test_out  = test_enc.drop(columns=COLS_TO_DROP + ['Transported'], errors='ignore')

    # Reset index — concat creates non-contiguous indices after boolean split
    train_out = train_out.reset_index(drop=True)
    test_out  = test_out.reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, 'train_processed.csv')
    test_path  = os.path.join(OUTPUT_DIR, 'test_processed.csv')
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    # ── Validation ──────────────────────────────────────────────
    print("\n=== VALIDATION ===")
    print(f"{train_path}: {train_out.shape}")
    print(f"{test_path}:  {test_out.shape}")

    train_nulls = train_out.isnull().sum().sum()
    test_nulls  = test_out.isnull().sum().sum()
    print(f"train null count: {train_nulls}")
    print(f"test  null count: {test_nulls}")

    assert train_nulls == 0, f"FAIL: train has {train_nulls} nulls"
    assert test_nulls  == 0, f"FAIL: test has {test_nulls} nulls"

    # Confirm ordered column lists match (train features == all test columns)
    train_feature_cols = [c for c in train_out.columns if c != 'Transported']
    test_cols = list(test_out.columns)
    assert train_feature_cols == test_cols, (
        f"FAIL: Column mismatch\n  train features: {train_feature_cols}\n  test: {test_cols}"
    )

    print("All validation checks passed.")
    return train_out, test_out


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SPACESHIP TITANIC — PREPROCESSING PIPELINE")
    print("=" * 60)

    # Phase 1: Load
    train, test = load_data()

    # Phase 2: Feature extraction (pure parsing)
    print("[2/8] Extracting features from Cabin and PassengerId...")
    train = extract_features(train)
    test  = extract_features(test)

    # Phase 3: CryoSleep-aware spending imputation (domain rule)
    print("[3/8] Applying CryoSleep conditional imputation...")
    train = conditional_impute_cryo(train)
    test  = conditional_impute_cryo(test)

    # Phase 4: Group-based categorical imputation
    print("[4/8] Applying group-based imputation for HomePlanet and Destination...")
    train = group_impute_categorical(train)
    test  = group_impute_categorical(test)

    # Phase 5: Compute train-only stats (AFTER phases 3+4 for cleaner distribution)
    stats = compute_train_stats(train)

    # Phase 6: Global imputation using train stats only
    print("[6/8] Applying global imputation...")
    train = global_impute(train, stats)
    test  = global_impute(test, stats)

    # Phase 7: Derived feature engineering (requires null-free inputs)
    print("[7/8] Engineering derived features...")
    train = engineer_features(train)
    test  = engineer_features(test)
    analyze_new_features(train)

    # Phase 8: Encoding (bool cast + OHE via concat-then-split)
    print("[8/8] Encoding features...")
    train, test = encode_features(train, test)

    # Phase 9: Export + validate
    export_and_validate(train, test)


if __name__ == '__main__':
    main()
