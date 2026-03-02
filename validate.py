"""
validate.py — Cross-validates output/train_processed.csv and output/test_processed.csv
              against the original raw CSVs.

Each check returns (passed: bool, detail: str).
Writes a human-readable summary to output/report.txt.
Run: python3 validate.py
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np

OUTPUT_DIR = 'output'

SPENDING_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
OHE_GROUPS = {
    'HomePlanet':  ['HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars'],
    'Destination': ['Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e'],
    'Cabin_Deck':  ['Cabin_Deck_A', 'Cabin_Deck_B', 'Cabin_Deck_C', 'Cabin_Deck_D',
                    'Cabin_Deck_E', 'Cabin_Deck_F', 'Cabin_Deck_G', 'Cabin_Deck_T'],
    'Cabin_Side':  ['Cabin_Side_P', 'Cabin_Side_S'],
    'Age_Group':   ['Age_Group_0-5', 'Age_Group_6-12', 'Age_Group_13-18',
                    'Age_Group_19-35', 'Age_Group_36-55', 'Age_Group_55+'],
}


# ─────────────────────────────────────────────────────────────────
# Check 1 — Shape & Schema
# ─────────────────────────────────────────────────────────────────
def check_shape_and_nulls(train_proc, test_proc):
    failures = []

    if train_proc.shape != (8693, 39):
        failures.append(f"train shape is {train_proc.shape}, expected (8693, 39)")
    if test_proc.shape != (4277, 38):
        failures.append(f"test shape is {test_proc.shape}, expected (4277, 38)")

    train_nulls = train_proc.isnull().sum().sum()
    test_nulls  = test_proc.isnull().sum().sum()
    if train_nulls != 0:
        failures.append(f"train has {train_nulls} nulls")
    if test_nulls != 0:
        failures.append(f"test has {test_nulls} nulls")

    # Column order: all test columns must appear in train (in same order), train adds Transported
    train_feature_cols = [c for c in train_proc.columns if c != 'Transported']
    test_cols = list(test_proc.columns)
    if train_feature_cols != test_cols:
        failures.append(f"Column order mismatch between train features and test")

    if failures:
        return False, "; ".join(failures)
    return True, f"{train_proc.shape[0]}×{train_proc.shape[1]}, {test_proc.shape[0]}×{test_proc.shape[1]}, 0 nulls, schema aligned"


# ─────────────────────────────────────────────────────────────────
# Check 2 — Row Identity Preservation
# ─────────────────────────────────────────────────────────────────
def check_row_identity(raw_train, train_proc, raw_test, test_proc):
    failures = []

    raw_train_ids  = set(raw_train['PassengerId'])
    proc_train_ids = set(train_proc['PassengerId'])
    missing = raw_train_ids - proc_train_ids
    extra   = proc_train_ids - raw_train_ids
    if missing or extra:
        failures.append(f"train: {len(missing)} IDs missing, {len(extra)} extra IDs")

    raw_test_ids  = set(raw_test['PassengerId'])
    proc_test_ids = set(test_proc['PassengerId'])
    missing = raw_test_ids - proc_test_ids
    extra   = proc_test_ids - raw_test_ids
    if missing or extra:
        failures.append(f"test: {len(missing)} IDs missing, {len(extra)} extra IDs")

    if failures:
        return False, "; ".join(failures)
    return True, f"{len(raw_train_ids)}/{len(raw_train_ids)} train IDs matched, {len(raw_test_ids)}/{len(raw_test_ids)} test IDs matched"


# ─────────────────────────────────────────────────────────────────
# Check 3 — Target Preservation
# ─────────────────────────────────────────────────────────────────
def check_target_preservation(raw_train, train_proc):
    # Align on PassengerId, then compare Transported
    merged = raw_train[['PassengerId', 'Transported']].merge(
        train_proc[['PassengerId', 'Transported']],
        on='PassengerId',
        suffixes=('_raw', '_proc')
    )
    # Normalise both to bool for comparison
    raw_bool  = merged['Transported_raw'].astype(bool)
    proc_bool = merged['Transported_proc'].astype(bool)
    mismatches = (raw_bool != proc_bool).sum()
    if mismatches > 0:
        return False, f"{mismatches} Transported values differ from raw"
    return True, f"{len(merged)}/{len(raw_train)} Transported values match"


# ─────────────────────────────────────────────────────────────────
# Check 4 — Cabin Parsing
# ─────────────────────────────────────────────────────────────────
def check_cabin_parsing(raw_train, train_proc, raw_test, test_proc):
    """
    Deck and Side are OHE-expanded in the processed files (Deck_A, Side_P, etc.).
    Reconstruct them via idxmax on their dummy columns, then compare to raw Cabin.
    Num is still a numerical column and can be compared directly.
    Only checks rows where raw Cabin was not null (nulls were imputed, so they can't match).
    """
    failures = []
    total_checked = 0

    deck_cols = sorted([c for c in train_proc.columns if c.startswith('Cabin_Deck_')])
    side_cols = sorted([c for c in train_proc.columns if c.startswith('Cabin_Side_')])

    for label, raw_df, proc_df in [('train', raw_train, train_proc), ('test', raw_test, test_proc)]:
        d_cols = [c for c in deck_cols if c in proc_df.columns]
        s_cols = [c for c in side_cols if c in proc_df.columns]

        merged = raw_df[['PassengerId', 'Cabin']].merge(
            proc_df[['PassengerId', 'Cabin_Num'] + d_cols + s_cols],
            on='PassengerId'
        )
        non_null = merged[merged['Cabin'].notna()].copy()
        if non_null.empty:
            continue

        parts = non_null['Cabin'].str.split('/', expand=True)
        expected_deck = parts[0]
        expected_num  = pd.to_numeric(parts[1], errors='coerce')
        expected_side = parts[2]

        # Cabin_Num: direct comparison
        num_bad = (~np.isclose(expected_num.values.astype(float),
                               non_null['Cabin_Num'].values.astype(float),
                               equal_nan=True)).sum()

        # Cabin_Deck: reconstruct from OHE ('Cabin_Deck_F' → strip prefix → 'F')
        reconstructed_deck = (
            non_null[d_cols].idxmax(axis=1).str.replace('Cabin_Deck_', '', regex=False)
        )
        deck_bad = (expected_deck.values != reconstructed_deck.values).sum()

        # Cabin_Side: reconstruct from OHE ('Cabin_Side_P' → 'P')
        reconstructed_side = (
            non_null[s_cols].idxmax(axis=1).str.replace('Cabin_Side_', '', regex=False)
        )
        side_bad = (expected_side.values != reconstructed_side.values).sum()

        if deck_bad or num_bad or side_bad:
            failures.append(f"{label}: {deck_bad} Deck, {num_bad} Num, {side_bad} Side mismatches")
        else:
            total_checked += len(non_null)

    if failures:
        return False, "; ".join(failures)
    return True, f"{total_checked} non-null Cabin rows verified (Deck, Num, Side all match)"


# ─────────────────────────────────────────────────────────────────
# Check 5 — Group Extraction
# ─────────────────────────────────────────────────────────────────
def check_group_extraction(raw_train, train_proc, raw_test, test_proc):
    failures = []
    total = 0

    for label, raw_df, proc_df in [('train', raw_train, train_proc), ('test', raw_test, test_proc)]:
        merged = raw_df[['PassengerId']].merge(
            proc_df[['PassengerId', 'Group_ID']],
            on='PassengerId'
        )
        # Group_ID is a zero-padded string ("0001") in the pipeline, but CSV
        # round-trips strip leading zeros to int. Compare numerically.
        expected_group = merged['PassengerId'].str.split('_').str[0].astype(int)
        actual_group   = merged['Group_ID'].astype(int)
        bad = (expected_group.values != actual_group.values).sum()
        if bad:
            failures.append(f"{label}: {bad} Group values wrong")
        total += len(merged)

    if failures:
        return False, "; ".join(failures)
    return True, f"all {total} PassengerId → Group_ID extractions correct"


# ─────────────────────────────────────────────────────────────────
# Check 6 — CryoSleep Invariant
# ─────────────────────────────────────────────────────────────────
def check_cryo_invariant(train_proc, test_proc):
    failures = []

    for label, df in [('train', train_proc), ('test', test_proc)]:
        cryo_rows = df[df['CryoSleep'] == 1]
        bad = (cryo_rows[SPENDING_COLS] != 0).any(axis=1).sum()
        if bad:
            failures.append(f"{label}: {bad} CryoSleep=1 rows have non-zero spending")

    if failures:
        return False, "; ".join(failures)
    n_cryo_train = (train_proc['CryoSleep'] == 1).sum()
    n_cryo_test  = (test_proc['CryoSleep'] == 1).sum()
    return True, f"{n_cryo_train} train + {n_cryo_test} test CryoSleep rows all have spending = 0"


# ─────────────────────────────────────────────────────────────────
# Check 7 — OHE Mutual Exclusivity
# ─────────────────────────────────────────────────────────────────
def check_ohe_mutual_exclusivity(train_proc, test_proc):
    failures = []

    for label, df in [('train', train_proc), ('test', test_proc)]:
        for group_name, cols in OHE_GROUPS.items():
            present_cols = [c for c in cols if c in df.columns]
            if not present_cols:
                failures.append(f"{label}: no columns found for OHE group '{group_name}'")
                continue
            row_sums = df[present_cols].sum(axis=1)
            bad = (row_sums != 1).sum()
            if bad:
                failures.append(f"{label}.{group_name}: {bad} rows don't sum to 1 (got min={row_sums.min()}, max={row_sums.max()})")

    if failures:
        return False, "; ".join(failures)
    return True, f"HomePlanet ✓, Destination ✓, Cabin_Deck ✓, Cabin_Side ✓, Age_Group ✓ (all row sums == 1)"


# ─────────────────────────────────────────────────────────────────
# Check 8 — Numerical Range Sanity
# ─────────────────────────────────────────────────────────────────
def check_numerical_ranges(train_proc, test_proc):
    failures = []

    for label, df in [('train', train_proc), ('test', test_proc)]:
        # Age
        bad_age = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
        if bad_age:
            failures.append(f"{label}: {bad_age} Age values outside [0, 120]")

        # Cabin_Num
        bad_num = (df['Cabin_Num'] < 0).sum()
        if bad_num:
            failures.append(f"{label}: {bad_num} Cabin_Num values < 0")

        # Spending columns
        for col in SPENDING_COLS:
            bad = (df[col] < 0).sum()
            if bad:
                failures.append(f"{label}.{col}: {bad} negative values")

        # CryoSleep and VIP should be 0 or 1 only
        for col in ['CryoSleep', 'VIP']:
            bad = (~df[col].isin([0, 1])).sum()
            if bad:
                failures.append(f"{label}.{col}: {bad} values outside {{0, 1}}")

        # All OHE columns should be 0 or 1
        all_ohe_cols = [c for group in OHE_GROUPS.values() for c in group if c in df.columns]
        for col in all_ohe_cols:
            bad = (~df[col].isin([0, 1])).sum()
            if bad:
                failures.append(f"{label}.{col}: {bad} values outside {{0, 1}}")

    if failures:
        return False, "; ".join(failures)

    age_range = f"[{train_proc['Age'].min():.0f}, {train_proc['Age'].max():.0f}]"
    return True, f"Age {age_range}, Cabin_Num ≥ 0, spending ≥ 0, binary cols {{0,1}} only"


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SPACESHIP TITANIC — PREPROCESSING VALIDATION")
    print("=" * 70)

    train_proc_path = os.path.join(OUTPUT_DIR, 'train_processed.csv')
    test_proc_path  = os.path.join(OUTPUT_DIR, 'test_processed.csv')

    raw_train  = pd.read_csv('train.csv')
    raw_test   = pd.read_csv('test.csv')
    train_proc = pd.read_csv(train_proc_path)
    test_proc  = pd.read_csv(test_proc_path)

    checks = [
        ("Shape & Schema",            check_shape_and_nulls(train_proc, test_proc)),
        ("Row Identity",              check_row_identity(raw_train, train_proc, raw_test, test_proc)),
        ("Target Preservation",       check_target_preservation(raw_train, train_proc)),
        ("Cabin Parsing",             check_cabin_parsing(raw_train, train_proc, raw_test, test_proc)),
        ("Group Extraction",          check_group_extraction(raw_train, train_proc, raw_test, test_proc)),
        ("CryoSleep Invariant",       check_cryo_invariant(train_proc, test_proc)),
        ("OHE Mutual Exclusivity",    check_ohe_mutual_exclusivity(train_proc, test_proc)),
        ("Numerical Range Sanity",    check_numerical_ranges(train_proc, test_proc)),
    ]

    print()
    passed = 0
    for i, (name, (ok, detail)) in enumerate(checks, 1):
        status = "PASS" if ok else "FAIL"
        print(f"Check {i}: {name:<30} {status}  ({detail})")
        if ok:
            passed += 1

    print()
    print(f"{passed}/{len(checks)} checks passed.")

    # ── Write human-readable report ───────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    _write_report(report_path, raw_train, raw_test, train_proc, test_proc, checks, passed)
    print(f"\nReport written to {report_path}")

    if passed < len(checks):
        raise SystemExit(1)


def _write_report(path, raw_train, raw_test, train_proc, test_proc, checks, passed):
    """Write a plain-text summary of the validation run."""
    SEP  = "=" * 70
    SEP2 = "-" * 70
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect feature-group info for the report
    feature_cols = [c for c in train_proc.columns if c not in ('Transported', 'PassengerId', 'Group_ID', 'Name', 'Cabin')]
    numeric_features = [c for c in feature_cols if train_proc[c].dtype in ['int64', 'float64']
                        and not any(c.startswith(p) for p in ('HomePlanet_', 'Destination_', 'Cabin_Deck_', 'Cabin_Side_', 'Age_Group_'))]
    ohe_features     = [c for c in feature_cols if any(c.startswith(p) for p in
                        ('HomePlanet_', 'Destination_', 'Cabin_Deck_', 'Cabin_Side_', 'Age_Group_'))]
    bool_features    = [c for c in feature_cols if c in ('CryoSleep', 'VIP', 'Is_Alone', 'Has_Spent', 'Is_Child')]

    lines = [
        SEP,
        "SPACESHIP TITANIC — PREPROCESSING VALIDATION REPORT",
        f"Generated : {now}",
        SEP,
        "",
        "DATA SUMMARY",
        SEP2,
        f"  Raw training set  : {raw_train.shape[0]:,} rows × {raw_train.shape[1]} columns",
        f"  Raw test set      : {raw_test.shape[0]:,} rows × {raw_test.shape[1]} columns",
        f"  Processed train   : {train_proc.shape[0]:,} rows × {train_proc.shape[1]} columns",
        f"  Processed test    : {test_proc.shape[0]:,} rows × {test_proc.shape[1]} columns",
        f"  Null count (train): {train_proc.isnull().sum().sum()}",
        f"  Null count (test) : {test_proc.isnull().sum().sum()}",
        "",
        "FEATURES IN PROCESSED DATA",
        SEP2,
        f"  Numerical  ({len(numeric_features)}): {', '.join(numeric_features)}",
        f"  Boolean    ({len(bool_features)}): {', '.join(bool_features)}",
        f"  OHE dummies ({len(ohe_features)}): {', '.join(ohe_features)}",
        "",
        "TARGET DISTRIBUTION (TRAIN)",
        SEP2,
    ]

    transported_counts = raw_train['Transported'].value_counts()
    for val, count in transported_counts.items():
        pct = count / len(raw_train) * 100
        lines.append(f"  Transported={val}: {count:,} ({pct:.1f}%)")

    lines += [
        "",
        "VALIDATION CHECKS",
        SEP2,
    ]

    for i, (name, (ok, detail)) in enumerate(checks, 1):
        status = "PASS" if ok else "FAIL"
        lines.append(f"  {i:>2}. {name:<30}  [{status}]")
        lines.append(f"       {detail}")

    overall = "ALL CHECKS PASSED" if passed == len(checks) else f"{len(checks) - passed} CHECK(S) FAILED"
    lines += [
        "",
        SEP,
        f"RESULT : {passed}/{len(checks)} checks passed — {overall}",
        SEP,
        "",
    ]

    with open(path, 'w') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
