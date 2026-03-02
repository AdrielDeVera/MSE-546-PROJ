# Spaceship Titanic

Binary classification competition (Kaggle). Predict whether each passenger was `Transported` to an alternate dimension following a spacetime anomaly. Part of MSCI 546 — Advanced Machine Learning, Group 6.

**Baseline validation accuracy: 78.84%** (Random Forest)

---

## Project Structure

```
spaceship-titanic/
├── train.csv                    # Raw training data (8,693 rows)
├── test.csv                     # Raw test data (4,277 rows, no target)
├── sample_submission.csv        # Kaggle submission format template
├── run.py                       # Baseline model — EDA + Random Forest + submission
├── preprocess.py                # Advanced feature engineering pipeline
├── validate.py                  # Validates preprocessed output against raw data
├── requirements.txt             # Python dependencies
└── output/                      # Created by preprocess.py
    ├── train_processed.csv      # Fully engineered training features (39 cols)
    ├── test_processed.csv       # Fully engineered test features (38 cols)
    └── report.txt               # Validation report from validate.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Scripts

### `run.py` — Baseline Model

The self-contained baseline. Loads raw CSVs, generates EDA plots, trains a Random Forest, and writes a Kaggle submission file. No other scripts need to run first.

```bash
python3 run.py
```

**What it does:**
1. Loads `train.csv` and `test.csv`
2. Generates 4 EDA visualizations (CryoSleep impact, age distribution, cabin side, spending correlations)
3. Builds a scikit-learn `Pipeline`: median imputation for numericals, mode + one-hot encoding for categoricals
4. Trains `RandomForestClassifier(n_estimators=100, max_depth=10)` on an 80/20 train/val split
5. Evaluates on the validation set, saves a confusion matrix and feature importance plot
6. Writes `submission_baseline_rf.csv` (predictions on `test.csv`)

**Output files:**
- `plot_1_cryosleep_impact.png` through `plot_6_feature_importance.png`
- `submission_baseline_rf.csv`

---

### `preprocess.py` — Feature Engineering Pipeline

Produces fully-engineered, model-ready CSVs in `output/`. Run this before training any advanced model that needs enriched features.

```bash
python3 preprocess.py
```

**What it does (8 phases):**

| Phase | Description |
|-------|-------------|
| 1 | Load raw CSVs |
| 2 | Parse `Cabin` → `Cabin_Deck`, `Cabin_Num`, `Cabin_Side`; extract `Group_ID`, `Group_Size`, `Is_Alone` from `PassengerId` |
| 3 | Domain-aware imputation: CryoSleep passengers cannot spend, so their spending nulls are filled with 0 |
| 4 | Group-based imputation: fill missing `HomePlanet`/`Destination` using the mode within each travel group |
| 5 | Compute train-only medians and modes (no data leakage) |
| 6 | Apply those statistics to fill remaining nulls in both train and test |
| 7 | Engineer derived features: `Total_Spending`, `Has_Spent`, `Is_Child`, `Age_Group` bins |
| 8 | Encode: cast booleans to 0/1, one-hot encode categoricals using concat-then-split to guarantee identical schemas |

**Output files:**
- `output/train_processed.csv` — 8,693 rows × 39 columns (includes `Transported`)
- `output/test_processed.csv` — 4,277 rows × 38 columns (no `Transported`)

---

### `validate.py` — Preprocessing Validation

Verifies that `output/train_processed.csv` and `output/test_processed.csv` are correct. Run this after `preprocess.py` to confirm the pipeline worked as expected.

```bash
python3 validate.py
```

**Checks performed:**

| # | Check | What it verifies |
|---|-------|-----------------|
| 1 | Shape & Schema | Expected dimensions, zero nulls, aligned column order |
| 2 | Row Identity | Every `PassengerId` from the raw data is present (no rows added or dropped) |
| 3 | Target Preservation | `Transported` values match the raw training labels exactly |
| 4 | Cabin Parsing | Reconstructed `Deck`, `Num`, and `Side` match the raw `Cabin` strings |
| 5 | Group Extraction | `Group_ID` values correctly derived from `PassengerId` |
| 6 | CryoSleep Invariant | All CryoSleep=1 rows have spending = 0 |
| 7 | OHE Mutual Exclusivity | Each one-hot group sums to exactly 1 per row |
| 8 | Numerical Range Sanity | Age in [0, 120], spending ≥ 0, boolean columns are 0 or 1 only |

A human-readable report is written to `output/report.txt`. The script exits with code 1 if any check fails.

---

## Data Files

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 8,693 | 14 | Raw features + `Transported` target |
| `test.csv` | 4,277 | 13 | Raw features, no target |
| `sample_submission.csv` | 4,277 | 2 | `PassengerId`, `Transported` (placeholder values) |
| `submission_baseline_rf.csv` | 4,277 | 2 | Predictions from `run.py`, ready to upload to Kaggle |
| `output/train_processed.csv` | 8,693 | 39 | Fully engineered; produced by `preprocess.py` |
| `output/test_processed.csv` | 4,277 | 38 | Fully engineered; produced by `preprocess.py` |

Raw features: `PassengerId`, `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `Age`, `VIP`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `Name`, `Transported` (train only).

---

## Key EDA Findings

- **CryoSleep** is the strongest predictor — passengers in suspended animation are transported at >75% rate
- **Age 0–5** shows a non-linear spike in transport probability compared to adults
- **Cabin Side**: Starboard (S) passengers are transported more than Port (P)
- **Spending**: Active spending (Spa, VRDeck, FoodCourt) is negatively correlated with transport
