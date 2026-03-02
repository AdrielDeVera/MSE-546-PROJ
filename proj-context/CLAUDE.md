# Spaceship Titanic — Project Context

## Course Context (MSCI 546 — Advanced Machine Learning)
- **Instructor**: Prof. Sirisha Rambhatla
- **Team**: Group 6 (4 members), project worth 30% of final grade
- **Stage P2** (Proposal + Baseline): ✅ Complete — 78.84% validation accuracy
- **Stage P3** (Development): In progress — each member implements one distinct method; also requires one neural network method (MLP or TabNet); 6 methods total minimum
- **Stage P4/P5** (Report + Presentation): 20-slide PDF + 8-min in-class presentation with 4+ baselines shown

## What This Is
Kaggle binary classification competition. Goal: predict whether passengers were `Transported` (True/False) to an alternate dimension.

## How to Run
```
python3 run.py
```

## Code Structure
Single script `run.py` with 7 phases:
1. Load data + EDA visualizations (6 plots saved as PNGs)
2. Feature selection & preprocessing pipeline
3. Train/validation split (80/20)
4. Train Random Forest
5. Evaluate (accuracy, confusion matrix, classification report)
6. Feature importance analysis
7. Generate Kaggle submission CSV

## Model
- **Algorithm**: Random Forest (100 trees, max_depth=10, random_state=42)
- **Validation accuracy**: ~78.84%
- **Preprocessing**: scikit-learn Pipeline + ColumnTransformer (median impute numericals, most-frequent + OHE categoricals)

## Features
- **Numerical** (6): `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- **Categorical** (4): `HomePlanet`, `CryoSleep`, `Destination`, `VIP`
- **Excluded**: `Name` (text), `Cabin` (complex string — future work)

## Data Files
| File | Description |
|------|-------------|
| `train.csv` | 8,693 rows with target `Transported` |
| `test.csv` | 4,277 rows, no target |
| `sample_submission.csv` | Kaggle format template |
| `submission_baseline_rf.csv` | Generated predictions for submission |

## Dependencies
See `requirements.txt` — pandas, numpy, seaborn, matplotlib, scikit-learn.

## Key EDA Insights (from Proposal)
- **CryoSleep**: Dominant predictor — passengers in suspended animation transported at >75% rate
- **Age 0–5**: Non-linear spike in transport probability for toddlers vs. adults
- **Cabin Side**: Engineered feature — Starboard (S) passengers transported more than Port (P)
- **Spending**: Active spending (Spa, VRDeck, FoodCourt) negatively correlated with transport

## Next Improvement Ideas (P3 — one method per team member + 1 NN required)
- Parse `Cabin` into deck letter, number, and side (P/S)
- Engineer family group features from `PassengerId`
- Create total spending feature
- Hyperparameter tuning (GridSearchCV)
- Try gradient boosting (XGBoost, LightGBM)
- **Neural network** (MLP or TabNet with entity embeddings) — required by course spec
