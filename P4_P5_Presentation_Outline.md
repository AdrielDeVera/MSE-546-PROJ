# P4 / P5 — Slides-Style Report & In-Class Presentation Outline
## MSCI 546 — Group 6 | Spaceship Titanic (Kaggle)

> **Goal of this document:** Slide-by-slide blueprint with exact content, diagrams, speaker notes, and a checklist of what still needs to be completed to earn full marks.
>
> **Hard constraints:** ≤ 20 slides (incl. references) · 8-minute presentation · ≥ 4 baselines (incl. initial RF) · PDF submission

---

## ⚠️ WHAT STILL NEEDS TO BE DONE (before building slides)

| # | Action Item | Owner | Priority |
|---|-------------|-------|----------|
| 1 | **Add 2–3 more baseline models** (Logistic Regression, Decision Tree, KNN) using the engineered feature set from `preprocess.py`. Record val accuracy for each. | Team | 🔴 Critical |
| 2 | **Run `model3_xgboost.py` and record accuracy** — screenshot or log the final val accuracy number | Team | 🔴 Critical |
| 3 | **Run `model4_neural_network.py` and record accuracy** — also log the loss curve (training vs. validation loss by epoch) | Team | 🔴 Critical |
| 4 | **Generate feature importance plot from XGBoost** (already in script — save `output/plot_m3_xgb_feature_importance.png`) | Team | 🟡 High |
| 5 | **Generate MLP loss curve plot** (add `model.loss_curve_` plot to `model4_neural_network.py`) | Team | 🟡 High |
| 6 | **Create a results comparison table** with all models side-by-side (accuracy, F1, training time) | Team | 🟡 High |
| 7 | **Kaggle leaderboard score** — submit predictions and note public leaderboard score for at least the best model | Team | 🟡 High |
| 8 | **SHAP or permutation importance** for MLP interpretability | Team | 🟢 Nice-to-have |

---

## SLIDE-BY-SLIDE BLUEPRINT

### ── SECTION 1: FRAMING (Slides 1–2) ──

---

### SLIDE 1 — Title Slide
**⏱ ~15 seconds**

**Content:**
- Title: *Predicting Spaceship Titanic Passenger Transport: A Multi-Model Approach*
- Subtitle: MSCI 546 Advanced Machine Learning — Group 6
- Team member names
- Date

**Visual:** Stylized space/galaxy background image (subtle, doesn't distract). Optional: Kaggle logo + competition badge.

**Speaker note:** No narration needed — let the title speak. Transition immediately.

---

### SLIDE 2 — Agenda
**⏱ ~15 seconds**

**Content:** Simple 6-item list with icons:
1. Task & Motivation
2. Data & EDA
3. Metrics
4. Baselines
5. Our Solution
6. Results & Insights

**Visual:** Clean numbered list, possibly with a mini timeline bar at the bottom showing where you are in the talk.

**Speaker note:** "We'll walk through our full ML pipeline — from data to deployment — and show you how we beat the baseline by [X]%."

---

### ── SECTION 2: TASK (Slides 3) ──

---

### SLIDE 3 — Task Definition
**⏱ ~45 seconds**

**Content:**
- **Competition:** Spaceship Titanic — Kaggle (2022)
- **ML Task:** Binary Classification → predict `Transported` (True/False)
- **Real-world framing:** In 2912, the Spaceship Titanic collides with a spacetime anomaly. Did each passenger get transported to another dimension?
- **Why it matters (academically):** Tabular classification with mixed feature types — ideal testbed for ensemble methods and neural networks

**Visual:**
- Left: A 2-column "Input → Output" diagram showing raw features flowing into a model box outputting True/False
- Right: Kaggle competition screenshot or badge

**Diagram to create:**
```
[PassengerId, HomePlanet, CryoSleep, Age, Spending...]
             ↓
      [ ML Model ]
             ↓
   Transported: True / False
```

**Speaker note:** "The task is binary classification. Given 13 passenger features, we predict whether each person was transported to an alternate dimension. The target is perfectly balanced at ~50/50."

---

### ── SECTION 3: DATA (Slides 4–5) ──

---

### SLIDE 4 — Dataset Overview
**⏱ ~40 seconds**

**Content:**

**Raw Data:**
| Split | Rows | Columns | Has Target? |
|-------|------|---------|-------------|
| Train | 8,693 | 14 | ✅ Yes |
| Test | 4,277 | 13 | ❌ No |

**Feature types:**
- **Numerical (6):** Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- **Categorical (4):** HomePlanet, CryoSleep, Destination, VIP
- **String/Complex (2):** Cabin (format: Deck/Num/Side), Name
- **ID (1):** PassengerId (format: GroupNum_MemberNum)

**Visual:** Feature table with color-coded types. Small callout: "~2% missing values across most columns."

**Speaker note:** "We have 14 raw features — a healthy mix of numerical, categorical, and complex string features. Notice that `Cabin` and `PassengerId` encode rich structural information we can parse out."

---

### SLIDE 5 — EDA: Key Predictors
**⏱ ~50 seconds**

**Content:** 4-panel figure with captions — use existing EDA plots from `run.py`

**Visuals to use (all already generated):**
1. `plot_1_cryosleep_impact.png` — CryoSleep stacked bar (dominant predictor, >75% transport rate)
2. `plot_2_age_distribution.png` — KDE plot, spike at ages 0–5
3. `plot_3_cabin_side.png` — Starboard vs Port transport rates
4. Spending correlation heatmap

**Layout:** 2×2 grid of plots, each with a 1-line takeaway caption

**Captions to add to each plot:**
- CryoSleep: *"CryoSleep is the #1 predictor — 75%+ transport rate"*
- Age 0–5: *"Toddlers show a non-linear spike in transport probability"*
- Cabin Side: *"Starboard passengers transported more than Port passengers"*
- Spending: *"Active spending strongly protects against transport"*

**Speaker note:** "Four EDA findings shaped our entire modeling strategy. CryoSleep dominates. The age spike at 0–5 suggests a non-linear relationship. We engineered cabin side and spending features based on these insights."

---

### ── SECTION 4: FEATURE ENGINEERING (Slide 6) ──

---

### SLIDE 6 — Feature Engineering Pipeline
**⏱ ~45 seconds**

**Content:**

**From 14 raw features → 38 engineered features** (via `preprocess.py`)

| Engineering Step | Features Created | Why |
|-----------------|-----------------|-----|
| Parse `Cabin` | `Cabin_Deck`, `Cabin_Num`, `Cabin_Side` | Ship geography matters |
| Parse `PassengerId` | `Group_ID`, `Group_Size`, `Is_Alone` | Family groups behave differently |
| Domain-aware imputation | CryoSleep passengers → spending = 0 | Avoids false signal |
| Group-mode imputation | HomePlanet, Destination fill | Reduces noise |
| Derived spending | `Total_Spending`, `Has_Spent` | Aggregate > individual |
| Age bins | `Is_Child`, `Age_Group` | Captures non-linear age effect |
| OHE + boolean cast | All categoricals encoded | Model-ready format |

**Visual:** A pipeline diagram:
```
Raw CSV (14 cols)
     ↓
[Phase 2] Cabin & ID Parsing → +3 cols
     ↓
[Phase 3] CryoSleep Imputation → fixes nulls
     ↓
[Phase 4-6] Group + Statistical Imputation → 0 nulls
     ↓
[Phase 7] Derived Features (spending, age) → +4 cols
     ↓
[Phase 8] One-Hot Encoding → +16 cols
     ↓
Processed CSV (38 cols) — 0 missing values
```

**Speaker note:** "Our feature engineering pipeline runs in 8 phases. Key insight: we used domain knowledge — passengers in CryoSleep cannot spend money — so we zero-filled their spending nulls rather than imputing median values. This is a non-trivial model-building decision."

---

### ── SECTION 5: METRICS (Slide 7) ──

---

### SLIDE 7 — Evaluation Metrics
**⏱ ~30 seconds**

**Content:**

**Primary Metric: Classification Accuracy**
- Matches Kaggle leaderboard evaluation
- Valid here because: **classes are balanced (~50/50)** → no majority-class bias

**Secondary Metric: F1-Score**
- Harmonic mean of Precision & Recall
- Ensures we aren't sacrificing false negative rate

**Visual:**
- Formula for accuracy and F1 (clean LaTeX-style typography)
- Confusion matrix template with quadrant labels (TP, FP, FN, TN)
- Small note: "Balanced classes → accuracy is a fair metric here"

**Diagram:**
```
         Predicted
          T    F
Actual T [ TP | FN ]   ← Transported passengers
       F [ FP | TN ]   ← Not transported
```

**Speaker note:** "We use accuracy as our primary metric — directly aligning with the Kaggle leaderboard. Because the dataset is perfectly balanced at 50/50, accuracy is a meaningful measure and F1 stays consistent with it."

---

### ── SECTION 6: BASELINES (Slides 8–11) ──

> **Required: ≥ 4 baselines including the initial Random Forest**

---

### SLIDE 8 — Baseline 1: Initial Random Forest (Initial Baseline)
**⏱ ~30 seconds**

**Content:**
- Model: `RandomForestClassifier(n_estimators=100, max_depth=10)`
- Features: 10 raw features only (no feature engineering — simple median/mode imputation)
- Pipeline: scikit-learn `Pipeline` + `ColumnTransformer`
- **Validation Accuracy: 78.84%**
- **F1-Score: [fill in]**

**Visual:**
- Confusion matrix plot (already generated in `run.py`)
- Small model spec box

**Speaker note:** "Our initial baseline was a vanilla Random Forest on raw features with no engineering. This gave us 78.84% validation accuracy — a solid starting point. Everything we do from here is measured against this."

---

### SLIDE 9 — Baseline 2: Logistic Regression
**⏱ ~25 seconds**

> **⚠️ TODO: Run this model. Add ~10 lines of code using the processed feature set.**

**Content:**
- Model: `LogisticRegression(C=1.0, max_iter=1000)`
- Features: Full engineered feature set (38 features, from `preprocess.py`)
- Rationale: Linear baseline — establishes a lower bound for what a linear model can do on engineered features
- **Validation Accuracy: [FILL IN — expected ~79-81%]**
- **F1-Score: [FILL IN]**

**Visual:** Coefficient magnitude bar chart (top 10 features by |coef|) — 5 lines of matplotlib code to generate

**Speaker note:** "Logistic Regression on our engineered features shows how much the feature engineering alone contributes — even without a powerful model."

---

### SLIDE 10 — Baseline 3: Decision Tree
**⏱ ~25 seconds**

> **⚠️ TODO: Run this model.**

**Content:**
- Model: `DecisionTreeClassifier(max_depth=8, random_state=42)`
- Features: Full engineered feature set
- Rationale: Single-tree interpretable baseline — shows ceiling for a single deterministic splitter
- **Validation Accuracy: [FILL IN — expected ~75-78%]**
- **F1-Score: [FILL IN]**

**Visual:** Decision tree top-3 levels visualization (use `sklearn.tree.plot_tree`, `max_depth=3`) — shows CryoSleep as root split

**Speaker note:** "The Decision Tree is valuable here because it produces an interpretable model. We can literally read the tree and see that CryoSleep is the first split — confirming our EDA findings."

---

### SLIDE 11 — Baseline 4: K-Nearest Neighbours
**⏱ ~25 seconds**

> **⚠️ TODO: Run this model.**

**Content:**
- Model: `KNeighborsClassifier(n_neighbors=15)` with StandardScaler preprocessing
- Features: Full engineered feature set, scaled
- Rationale: Instance-based learner — no parametric assumptions, tests local similarity structure
- **Validation Accuracy: [FILL IN — expected ~77-80%]**
- **F1-Score: [FILL IN]**

**Visual:** K vs. Accuracy curve (test k=5, 10, 15, 20, 25 and plot) — shows model selection rationale

**Speaker note:** "KNN tests whether passenger similarity is locally structured. We tuned k and selected [k=15] based on validation accuracy."

---

### ── SECTION 7: PROPOSED SOLUTION (Slides 12–14) ──

---

### SLIDE 12 — Our Approach: Two-Model Strategy
**⏱ ~30 seconds**

**Content:**
- We implemented two advanced models as our proposed solutions:
  1. **XGBoost** — gradient-boosted trees optimized for tabular data
  2. **MLP Neural Network** — deep learning with engineered features

**Visual:** Simple 2-column layout comparing the "family" of each approach:
```
┌─────────────────────┬──────────────────────┐
│   XGBoost           │   MLP Neural Net     │
│ Tree-based ensemble │ Deep learning        │
│ Handles non-linear  │ Learns representations│
│ Built-in feature    │ Flexible architecture │
│ importance          │ Requires scaling     │
└─────────────────────┴──────────────────────┘
```

**Speaker note:** "We chose two architecturally distinct models to cover both tree-based and neural approaches, matching the course requirement for a neural network method."

---

### SLIDE 13 — Model A: XGBoost
**⏱ ~45 seconds**

**Content:**

**Architecture / Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 500 | More trees, early stopping prevents overfit |
| learning_rate | 0.05 | Slow learning → better generalization |
| max_depth | 6 | Controls individual tree complexity |
| subsample | 0.8 | Row sampling → reduces variance |
| colsample_bytree | 0.8 | Feature sampling → reduces correlation |
| eval_metric | logloss | Probabilistic loss for binary task |

**Results:**
- **Validation Accuracy: [FILL IN]**
- **F1-Score: [FILL IN]**
- **Kaggle Public Score: [FILL IN]**

**Visuals (both already generated in script):**
- `output/plot_m3_1_feature_correlation.png` — top 20 features by correlation
- `output/plot_m3_2_confusion_matrix.png` — confusion matrix

> **⚠️ TODO: Also generate XGBoost native feature importance plot:**
> ```python
> from xgboost import plot_importance
> plot_importance(model, max_num_features=15)
> plt.savefig('output/plot_m3_3_xgb_importance.png', ...)
> ```

**Speaker note:** "XGBoost is our primary tree-based solution. We used 500 boosting rounds with a slow learning rate and two sampling strategies to prevent overfitting. The validation set was passed as an eval set, allowing us to monitor training live."

---

### SLIDE 14 — Model B: Neural Network (MLP)
**⏱ ~45 seconds**

**Content:**

**Architecture:**
```
Input (38 features)
     ↓
Dense(256, ReLU)
     ↓
Dense(128, ReLU)
     ↓
Dense(64, ReLU)
     ↓
Output(1, Sigmoid) → Transported probability
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Regularization | L2 α=0.001 |
| Batch size | 256 |
| Max epochs | 500 (early stopping at n=20) |
| Preprocessing | StandardScaler on 9 numerical cols |

**Results:**
- **Validation Accuracy: [FILL IN]**
- **F1-Score: [FILL IN]**
- **Stopped at epoch: [FILL IN]**

**Visuals:**
- Architecture diagram (the one above, prettified)
- Learning curve: training vs validation loss by epoch

> **⚠️ TODO: Generate learning curve plot:**
> ```python
> plt.plot(model.loss_curve_, label='Training Loss')
> plt.plot(model.validation_scores_, label='Val Accuracy')
> plt.savefig('neural_output/plot_m4_2_learning_curve.png')
> ```

**Speaker note:** "The MLP takes our 38 engineered features with standardized numerical columns. We used early stopping with a patience of 20 epochs to prevent overfitting. Key design choice: StandardScaler was fit only on the training set."

---

### ── SECTION 8: RESULTS & INSIGHTS (Slides 15–18) ──

---

### SLIDE 15 — Model Comparison: All Baselines vs. Solutions
**⏱ ~45 seconds**

**Content:** Master results table

| Model | Val Accuracy | F1-Score | Notes |
|-------|-------------|---------|-------|
| **Baseline 1: Random Forest (Initial)** | 78.84% | [fill] | Raw features, no engineering |
| **Baseline 2: Logistic Regression** | [fill] | [fill] | Engineered features |
| **Baseline 3: Decision Tree** | [fill] | [fill] | Engineered features |
| **Baseline 4: KNN (k=15)** | [fill] | [fill] | Scaled engineered features |
| **XGBoost (Proposed)** | [fill] | [fill] | 500 trees, tuned HPs |
| **MLP Neural Net (Proposed)** | [fill] | [fill] | 256→128→64, early stop |

**Visual:**
- Horizontal bar chart sorted by accuracy (all models, color-coded: baselines in grey, proposed solutions in dark purple)
- Add a vertical dashed line at 78.84% (initial baseline) to visually show improvement

**Speaker note:** "Here's the full picture. The dashed line marks our initial baseline at 78.84%. [Point to proposed models] Our XGBoost/MLP achieves [X]%, an improvement of [Y] percentage points."

---

### SLIDE 16 — Feature Importance Insights
**⏱ ~45 seconds**

**Content:**

**Top features (from XGBoost feature importance):**
- CryoSleep: dominant (confirms EDA)
- Total_Spending / Has_Spent: high importance (validates engineering decision)
- Cabin_Deck, Cabin_Side: moderate importance (validates cabin parsing)
- Age / Is_Child: moderate importance (validates age binning)
- Group features (Is_Alone, Group_Size): lower but present

**Visual:**
- XGBoost native feature importance bar chart (horizontal, top 15 features)
- Side annotation: "Features our EDA predicted would matter — confirmed by the model"

> **⚠️ TODO: Generate this plot (see TODO on Slide 13)**

**Speaker note:** "Feature importance from XGBoost confirms every major EDA hypothesis. CryoSleep is #1. Our engineered spending and cabin features all rank highly — this validates the feature engineering effort."

---

### SLIDE 17 — Model Interpretability
**⏱ ~40 seconds**

**Content:**

**Two interpretability views:**

1. **Decision Tree (Baseline 3) — Top 3 levels**
   - Root split: CryoSleep = True/False → cleanly separates >75% of transported passengers
   - Shows model is learning what we found in EDA
   - Interpretable by design

2. **MLP — Total Spending KDE by Transport Status** (plot already generated: `neural_output/plot_m4_1_spending_distribution.png`)
   - Passengers with Total_Spending ≈ 0 are overwhelmingly transported
   - Distribution is bimodal — confirms spending is a strong separator

**Visual:** Side-by-side: decision tree diagram (left) + spending KDE plot (right)

**Speaker note:** "Two lenses on model behavior. The decision tree literally shows us the rules. The spending distribution from the neural network analysis confirms the non-linear signal we engineered — high spending protects you, zero spending is suspicious."

---

### SLIDE 18 — Improvement Over Baselines
**⏱ ~30 seconds**

**Content:**

| Comparison | Δ Accuracy | Insight |
|------------|-----------|---------|
| XGBoost vs. Initial RF | +[X]pp | Benefit of boosting + engineering |
| MLP vs. Initial RF | +[X]pp | Benefit of deep features |
| Best model vs. Initial RF | +[X]pp | Total gain |

**Visual:**
- Simple lift chart: bar for each model showing absolute improvement over 78.84%
- Callout: "Feature engineering alone contributed approximately [X]pp" (compare LR on raw vs. LR on engineered)

**Speaker note:** "Our best model improves by [X] percentage points over the initial baseline. Notably, much of this gain comes from feature engineering — the Logistic Regression on engineered features already outperforms the raw-feature Random Forest."

---

### ── SECTION 9: CONCLUSION & REFERENCES (Slides 19–20) ──

---

### SLIDE 19 — Key Takeaways
**⏱ ~30 seconds**

**Content (5 bullets max — keep concise):**
1. **CryoSleep is the dominant signal** — confirmed by EDA, feature importance, and the decision tree
2. **Feature engineering drove the biggest gains** — parsing Cabin, groups, and aggregating spending
3. **XGBoost outperforms MLP** [or vice versa — fill in after results] on this tabular dataset
4. **Domain knowledge matters** — CryoSleep imputation rule was a non-trivial design decision
5. **Room to grow:** Ensemble/stacking, SHAP analysis, and hyperparameter search with cross-validation

**Visual:** Clean 5-point list with icons. Optional: "podium" bar chart showing top 3 models.

**Speaker note:** "Three takeaways: feature engineering was our biggest lever, tree-based models win on tabular data, and domain knowledge beats pure ML automation. Thank you — happy to take questions."

---

### SLIDE 20 — References
**⏱ (no speaking required)**

**Content:**

Howard, A., Chow, A., & Holbrook, R. (2022). *Spaceship Titanic*. Kaggle. https://kaggle.com/competitions/spaceship-titanic

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

[Add any papers cited for specific methods used]

---

## TIMING BREAKDOWN (8-minute hard cap)

| Section | Slides | Time |
|---------|--------|------|
| Title + Agenda | 1–2 | ~30 sec |
| Task | 3 | ~45 sec |
| Data + EDA | 4–5 | ~90 sec |
| Feature Engineering | 6 | ~45 sec |
| Metrics | 7 | ~30 sec |
| Baselines (4×) | 8–11 | ~105 sec |
| Proposed Solution | 12–14 | ~120 sec |
| Results + Insights | 15–18 | ~120 sec |
| Takeaways | 19 | ~30 sec |
| **Total** | **19 content + 1 refs** | **~8 min** ✅ |

---

## DIAGRAMS TO CREATE / GENERATE

| # | Diagram | How to Get It | Slide |
|---|---------|--------------|-------|
| 1 | Task input→output flow diagram | Create in PowerPoint/Canva (or ASCII art) | 3 |
| 2 | Feature engineering pipeline diagram | Create in PowerPoint/draw.io | 6 |
| 3 | 2×2 EDA grid (existing plots) | Already in `output/` from `run.py` | 5 |
| 4 | XGBoost feature importance | Add 5 lines to model3_xgboost.py, run it | 13, 16 |
| 5 | MLP learning curve (loss by epoch) | Add 5 lines to model4_neural_network.py, run it | 14 |
| 6 | Master comparison bar chart | Generate in Python after all models run | 15 |
| 7 | Improvement lift chart | Generate in Python after all models run | 18 |
| 8 | Decision tree top-3 levels | `sklearn.tree.plot_tree(..., max_depth=3)` | 10, 17 |
| 9 | KNN k vs. accuracy curve | Loop k=5,10,15,20,25, plot val accuracy | 11 |
| 10 | Spending KDE (already exists) | `neural_output/plot_m4_1_spending_distribution.png` | 17 |

---

## GRADING RUBRIC CHECKLIST

| Rubric Item | Status | Where in Deck |
|-------------|--------|--------------|
| ✅ Task clearly described | Ready | Slide 3 |
| ✅ Data described | Ready | Slide 4 |
| ✅ Metrics defined | Ready | Slide 7 |
| ⚠️ ≥ 4 baselines (incl. initial RF) | **Needs 3 more models** | Slides 8–11 |
| ✅ Initial baseline (RF 78.84%) | Ready | Slide 8 |
| ✅ Proposed solution described | Ready | Slides 12–14 |
| ✅ Visualizations & interpretations | Mostly ready | Slides 5, 16, 17 |
| ⚠️ Improvement over baselines shown | **Needs result numbers** | Slides 15, 18 |
| ✅ ≤ 20 slides including references | Ready (20 slides) | — |
| ✅ References included | Ready | Slide 20 |
| ✅ Quality of slides | Depends on design | — |
| ⚠️ Key takeaways | **Fill in after results** | Slide 19 |

---

## RECOMMENDED SLIDE DESIGN TIPS

- **Font:** Keep to 2 fonts max. Title font 28–32pt, body 18–22pt, captions 14pt.
- **Colour scheme:** Match existing plots (dark purple `#3B1E54` + light purple `#D4BEE4` + white)
- **1 idea per slide rule:** Don't cram — if a slide needs scrolling, split it
- **Every plot needs:** a title, axis labels, and a 1-sentence takeaway caption
- **Consistency:** Same confusion matrix style across all models
- **PDF export:** Use Google Slides → File → Download → PDF; verify plots don't blur

---

*Generated by Claude · MSCI 546 Group 6 · March 2026*
