import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 80)
print("SPACESHIP TITANIC - PROFESSIONAL EDA & BASELINE MODEL")
print("=" * 80)

# ============================================================================
# PHASE 1: SETUP & PROFESSIONAL EDA
# ============================================================================

print("\n[1/6] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"\nFirst few rows:")
print(train.head())

# Setup "Professional" Plot Style
sns.set_theme(style="whitegrid")
# A custom palette that looks 'sci-fi' (cool blues and purples)
custom_palette = ["#3B1E54", "#D4BEE4"]

# Data Prep for Plotting
plot_df = train.copy()
plot_df['Transported'] = plot_df['Transported'].astype(str)

print("\n[2/6] Generating professional EDA visualizations...")

# --- PLOT 1: The "CryoSleep" Paradox (Stacked Bar) ---
# Why this isn't "simple": It uses percentages (normalized), not just raw counts.
# This proves CryoSleep is a huge risk factor.
print("  → Plot 1: CryoSleep Impact (Stacked Bar)")
plt.figure(figsize=(10, 6))
ct = pd.crosstab(plot_df['CryoSleep'], plot_df['Transported'], normalize='index')
ct.plot(kind='bar', stacked=True, color=custom_palette, figsize=(10, 6))
plt.title('Impact of CryoSleep on Transport Probability', fontsize=14, fontweight='bold')
plt.xlabel('In CryoSleep?', fontsize=12)
plt.ylabel('Proportion of Passengers', fontsize=12)
plt.legend(title='Transported', loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot_1_cryosleep_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 2: Age Distribution (KDE Plot) ---
# Why this isn't "simple": It uses a Kernel Density Estimate (smooth curve)
# instead of a blocky histogram. It clearly highlights the "Child Spike" at age 0-5.
print("  → Plot 2: Age Distribution (KDE)")
plt.figure(figsize=(12, 6))
sns.kdeplot(data=plot_df, x='Age', hue='Transported', fill=True,
            palette=custom_palette, alpha=0.6, linewidth=2)
plt.title('Age Distribution by Transport Status', fontsize=14, fontweight='bold')
plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('plot_2_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 3: Feature Engineering "Cabin Side" ---
# Why this isn't "simple": It requires parsing the string data first.
# Shows "Strategic Difficulty" by creating new data.
print("  → Plot 3: Cabin Side Analysis (Starboard vs Port)")
plot_df[['Deck', 'Num', 'Side']] = plot_df['Cabin'].str.split('/', expand=True)

plt.figure(figsize=(10, 6))
sns.countplot(data=plot_df, x='Side', hue='Transported', palette=custom_palette)
plt.title('Starboard (S) vs. Port (P) Survival Rates', fontsize=14, fontweight='bold')
plt.xlabel('Cabin Side', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Transported')
plt.tight_layout()
plt.savefig('plot_3_cabin_side.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 4: Correlation Matrix (The "Triangle" Mask) ---
# Why this isn't "simple": Standard heatmaps duplicate data (top right = bottom left).
# This code masks the upper triangle for a clean, professional look.
print("  → Plot 4: Correlation Matrix (Masked Triangle)")
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
corr = plot_df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # The magic mask

plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Between Spending Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_4_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ All 4 professional plots saved!")

# ============================================================================
# PHASE 2: THE "SMART" BASELINE MODEL
# ============================================================================

print("\n[3/6] Building Smart Baseline Model (Random Forest with Pipeline)...")

# 1. Feature Selection
# We drop 'Name' (text) and 'Cabin' (complex string) for the *simple* baseline.
# We will use 'Cabin' in later weeks.
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
target = 'Transported'

X = train[features]
y = train[target]

# 2. Preprocessing Pipeline
# Numerical data: Fill missing with Median
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numerical_transformer = SimpleImputer(strategy='median')

# Categorical data: Fill missing with Mode, then One-Hot Encode
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Full Pipeline with Random Forest
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
])

print("  ✓ Pipeline created with:")
print(f"    - Numerical features: {len(numerical_cols)}")
print(f"    - Categorical features: {len(categorical_cols)}")

# ============================================================================
# PHASE 3: TRAIN/VALIDATION SPLIT & TRAINING
# ============================================================================

print("\n[4/6] Splitting data and training model...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")

# Train the model
model_pipeline.fit(X_train, y_train)
print("  ✓ Model training complete!")

# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================

print("\n[5/6] Evaluating model performance...")

# Predictions
y_train_pred = model_pipeline.predict(X_train)
y_val_pred = model_pipeline.predict(X_val)

# Accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\n  Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

print("\n  Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('plot_5_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Confusion matrix saved!")

# Feature Importance Plot
print("\n[6/6] Generating feature importance plot...")
feature_names = (numerical_cols +
                 list(model_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_cols)))

importances = model_pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features

plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importances[indices], color='#3B1E54')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plot_6_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature importance plot saved!")

# ============================================================================
# PHASE 5: GENERATE PREDICTIONS FOR SUBMISSION
# ============================================================================

print("\n[7/7] Generating predictions for test set...")
X_test = test[features]
test_predictions = model_pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_predictions
})
submission.to_csv('submission_baseline_rf.csv', index=False)
print(f"  ✓ Submission file created: submission_baseline_rf.csv")
print(f"  → {len(submission)} predictions generated")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("EXECUTION COMPLETE!")
print("=" * 80)
print("\nFiles Generated:")
print("  1. plot_1_cryosleep_impact.png")
print("  2. plot_2_age_distribution.png")
print("  3. plot_3_cabin_side.png")
print("  4. plot_4_correlation_matrix.png")
print("  5. plot_5_confusion_matrix.png")
print("  6. plot_6_feature_importance.png")
print("  7. submission_baseline_rf.csv")
print("\nModel Performance:")
print(f"  → Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"  → Ready for Kaggle submission!")
print("=" * 80)
