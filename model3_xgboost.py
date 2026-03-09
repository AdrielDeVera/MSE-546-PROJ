import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

print("=" * 80)
print("SPACESHIP TITANIC - MODEL 3: XGBOOST CLASSIFIER")
print("=" * 80)

# ============================================================================
# [1/7] LOAD PROCESSED DATA
# ============================================================================

print("\n[1/7] Loading processed data...")

train = pd.read_csv('output/train_processed.csv')
test  = pd.read_csv('output/test_processed.csv')

DROP = ['PassengerId', 'Group_ID']

X      = train.drop(DROP + ['Transported'], axis=1)
y      = train['Transported'].astype(int)
X_test = test.drop(DROP, axis=1)
test_ids = test['PassengerId']

print(f"  Train features shape: {X.shape}")
print(f"  Test  features shape: {X_test.shape}")
print(f"  Target distribution:\n{y.value_counts().to_string()}")
print(f"  Features ({X.shape[1]}): {list(X.columns)}")
print("  ✓ Data loaded successfully!")

# ============================================================================
# [2/7] EDA — FEATURE CORRELATION WITH TARGET
# ============================================================================

print("\n[2/7] Generating EDA: feature correlation with target...")

correlations = X.corrwith(y).abs().sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
correlations.sort_values().plot(kind='barh', color='#3B1E54')
plt.xlabel('Absolute Correlation with Transported', fontsize=12)
plt.title('Top 20 Features by Correlation with Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/plot_m3_1_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Top 10 correlated features:")
for feat, val in correlations.head(10).items():
    print(f"    {feat:<35} {val:.4f}")
print("  ✓ Saved output/plot_m3_1_feature_correlation.png")

# ============================================================================
# [3/7] TRAIN/VALIDATION SPLIT
# ============================================================================

print("\n[3/7] Splitting data into train/validation sets...")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Training set:   {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print("  ✓ Split complete (test_size=0.2, random_state=42)")
