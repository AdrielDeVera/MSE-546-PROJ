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
