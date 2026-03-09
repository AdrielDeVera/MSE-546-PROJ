import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUTPUT_DIR = 'neural_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE_COLS = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
              'Cabin_Num', 'Group_Size', 'Total_Spending']

print("=" * 80)
print("SPACESHIP TITANIC - MODEL 4: NEURAL NETWORK (MLPClassifier)")
print("=" * 80)

# ============================================================================
# [1/7] LOAD PROCESSED DATA
# ============================================================================

print("\n[1/7] Loading processed data...")

train = pd.read_csv('output/train_processed.csv')
test  = pd.read_csv('output/test_processed.csv')

DROP_COLS = ['PassengerId', 'Group_ID', 'Transported']
X = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
y = train['Transported'].astype(int)
X_test_full = test.drop(columns=[c for c in ['PassengerId', 'Group_ID'] if c in test.columns])
test_ids = test['PassengerId']

print(f"  Train shape : {train.shape}")
print(f"  Test  shape : {test.shape}")
print(f"  Features    : {X.shape[1]}")
print(f"  Target dist :\n{y.value_counts().to_string()}")
print(f"\n  Scale cols  : {SCALE_COLS}")
print(f"  Output dir  : {OUTPUT_DIR}/")
print("  ✓ Data loaded successfully!")
