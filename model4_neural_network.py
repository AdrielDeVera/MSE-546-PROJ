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

# ============================================================================
# [2/7] EDA — SPENDING FEATURE DISTRIBUTIONS
# ============================================================================

print("\n[2/7] Generating EDA visualizations...")

sns.set_theme(style="whitegrid")
custom_palette = ["#3B1E54", "#D4BEE4"]

plot_df = train.copy()
plot_df['Transported'] = plot_df['Transported'].astype(str)

print("  → Plot 1: Total_Spending KDE by Transported label")
plt.figure(figsize=(12, 6))
sns.kdeplot(data=plot_df, x='Total_Spending', hue='Transported', fill=True,
            palette=custom_palette, alpha=0.6, linewidth=2)
plt.title('Total Spending Distribution by Transport Status', fontsize=14, fontweight='bold')
plt.xlabel('Total Spending', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(left=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot_m4_1_spending_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved → {OUTPUT_DIR}/plot_m4_1_spending_distribution.png")

# ============================================================================
# [3/7] TRAIN/VAL SPLIT
# ============================================================================

print("\n[3/7] Splitting data (80/20, random_state=42)...")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Training set  : {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print("  ✓ Split complete!")
