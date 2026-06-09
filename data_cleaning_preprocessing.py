# ============================================================
# RD INFRO TECHNOLOGY - Machine Learning Internship Program
# Task 3: Data Cleaning & Preprocessing
# Dataset: Titanic Survival Dataset
# Author: Shinjini
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 60)
print("  RD INFRO TECHNOLOGY | Task 3: Data Cleaning & Preprocessing")
print("=" * 60)

df = sns.load_dataset('titanic')
print(f"\n[STEP 1] Dataset Loaded")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n  Columns: {list(df.columns)}")

print("\n[STEP 2] Initial Inspection")
print("\n  Data Types:")
print(df.dtypes.to_string())

print("\n  Missing Values (before cleaning):")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(missing_df.to_string())

print("\n  Basic Statistics (numeric columns):")
print(df.describe().round(2).to_string())

print("\n[STEP 3] Dropping Redundant Columns")

cols_to_drop = ['alive', 'class', 'who', 'adult_male', 'embark_town', 'deck']
df.drop(columns=cols_to_drop, inplace=True)

initial_len = len(df)
df.drop_duplicates(inplace=True)
print(f"  Dropped columns: {cols_to_drop}")
print(f"  Duplicate rows removed: {initial_len - len(df)}")
print(f"  Remaining shape: {df.shape}")

print("\n[STEP 4] Handling Missing Values")

age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)
print(f"  'age'     → filled with median ({age_median})")

embarked_mode = df['embarked'].mode()[0]
df['embarked'].fillna(embarked_mode, inplace=True)
print(f"  'embarked'→ filled with mode ('{embarked_mode}')")

fare_median = df['fare'].median()
df['fare'].fillna(fare_median, inplace=True)
print(f"  'fare'    → filled with median ({fare_median:.2f})")

print(f"\n  Missing values after treatment: {df.isnull().sum().sum()}")

print("\n[STEP 5] Outlier Removal (IQR Method)")

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    print(f"  '{column}': removed {before - len(df)} outliers (range [{lower:.1f}, {upper:.1f}])")
    return df

df = remove_outliers_iqr(df, 'fare')
df = remove_outliers_iqr(df, 'age')
print(f"  Shape after outlier removal: {df.shape}")
df.reset_index(drop=True, inplace=True)

print("\n[STEP 6] Encoding Categorical Variables")

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
print(f"  'sex'      → binary encoded (male=0, female=1)")

df = pd.get_dummies(df, columns=['embarked'], prefix='embarked', drop_first=True)
print(f"  'embarked' → one-hot encoded (drop_first=True to avoid dummy trap)")

df['alone'] = df['alone'].astype(int)
print(f"  'alone'    → bool converted to int")

print(f"\n  Columns after encoding: {list(df.columns)}")

print("\n[STEP 7] Feature Engineering")

df['family_size'] = df['sibsp'] + df['parch'] + 1
print(f"  Created 'family_size' = sibsp + parch + 1")

df['age_group'] = pd.cut(df['age'],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
df['age_group'] = LabelEncoder().fit_transform(df['age_group'])
print(f"  Created 'age_group' (binned + label encoded)")

print(f"  Final columns: {list(df.columns)}")

print("\n[STEP 8] Feature Scaling (StandardScaler)")

X = df.drop(columns=['survived'])
y = df['survived']

numeric_cols = ['age', 'fare', 'sibsp', 'parch', 'family_size']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"  Scaled columns: {numeric_cols}")
print(f"  Mean after scaling (should be ~0): {X[numeric_cols].mean().round(4).to_dict()}")

print("\n[STEP 9] Train-Test Split (80/20)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set : {X_test.shape[0]} samples")
print(f"  Class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")

cleaned_path = 'titanic_cleaned.csv'
df_final = pd.concat([X, y], axis=1)
df_final.to_csv(cleaned_path, index=False)
print(f"\n[STEP 10] Cleaned dataset saved → titanic_cleaned.csv")
print(f"  Final shape: {df_final.shape}")

print("\n[STEP 11] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Titanic Dataset — Data Cleaning & Preprocessing Report\nRD INFRO TECHNOLOGY Internship | Task 3',
             fontsize=14, fontweight='bold', y=1.01)

axes[0, 0].bar(['Not Survived', 'Survived'], y.value_counts().sort_index(),
               color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 0].set_title('Survival Distribution (Cleaned)')
axes[0, 0].set_ylabel('Count')
for bar in axes[0, 0].patches:
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha='center', fontsize=11)

axes[0, 1].hist(df['age'], bins=25, color='#3498db', edgecolor='black', alpha=0.85)
axes[0, 1].set_title('Age Distribution (after Scaling — standardized)')
axes[0, 1].set_xlabel('Standardized Age')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(df['fare'], bins=25, color='#9b59b6', edgecolor='black', alpha=0.85)
axes[0, 2].set_title('Fare Distribution (outliers removed)')
axes[0, 2].set_xlabel('Standardized Fare')
axes[0, 2].set_ylabel('Frequency')

sex_survival = df_final.groupby('sex')['survived'].value_counts().unstack().fillna(0)
sex_survival.index = ['Male (0)', 'Female (1)']
sex_survival.columns = ['Not Survived', 'Survived']
sex_survival.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#2ecc71'],
                  edgecolor='black', rot=0)
axes[1, 0].set_title('Survival by Gender')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(loc='upper right')

pclass_survival = df_final.groupby('pclass')['survived'].mean().round(3)
axes[1, 1].bar(['Class 1', 'Class 2', 'Class 3'], pclass_survival.values,
               color=['#f39c12', '#27ae60', '#c0392b'], edgecolor='black')
axes[1, 1].set_title('Survival Rate by Passenger Class')
axes[1, 1].set_ylabel('Survival Rate')
for i, v in enumerate(pclass_survival.values):
    axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=11)

corr_cols = ['survived', 'pclass', 'sex', 'age', 'fare', 'family_size', 'alone']
corr = df_final[corr_cols].corr()
im = axes[1, 2].imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1, 2].set_xticks(range(len(corr_cols)))
axes[1, 2].set_yticks(range(len(corr_cols)))
axes[1, 2].set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=8)
axes[1, 2].set_yticklabels(corr_cols, fontsize=8)
axes[1, 2].set_title('Correlation Heatmap')
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        axes[1, 2].text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=7, color='black')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('preprocessing_report.png', dpi=150, bbox_inches='tight')
print("  Saved: preprocessing_report.png")

print("\n" + "=" * 60)
print("  PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Original shape     : (891, 15)")
print(f"  Final shape        : {df_final.shape}")
print(f"  Missing values     : 0")
print(f"  Duplicates removed : Yes")
print(f"  Outliers removed   : Yes (IQR method)")
print(f"  Encoding           : Binary + One-Hot")
print(f"  Scaling            : StandardScaler on numeric cols")
print(f"  New features       : family_size, age_group")
print(f"  Train/Test split   : 80% / 20%")
print("=" * 60)
print("\n  Task 3 Complete! Files ready for GitHub submission.\n")
