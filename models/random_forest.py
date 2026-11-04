#  *************************************************************************
#  * File:              random_forest.py
#  * Author:            Justin Williams
#  * Date:              11/3/2026
#  * File Description:  A demonstration of the Random Forest Classification
#  *                    model.
#  *************************************************************************

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

TARGET_COL = "Recurred"
CATEGORICAL_COLS = [
    "Gender",
    "Smoking",
    "Hx Smoking",
    "Hx Radiotherapy",
    "Thyroid Function",
    "Physical Examination",
    "Adenopathy",
    "Pathology",
    "Focality",
    "Risk",
    "T",
    "N",
    "M",
    "Stage",
    "Response"
]

RAND_STATE = 42

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv("data/processed/Thyroid_Diff.csv")

# -------------------------------
# DATA PREPROCESSOR
# -------------------------------

x = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].map({"No": 0, "Yes": 1})

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=RAND_STATE
)

# Using ColumnTransformer to avoid data leakage from test split during training
# and enforce same One Hot encoding across the split
colTransform = ColumnTransformer(
    transformers=[
        ("cat", preprocessing.OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", ["Age"])
    ],
    verbose_feature_names_out=False
)

# -------------------------------
# MODEL TRAINING
# -------------------------------

model = Pipeline([
    ("trans", colTransform),
    ("clf", RandomForestClassifier(max_depth=5, n_estimators=10, random_state=RAND_STATE))
])

model.fit(x_train, y_train)

# -------------------------------
# MODEL EVALUATION
# -------------------------------

print("RandomForestClassifier Accuracy:", model.score(x_test, y_test))