#  *************************************************************************
#  * File:              random_forest.py
#  * Author:            Justin Williams
#  * Date:              11/3/2026
#  * File Description:  A demonstration of the Random Forest Classification
#  *                    model.
#  *************************************************************************

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

classifier = RandomForestClassifier(max_depth=5, n_estimators=10)

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv("data/processed/Thyroid_Diff.csv")

print(df.info())

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# TODO: Encode categorical data, standardize data

# -------------------------------
# MODEL TRAINING
# -------------------------------

# TODO: Split data, train model on training split

# -------------------------------
# MODEL EVALUATION
# -------------------------------

# TODO: Evaluate model performance on testing split
