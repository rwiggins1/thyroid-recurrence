
from pathlib import Path
import pandas as pd

# Find the main project folder
project_root = Path(__file__).resolve().parent.parent

# Locate the original thyroid cancer dataset
dataset_path = project_root / "data" / "raw" / "Thyroid_Diff.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Display information about the dataset
print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five patients:")
print(df.head())

print("\nRecurrence counts:")
print(df["Recurred"].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Separate patient features from the prediction target
X = df.drop(columns=["Recurred"])
y = df["Recurred"].map({"No": 0, "Yes": 1})

# Identify numeric and categorical features
numeric_features = ["Age"]
categorical_features = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()

# Split patients into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Prepare the features
preprocessor = ColumnTransformer([
    ("age", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Build a pipeline containing preprocessing and KNN
knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

# Train the model
knn_model.fit(X_train, y_train)

# Predict recurrence for the test patients
y_pred = knn_model.predict(X_test)

# Evaluate the model
print("\n--- KNN Model Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# test the different K values

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n--- Testing Different K Values ---")

# Values of K we want to test
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for k in k_values:

    # Build a new model for each K value
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])

    # Evaluate recall across five training folds
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="recall"
    )

    average_recall = scores.mean()
    results.append((k, average_recall))

    print(f"K = {k:2d} | Average Recall = {average_recall:.3f}")

# Find the K value with the highest average recall
best_k, best_recall = max(results, key=lambda x: x[1])

print("\nBest K:", best_k)
print(f"Best Cross-Validation Recall: {best_recall:.3f}")


# test K = 1

# Train a final KNN model using the best K
best_knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k))
])

best_knn_model.fit(X_train, y_train)

# Predict recurrence
best_predictions = best_knn_model.predict(X_test)

# Evaluate the selected model
print("\n--- Best KNN Model Results ---")
print("Best K:", best_k)
print("Accuracy:", round(accuracy_score(y_test, best_predictions), 3))
print("Recall:", round(recall_score(y_test, best_predictions), 3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_predictions))


from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import cross_val_score

print("\n--- KNN With vs Without PCA ---")

# PCA needs dense input from the encoder
pca_preprocessor = ColumnTransformer([
    ("age", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical_features)
])

# Compare the two pipelines using training-only CV
models_to_compare = {
    "Without PCA": Pipeline([
        ("preprocessor", pca_preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k))
    ]),
    "With PCA": Pipeline([
        ("preprocessor", pca_preprocessor),
        ("pca", PCA(n_components=0.95)),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k))
    ])
}

for name, model in models_to_compare.items():

    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="recall"
    )

    print(f"{name}:")
    print(f"  Average CV Recall: {scores.mean():.3f}")
    print(f"  CV Standard Deviation: {scores.std():.3f}")
