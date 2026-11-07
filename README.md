# Predicting Thyroid Cancer Recurrence

**Team**: Ricky Wiggins, Justin Williams, Robert McLean

A machine learning classification study that predicts whether **differentiated thyroid cancer recurs** after initial treatment. It uses clinicopathologic patient data from the UCI Machine Learning Repository. We train a range of classifiers on a shared preprocessing pipeline and compare them, focusing on recall for the recurrence class.

---

## The Problem

Differentiated thyroid cancer is usually treatable, but in some patients it comes back after initial treatment. Knowing who is at risk determines how closely a patient needs to be followed.

**Question:** Given a patient's clinical profile after treatment, can we predict whether their cancer will recur?

**Why it matters:** Reliable risk prediction could identify low-risk patients who can safely avoid intensive follow-up, and flag high-risk patients who need closer monitoring.

**Why ML fits:** Recurrence depends on many interacting factors at once, and a classifier is built to learn exactly that kind of pattern from labeled data.

---

## Dataset

[UCI Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) (donated 2023)

| Property | Value |
|---|---|
| Patients | 383, each followed for at least 10 years (15-year cohort) |
| Features | 16 clinicopathologic features (mostly categorical; age is the only numeric field) |
| Target | `Recurred` (Yes / No) |
| Class balance | About 1 in 4 patients recurred, so the target is **imbalanced** |
| Missing values | None |

**Feature groups**

- **Demographics:** Age, Gender
- **History:** Smoking, Hx Smoking, Hx Radiotherapy
- **Clinical:** Thyroid Function, Physical Examination, Adenopathy
- **Pathology:** Pathology, Focality, Risk
- **Staging:** T, N, M, Stage
- **Treatment:** Response (response to initial therapy; recorded after treatment and likely a strong predictor)

**Files**

- `data/raw/Thyroid_Diff.csv`: the original UCI file
- `data/processed/Thyroid_Diff.csv`: a cleaned copy (it fixes the `Hx Radiothreapy` column-name typo, renaming it `Hx Radiotherapy`)

---

## Pipeline

Every model gets the same preprocessing, so any difference in results comes from the models themselves.

1. **Encode & scale**: one-hot encode the categorical features and standardize `Age` so that no feature dominates because of its range.
2. **PCA**: one-hot encoding widens the feature space on only 383 rows. PCA compresses it and lets us visualize the classes.
3. **Handle imbalance**: apply SMOTE to the **training split only**, and compare it against class weighting.
4. **Train models**: fit each classifier on stratified training data and tune its hyperparameters with cross-validation.
5. **Evaluate**: compare models on recall and the confusion matrix, not accuracy alone.

---

## Models

SVM and k-NN are the two starting models. They approach the problem in opposite ways, which makes them a useful contrast. The rest of the lineup covers the field, and the results decide which model wins.

| Model | What it adds | Trade-off |
|---|---|---|
| **Support Vector Machine** (starting model) | Maximum-margin boundary; compares linear and RBF kernels and tunes `C` and `gamma` | Less interpretable |
| **k-Nearest Neighbors** (starting model) | Simple "patients like this one" baseline; sweeps `k` and tests running PCA first | Sensitive to the curse of dimensionality |
| Logistic Regression | Interpretable coefficients that show which factors drive recurrence | Assumes a mostly linear relationship |
| Decision Tree | Human-readable rules a clinician could follow | Overfits easily on small data |
| Random Forest | Usually more accurate than one tree; ranks feature importance | Less transparent than a single tree |
| Naive Bayes | Fast, strong baseline for categorical features | Assumes the features are independent |
| Gradient Boosting | Often the top performer on tabular data | More hyperparameters to tune |

Published work on this dataset suggests SVM and logistic regression tend to perform best, while k-NN is weaker. We keep an open field rather than committing to one model early.

---

## Evaluation

With about 25% recurrence, a model that always predicts "no recurrence" scores about 75% accuracy and is useless. So we use:

- **Recall / sensitivity (headline metric):** of the patients who actually recurred, how many did we catch? A missed recurrence is the costly error.
- **Confusion matrix:** the full breakdown of where each model fails.
- **Stratified cross-validation:** with only 383 rows, this keeps one lucky split from deciding the winner.

### Fixing the imbalance

- **SMOTE (primary):** creates synthetic recurrence cases by interpolating between real cases and their nearest neighbors. We apply it to the training data only, so the test set stays honest.
- **Class weighting (alternative):** makes missed recurrences more costly during training, without creating new data.

We compare both approaches to see which one lifts recall more.

---

## Project Structure

```
thyroid-recurrence/
├── data/
│   ├── raw/            # Original UCI dataset
│   └── processed/      # Cleaned dataset
├── models/             # Model notebooks (one per classifier)
├── results/            # Metrics, plots, confusion matrices
├── src/                # Shared preprocessing / evaluation code
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/rwiggins1/thyroid-recurrence.git
```

```bash
cd thyroid-recurrence
```

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Then open a notebook in `models/` with Jupyter or your editor of choice.
---

## Future Work

1. **Feature importance & explanation:** rank which factors best predict recurrence, so the model explains risk rather than only flagging it.
2. **Before vs. after treatment:** train on diagnosis-time features alone, then add `Response`. Does treatment response really drive the prediction?
3. **Patient subgroups via clustering:** use unsupervised clustering to find natural risk profiles and check them against the known stage and risk labels.
4. **Deeper imbalance handling:** tune the resampling and measure how much each method lifts recurrence recall.

---

## License

See [LICENSE](LICENSE).
