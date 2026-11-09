"""
Titanic survival prediction with logistic regression (Python).

Loads train.csv / test.csv, cleans and engineers features, trains a
logistic regression model, reports accuracy, and writes predictions.
Every step prints to the terminal so the run can be followed without
reading the code.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# In Docker, DATA_DIR=/app/data (set in the Dockerfile).
# Running locally, default to src/data relative to this file.
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
GENDER_SUB_PATH = DATA_DIR / "gender_submission.csv"
OUTPUT_PATH = DATA_DIR / "predictions_python.csv"

COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_files():
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.")
            print("Download the data from Kaggle and place it in src/data/ (see README).")
            sys.exit(1)


def extract_title(names):
    titles = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def learn_fill_values(train):
    """Imputation values are learned from the training set only."""
    return {
        "age_by_title": train.groupby("Title")["Age"].median(),
        "age_overall": train["Age"].median(),
        "fare": train["Fare"].median(),
        "embarked": train["Embarked"].mode()[0],
    }


def clean_and_engineer(df, fill, label):
    df = df.copy()
    print(f"\n[{label}] Missing values before cleaning:")
    print(df[["Age", "Fare", "Embarked", "Cabin"]].isna().sum().to_string())

    n_age = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill["age_by_title"]))
    df["Age"] = df["Age"].fillna(fill["age_overall"])
    print(f"[{label}] Filled {n_age} missing Age values with median age by Title")

    n_fare = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill["fare"])
    print(f"[{label}] Filled {n_fare} missing Fare values with training median ({fill['fare']:.2f})")

    n_emb = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill["embarked"])
    print(f"[{label}] Filled {n_emb} missing Embarked values with mode ('{fill['embarked']}')")

    df["Sex"] = (df["Sex"] == "male").astype(int)
    print(f"[{label}] Encoded Sex as binary (male=1, female=0)")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[{label}] Added FamilySize (SibSp + Parch + 1) and IsAlone")

    pct_missing = df["Cabin"].isna().mean()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"[{label}] Added HasCabin flag (Cabin is {pct_missing:.0%} missing, too sparse to use directly)")

    df = pd.get_dummies(df, columns=["Embarked", "Title", "Pclass"], drop_first=True, dtype=int)
    print(f"[{label}] One-hot encoded Embarked, Title and Pclass")

    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    print(f"[{label}] Dropped Name, Ticket and Cabin")
    return df


def main():
    check_files()

    # load training data
    section("STEP 1: Loading training data")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Loaded {TRAIN_PATH.name}: {train.shape[0]} rows, {train.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(train.head().to_string())

    # explore
    section("STEP 2: Exploring training data")
    print(f"Overall survival rate: {train['Survived'].mean():.3f}")
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())

    # ---- Step 14: adjust --------------------------------------------------
    section("STEP 3: Cleaning and feature engineering")
    train["Title"] = extract_title(train["Name"])
    print("Extracted Title from Name. Counts:")
    print(train["Title"].value_counts().to_string())
    fill = learn_fill_values(train)
    train_clean = clean_and_engineer(train, fill, "train")

    features = [c for c in train_clean.columns if c not in ("PassengerId", "Survived")]
    print(f"\nFinal feature set ({len(features)}): {features}")
    X = train_clean[features]
    y = train_clean["Survived"]

    # ---- Step 15-16: model + training accuracy ---------------------------
    section("STEP 4: Training logistic regression")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    holdout_model = LogisticRegression(max_iter=1000)
    holdout_model.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, holdout_model.predict(X_val))
    print(f"Hold-out validation accuracy (80/20 split): {val_acc:.4f}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Refit on full training set. Training accuracy: {train_acc:.4f}")

    print("\nModel coefficients (positive = higher survival odds):")
    coefs = pd.Series(model.coef_[0], index=features).sort_values()
    print(coefs.round(3).to_string())

    # ---- Step 17: test predictions ---------------------------------------
    section("STEP 5: Predicting on the test set")
    test = pd.read_csv(TEST_PATH)
    print(f"Loaded {TEST_PATH.name}: {test.shape[0]} rows, {test.shape[1]} columns")
    test["Title"] = extract_title(test["Name"])
    test_clean = clean_and_engineer(test, fill, "test")
    # Make sure test has exactly the training columns (a category could be missing)
    X_test = test_clean.reindex(columns=features, fill_value=0)
    test_pred = model.predict(X_test)
    print(f"\nPredicted survivors: {test_pred.sum()} of {len(test_pred)} ({test_pred.mean():.3f})")

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH.name} in the data folder")

    # ---- Step 18: test accuracy ------------------------------------------
    section("STEP 6: Test set accuracy")
    print("Note: Kaggle's test.csv does not include the Survived column,")
    print("so true test accuracy can only be obtained by submitting to Kaggle.")
    if GENDER_SUB_PATH.exists():
        ref = pd.read_csv(GENDER_SUB_PATH).set_index("PassengerId").loc[test["PassengerId"], "Survived"]
        agree = accuracy_score(ref.values, test_pred)
        print(f"Agreement with gender_submission.csv (Kaggle's 'women survive' baseline): {agree:.4f}")
    else:
        print("gender_submission.csv not found; skipping baseline comparison.")
    print(f"Best offline estimate of test accuracy (hold-out validation): {val_acc:.4f}")

    section("DONE")


if __name__ == "__main__":
    main()
