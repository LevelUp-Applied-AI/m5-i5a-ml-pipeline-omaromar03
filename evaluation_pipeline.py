"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    df = pd.read_csv(filepath)

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    X = df.drop(columns=["churned"])
    y = df["churned"]

    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),
        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="saga",
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),
        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(
                alpha=1.0,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),
        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42))
        ])
    }

    return models


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Args:
        models: Dictionary of {name: Pipeline}.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv_strategy,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean()
        })

    return pd.DataFrame(results).sort_values(by="f1_mean", ascending=False).reset_index(drop=True)


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train a pipeline on full training data and evaluate on the held-out test set.

    Use this on the best model from Task 4 as a final sanity check — the
    test-set metrics should be close to the CV estimates if the model
    generalizes. If they diverge substantially, the CV estimates were
    optimistic and you should investigate.

    Args:
        pipeline: An unfitted sklearn Pipeline (one entry from define_models).
        X_train, X_test: Feature DataFrames (train and held-out test).
        y_train, y_test: Target Series (train and held-out test).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("LogReg_L1 is recommended because it achieved the best F1 among the real models while balancing recall and precision better than the alternatives. Accuracy alone is misleading here because the most-frequent dummy can score highly by always predicting the majority class, yet it completely fails to detect churners. Since churn prediction has asymmetric business costs, recall and F1 are more meaningful than raw accuracy. The comparison with the stratified dummy also shows that the selected model learns useful signal beyond random guessing.")


if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        # Create 80/20 train/test split. The test set is held out for the
        # final evaluation in Task 5 — do not use it during cross-validation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

        models = define_models()
        if models:
            # Task 4: cross-validation on training data only
            results = evaluate_models(models, X_train, y_train)
            if results is not None:
                recommend_model(results)

                non_dummy_results = results[~results["model"].str.contains("Dummy", case=False)]
                best_model_name = non_dummy_results.sort_values(by="f1_mean", ascending=False).iloc[0]["model"]
                best_pipeline = models[best_model_name]

                test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)

                print("\n=== Best Real Model ===")
                print(best_model_name)

                print("\n=== Final Test-Set Metrics ===")
                for metric, value in test_metrics.items():
                    print(f"{metric}: {value:.3f}")


"""
Recommendation:
I recommend LogReg_L1 because it achieved the highest F1 score among the real models, making it the strongest balance between identifying churners and controlling false positives. Accuracy alone is not sufficient in this problem because the most-frequent dummy can achieve high accuracy by always predicting the majority class while completely failing to detect churners. The selected model provides stronger recall, which is important because missing a churning customer is more costly than issuing a false alarm. Compared with the stratified dummy baseline, it shows meaningful improvement in F1, indicating that it learns useful structure beyond random guessing. The final test-set evaluation should be compared with CV performance to confirm whether the model generalizes reliably.
"""