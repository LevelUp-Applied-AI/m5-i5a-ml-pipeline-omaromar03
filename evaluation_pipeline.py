"""
Module 5 Week A — Integration Task: ML Evaluation Pipeline

Build a reproducible ML evaluation pipeline for the Petra Telecom churn dataset.

This script:
1. Loads and splits the data
2. Builds preprocessing with ColumnTransformer
3. Defines 5 model configurations
4. Runs 5-fold stratified cross-validation
5. Selects the best real model based on mean F1
6. Evaluates the selected model on the held-out test set
7. Prints a recommendation grounded in business context

Run:
    python evaluation_pipeline.py
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "churned"


def load_data(filepath: str = "data/telecom_churn.csv") -> pd.DataFrame:
    """
    Load dataset from CSV.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    df = pd.read_csv(filepath)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature columns.

    Args:
        X: Feature DataFrame.

    Returns:
        (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Split data into train/test and identify feature groups.

    Args:
        df: Input DataFrame containing features and target.

    Returns:
        X_train, X_test, y_train, y_test, numeric_features, categorical_features
    """
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    # Make sure target is numeric/binary if possible
    if y.dtype == "object":
        y = y.astype(str).str.strip().str.lower()
        mapping_candidates = {
            "yes": 1,
            "true": 1,
            "1": 1,
            "churned": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "not churned": 0,
        }
        unique_values = set(y.unique())
        if unique_values.issubset(set(mapping_candidates.keys())):
            y = y.map(mapping_candidates)

    if y.isnull().any():
        raise ValueError("Target column contains null values after preprocessing.")

    numeric_features, categorical_features = identify_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build preprocessing transformer for numeric and categorical columns.

    Args:
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.

    Returns:
        Configured ColumnTransformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def build_pipeline(
    model,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build full preprocessing + model pipeline.

    Args:
        model: Scikit-learn estimator.
        numeric_features: Numeric columns.
        categorical_features: Categorical columns.

    Returns:
        Pipeline with preprocessor and classifier.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def define_models(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Pipeline]:
    """
    Define the five required model configurations.

    Args:
        numeric_features: Numeric columns.
        categorical_features: Categorical columns.

    Returns:
        Dictionary mapping model names to full pipelines.
    """
    models = {
        "LogReg (default)": build_pipeline(
            LogisticRegression(
                C=1.0,
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            numeric_features,
            categorical_features,
        ),
        "LogReg (L1, C=0.1)": build_pipeline(
            LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="saga",
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            numeric_features,
            categorical_features,
        ),
        "RidgeClassifier": build_pipeline(
            RidgeClassifier(
                alpha=1.0,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            numeric_features,
            categorical_features,
        ),
        "Most-frequent Dummy": build_pipeline(
            DummyClassifier(strategy="most_frequent"),
            numeric_features,
            categorical_features,
        ),
        "Stratified Dummy": build_pipeline(
            DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
            numeric_features,
            categorical_features,
        ),
    }

    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Run stratified 5-fold cross-validation for all models.

    Args:
        models: Dictionary of model pipelines.
        X_train: Training features.
        y_train: Training target.

    Returns:
        Results DataFrame sorted by mean F1 descending.
    """
    scoring = ["accuracy", "precision", "recall", "f1"]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []

    for model_name, pipeline in models.items():
        cv_output = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {
            "Model": model_name,
            "Mean Accuracy": cv_output["test_accuracy"].mean(),
            "Accuracy Std": cv_output["test_accuracy"].std(),
            "Mean Precision": cv_output["test_precision"].mean(),
            "Precision Std": cv_output["test_precision"].std(),
            "Mean Recall": cv_output["test_recall"].mean(),
            "Recall Std": cv_output["test_recall"].std(),
            "Mean F1": cv_output["test_f1"].mean(),
            "F1 Std": cv_output["test_f1"].std(),
        }
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        by="Mean F1", ascending=False
    ).reset_index(drop=True)

    return results_df


def final_evaluation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Fit pipeline on full training data and evaluate on held-out test set.

    Args:
        pipeline: Selected model pipeline.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        Dictionary with accuracy, precision, recall, and f1.
    """
    model = clone(pipeline)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics


def get_best_real_model(results_df: pd.DataFrame) -> str:
    """
    Select the best non-dummy model based on highest mean F1.

    Args:
        results_df: Cross-validation results table.

    Returns:
        Best real model name.
    """
    real_models = results_df[
        ~results_df["Model"].str.contains("Dummy", case=False, na=False)
    ].copy()

    if real_models.empty:
        raise ValueError("No real models found in results table.")

    best_model_name = real_models.sort_values(
        by="Mean F1", ascending=False
    ).iloc[0]["Model"]

    return best_model_name


def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric results for prettier console output.

    Args:
        results_df: Raw results DataFrame.

    Returns:
        Formatted DataFrame.
    """
    formatted = results_df.copy()

    numeric_cols = [
        "Mean Accuracy",
        "Accuracy Std",
        "Mean Precision",
        "Precision Std",
        "Mean Recall",
        "Recall Std",
        "Mean F1",
        "F1 Std",
    ]

    for col in numeric_cols:
        formatted[col] = formatted[col].map(lambda x: f"{x:.3f}")

    return formatted


def generate_recommendation(
    results_df: pd.DataFrame,
    best_model_name: str,
    test_metrics: Dict[str, float],
) -> str:
    """
    Generate the required business recommendation paragraph.

    Args:
        results_df: Cross-validation results.
        best_model_name: Selected best real model.
        test_metrics: Final test metrics.

    Returns:
        Recommendation paragraph as a string.
    """
    best_row = results_df.loc[results_df["Model"] == best_model_name].iloc[0]
    most_freq_row = results_df.loc[
        results_df["Model"] == "Most-frequent Dummy"
    ].iloc[0]
    strat_dummy_row = results_df.loc[
        results_df["Model"] == "Stratified Dummy"
    ].iloc[0]

    cv_f1 = best_row["Mean F1"]
    cv_f1_std = best_row["F1 Std"]
    cv_precision = best_row["Mean Precision"]
    cv_recall = best_row["Mean Recall"]

    test_f1 = test_metrics["f1"]
    f1_gap_vs_strat = cv_f1 - strat_dummy_row["Mean F1"]
    test_matches_cv = abs(test_f1 - cv_f1) <= cv_f1_std

    stability_sentence = (
        "The held-out test-set F1 is close to the cross-validation estimate, "
        "which suggests the model generalizes reasonably well to unseen customers."
        if test_matches_cv
        else
        "The held-out test-set F1 is noticeably lower than the cross-validation estimate, "
        "so the CV results may have been slightly optimistic and the model should be treated with caution."
    )

    recommendation = (
        f"I recommend **{best_model_name}** because it achieved the strongest mean F1 score "
        f"among the real models during cross-validation ({cv_f1:.3f}), making it the best balance "
        f"between identifying churners and limiting unnecessary false alarms. Accuracy alone is not "
        f"sufficient for this problem: the **Most-frequent Dummy** can achieve high accuracy "
        f"({most_freq_row['Mean Accuracy']:.3f}) simply by predicting the majority class, but that approach "
        f"largely fails to detect actual churners, which is costly in a churn-retention setting. "
        f"The recommended model shows a precision/recall trade-off of about {cv_precision:.3f} precision "
        f"and {cv_recall:.3f} recall, meaning it catches a meaningful share of churners while still making "
        f"some false-positive predictions. Compared with the **Stratified Dummy**, its F1 is higher by about "
        f"{f1_gap_vs_strat:.3f}, showing that it learns real signal beyond random guessing, although the margin "
        f"also suggests that linear models may still be limited by the current features. {stability_sentence}"
    )

    return recommendation


def main() -> None:
    """
    Main execution flow.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 80)
    print("ML EVALUATION PIPELINE")
    print("=" * 80)

    # Task 1
    df = load_data("data/telecom_churn.csv")
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = (
        prepare_data(df)
    )

    print("\nDataset loaded successfully.")
    print(f"Full shape: {df.shape}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Task 2 + Task 3
    models = define_models(numeric_features, categorical_features)

    # Task 4
    results_df = evaluate_models(models, X_train, y_train)
    formatted_results = format_results_table(results_df)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(formatted_results.to_string(index=False))

    # Task 5
    best_model_name = get_best_real_model(results_df)
    best_pipeline = models[best_model_name]
    test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 80)
    print("BEST REAL MODEL")
    print("=" * 80)
    print(best_model_name)

    print("\n" + "=" * 80)
    print("FINAL TEST-SET METRICS")
    print("=" * 80)
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name.capitalize():<10}: {metric_value:.3f}")

    # Task 6
    recommendation = generate_recommendation(results_df, best_model_name, test_metrics)

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(recommendation)


if __name__ == "__main__":
    main()


"""
Recommendation:

I recommend **LogReg (L1, C=0.1)** because it achieved the highest mean F1 score...
(rest of paragraph)
"""