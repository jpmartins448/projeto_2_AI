from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_FILE_PRIMARY = "customer_churn.csv"
DATA_FILE_FALLBACK = "telecom_churn.csv"
MODEL_FILE = "churn_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
RANDOM_STATE = 42
PLOTS_DIR = Path("plots")


def load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV with a fallback to the provided telecom file."""
    if Path(DATA_FILE_PRIMARY).exists():
        return pd.read_csv(DATA_FILE_PRIMARY)
    return pd.read_csv(DATA_FILE_FALLBACK)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print basic dataset inspection info for a quick sanity check."""
    print("First 5 rows:\n", df.head())
    print("\nShape:", df.shape)
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nClass distribution (Churn):\n", df["Churn"].value_counts())


def build_preprocessing_pipeline(feature_names):
    """Build preprocessing pipeline for numeric features."""
    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[("num", scaler, feature_names)],
        remainder="drop",
    )

    return preprocessor


def add_noise_to_training_data(
    X_train: pd.DataFrame,
    noise_level: float = 0.01,
    random_state: int = RANDOM_STATE,
    binary_columns=None,
) -> pd.DataFrame:
    """Add small Gaussian noise to continuous numeric features for robustness."""
    if binary_columns is None:
        binary_columns = []

    noisy_X = X_train.copy()
    rng = np.random.default_rng(random_state)

    numeric_cols = noisy_X.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col in binary_columns:
            continue
        if noisy_X[col].nunique(dropna=True) <= 2:
            continue

        col_std = noisy_X[col].std()
        if pd.isna(col_std) or col_std == 0:
            continue

        noise = rng.normal(0, noise_level * col_std, size=len(noisy_X))
        noisy_X[col] = noisy_X[col] + noise

        if X_train[col].min() >= 0:
            noisy_X[col] = noisy_X[col].clip(lower=0)

    return noisy_X


def evaluate_holdout(model_name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate the trained pipeline on the holdout test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Regression-style metrics on churn probabilities (requested for the project).
    mae = mean_absolute_error(y_test, y_proba)
    rmse = np.sqrt(mean_squared_error(y_test, y_proba))
    r2 = r2_score(y_test, y_proba)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    return metrics


def train_and_evaluate(df: pd.DataFrame):
    """Run cross-validated grid search, then evaluate on a holdout test set."""
    feature_names = [col for col in df.columns if col != "Churn"]
    X = df[feature_names]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    binary_columns = ["ContractRenewal", "DataPlan"]
    X_train_noisy = add_noise_to_training_data(
        X_train,
        noise_level=0.01,
        random_state=RANDOM_STATE,
        binary_columns=binary_columns,
    )
    print("Added Gaussian noise to training data (noise_level=0.01).")

    preprocessor = build_preprocessing_pipeline(feature_names)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    model_configs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, solver="liblinear"),
            {
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 3, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

    cv_results = []
    test_results = []
    trained_pipelines = {}

    for name, (model, param_grid) in model_configs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
        )
        grid.fit(X_train_noisy, y_train)

        best_index = grid.best_index_
        cv_metrics = {
            "Model": name,
            "Best Params": grid.best_params_,
            "CV Accuracy": grid.cv_results_["mean_test_accuracy"][best_index],
            "CV Precision": grid.cv_results_["mean_test_precision"][best_index],
            "CV Recall": grid.cv_results_["mean_test_recall"][best_index],
            "CV F1": grid.cv_results_["mean_test_f1"][best_index],
        }

        best_pipeline = grid.best_estimator_
        trained_pipelines[name] = best_pipeline

        holdout_metrics = evaluate_holdout(name, best_pipeline, X_test, y_test)

        cv_results.append(cv_metrics)
        test_results.append(holdout_metrics)

    cv_results_table = pd.DataFrame(cv_results).sort_values(by="CV F1", ascending=False)

    test_results_table = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Test Accuracy": r["accuracy"],
                "Test Precision": r["precision"],
                "Test Recall": r["recall"],
                "Test F1": r["f1"],
                "Test MAE": r["mae"],
                "Test RMSE": r["rmse"],
                "Test R2": r["r2"],
            }
            for r in test_results
        ]
    )

    results_table = cv_results_table.merge(test_results_table, on="Model", how="left")

    best_model_name = cv_results_table.iloc[0]["Model"]
    best_metrics = next(r for r in test_results if r["model"] == best_model_name)
    best_pipeline = trained_pipelines[best_model_name]

    return best_pipeline, best_metrics, cv_results_table, results_table, feature_names


def save_artifacts(pipeline: Pipeline, feature_names):
    """Save the trained pipeline and feature names to disk."""
    joblib.dump(pipeline, MODEL_FILE)
    joblib.dump(feature_names, FEATURE_NAMES_FILE)


def ensure_plots_folder() -> None:
    """Create the plots folder if it does not exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def add_bar_labels(ax, values) -> None:
    """Add numeric labels above bar charts for readability."""
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Plot the churn distribution as a labeled bar chart."""
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Non-churned (0)", "Churned (1)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518"])

    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "churn_distribution.png", dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap using matplotlib imshow only."""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(corr.values, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature Correlation Heatmap")

    fig.colorbar(heatmap, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close(fig)


def plot_feature_histograms(df: pd.DataFrame) -> None:
    """Plot histograms for selected numeric features."""
    features = [
        "AccountWeeks",
        "DataUsage",
        "CustServCalls",
        "DayMins",
        "MonthlyCharge",
        "OverageFee",
    ]

    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[feature], bins=20, color="#4C78A8", edgecolor="black", alpha=0.8)

        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"hist_{feature.lower()}.png", dpi=300)
        plt.close(fig)


def plot_model_accuracy(results_table: pd.DataFrame) -> None:
    """Plot model accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(results_table["Model"], results_table["CV Accuracy"], color="#4C78A8")

    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    add_bar_labels(ax, results_table["CV Accuracy"].tolist())

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def plot_model_f1(results_table: pd.DataFrame) -> None:
    """Plot model F1-score comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(results_table["Model"], results_table["CV F1"], color="#F58518")

    ax.set_title("Model F1-Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    add_bar_labels(ax, results_table["CV F1"].tolist())

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_f1_comparison.png", dpi=300)
    plt.close(fig)


def plot_best_confusion_matrix(best_metrics: dict) -> None:
    """Plot confusion matrix for the best model."""
    matrix = best_metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Best Model Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "best_model_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_feature_importance(best_pipeline: Pipeline, feature_names) -> None:
    """Plot feature importance if the best model is Random Forest."""
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(sorted_features, sorted_importances, color="#72B7B2")
    ax.invert_yaxis()

    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=300)
    plt.close(fig)


def plot_regression_style_metrics(best_metrics: dict) -> None:
    """Plot regression-style metrics computed on churn probabilities."""
    labels = ["MAE", "RMSE", "R2"]
    values = [best_metrics["mae"], best_metrics["rmse"], best_metrics["r2"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Regression-style Metrics on Churn Probabilities")
    ax.set_ylabel("Score")

    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_style_metrics.png", dpi=300)
    plt.close(fig)


def generate_plots(
    df: pd.DataFrame,
    results_table: pd.DataFrame,
    best_metrics: dict,
    best_pipeline: Pipeline,
    feature_names,
) -> None:
    """Generate and save all required plots for the project."""
    ensure_plots_folder()

    plot_churn_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_histograms(df)
    plot_model_accuracy(results_table)
    plot_model_f1(results_table)
    plot_best_confusion_matrix(best_metrics)
    plot_feature_importance(best_pipeline, feature_names)
    plot_regression_style_metrics(best_metrics)


def predict_customer_churn(customer_data: dict):
    """Predict churn class and probability for a single customer.

    Returns:
        prediction (int): 0 or 1
        probability (float): churn probability in percentage
    """
    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found. Train the model first."
        )

    if not Path(FEATURE_NAMES_FILE).exists():
        raise FileNotFoundError(
            f"Feature names file '{FEATURE_NAMES_FILE}' not found. Train the model first."
        )

    pipeline = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)

    input_df = pd.DataFrame([customer_data], columns=feature_names)

    prediction = int(pipeline.predict(input_df)[0])

    if hasattr(pipeline, "predict_proba"):
        churn_prob = float(pipeline.predict_proba(input_df)[0][1]) * 100
    else:
        churn_prob = 0.0

    return prediction, churn_prob


def main():
    """Run the full training workflow and save the best model."""
    df = load_dataset()
    inspect_dataset(df)

    best_pipeline, best_metrics, cv_results_table, results_table, feature_names = train_and_evaluate(df)

    print("\nCross-validation summary (sorted by CV F1):\n")
    print(cv_results_table.to_string(index=False))

    print("\nHoldout test results (per model):\n")
    print(
        results_table[
            [
                "Model",
                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Test F1",
                "Test MAE",
                "Test RMSE",
                "Test R2",
            ]
        ].to_string(index=False)
    )

    print("\nBest model (by CV F1):", best_metrics["model"])
    print("Holdout Accuracy:", best_metrics["accuracy"])
    print("Holdout Precision:", best_metrics["precision"])
    print("Holdout Recall:", best_metrics["recall"])
    print("Holdout F1-score:", best_metrics["f1"])
    print("Holdout Confusion Matrix:\n", best_metrics["confusion_matrix"])
    print("\nClassification Report:\n", best_metrics["classification_report"])
    print(
        "\nNote: MAE, RMSE and R2 are regression-style metrics computed on churn probabilities."
    )
    print("Holdout MAE:", best_metrics["mae"])
    print("Holdout RMSE:", best_metrics["rmse"])
    print("Holdout R2:", best_metrics["r2"])

    ensure_plots_folder()
    results_table.to_csv(PLOTS_DIR / "cross_validation_results.csv", index=False)
    generate_plots(df, results_table, best_metrics, best_pipeline, feature_names)
    print("\nAll plots generated successfully in the plots folder.")
    print("Cross-validation results saved to plots/cross_validation_results.csv.")

    save_artifacts(best_pipeline, feature_names)
    print(f"\nSaved model to '{MODEL_FILE}' and features to '{FEATURE_NAMES_FILE}'.")


if __name__ == "__main__":
    main()
