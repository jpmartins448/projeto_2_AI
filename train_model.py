from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_FILE_PRIMARY = "customer_churn.csv"
DATA_FILE_FALLBACK = "telecom_churn.csv"
MODEL_FILE = "churn_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
RANDOM_STATE = 42
PLOTS_DIR = Path("plots")


def load_dataset() -> pd.DataFrame:
    """Load the dataset from CSV with a fallback to the provided telecom file."""
    
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


def evaluate_model(model_name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate the trained pipeline and return key metrics."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    return metrics


def train_and_evaluate(df: pd.DataFrame):
    """Train multiple models and return the best pipeline and metrics."""
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

    preprocessor = build_preprocessing_pipeline(feature_names)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained_pipelines[name] = pipeline

    results_table = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
            }
            for r in results
        ]
    ).sort_values(by="F1", ascending=False)

    best_model_name = results_table.iloc[0]["Model"]
    best_metrics = next(r for r in results if r["model"] == best_model_name)
    best_pipeline = trained_pipelines[best_model_name]

    return best_pipeline, best_metrics, results_table, feature_names


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
    ax.bar(results_table["Model"], results_table["Accuracy"], color="#4C78A8")

    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    add_bar_labels(ax, results_table["Accuracy"].tolist())

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def plot_model_f1(results_table: pd.DataFrame) -> None:
    """Plot model F1-score comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(results_table["Model"], results_table["F1"], color="#F58518")

    ax.set_title("Model F1-Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    add_bar_labels(ax, results_table["F1"].tolist())

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

    best_pipeline, best_metrics, results_table, feature_names = train_and_evaluate(df)

    print("\nModel comparison (sorted by F1):\n")
    print(results_table.to_string(index=False))

    print("\nBest model:", best_metrics["model"])
    print("Accuracy:", best_metrics["accuracy"])
    print("Precision:", best_metrics["precision"])
    print("Recall:", best_metrics["recall"])
    print("F1-score:", best_metrics["f1"])
    print("Confusion Matrix:\n", best_metrics["confusion_matrix"])
    print("\nClassification Report:\n", best_metrics["classification_report"])

    generate_plots(df, results_table, best_metrics, best_pipeline, feature_names)
    print("\nAll plots generated successfully in the plots folder.")

    save_artifacts(best_pipeline, feature_names)
    print(f"\nSaved model to '{MODEL_FILE}' and features to '{FEATURE_NAMES_FILE}'.")


if __name__ == "__main__":
    main()
