from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
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


MODEL_FILE = "churn_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
PLOTS_DIR = Path("plots")
DATASET_FILE = "telecom_churn.csv"

# Friendly labels for the UI while keeping model feature names unchanged
FEATURE_LABELS = {
    "AccountWeeks": "Account duration in weeks",
    "ContractRenewal": "Contract renewed?",
    "DataPlan": "Has data plan?",
    "DataUsage": "Data usage",
    "CustServCalls": "Customer service calls",
    "DayMins": "Daily call minutes",
    "DayCalls": "Daily calls",
    "MonthlyCharge": "Monthly charge",
    "OverageFee": "Overage fee",
    "RoamMins": "Roaming minutes",
}


def load_artifacts():
    """Load the trained model and feature names from disk."""
    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    return model, feature_names


@st.cache_data(show_spinner=False)
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the churn dataset with Streamlit caching."""
    return pd.read_csv(file_path)


def apply_example(example: dict) -> None:
    """Populate Streamlit session state with example values."""
    for key, value in example.items():
        st.session_state[key] = value


def build_input_form():
    """Render input widgets and return a dictionary of user inputs."""
    st.subheader("Customer Information")

    # Demo presets for quick testing
    example_low = {
        "AccountWeeks": 90,
        "ContractRenewal": 0,
        "DataPlan": 1,
        "DataUsage": 2.1,
        "CustServCalls": 2,
        "DayMins": 250.0,
        "DayCalls": 100,
        "MonthlyCharge": 85.0,
        "OverageFee": 9.0,
        "RoamMins": 9.5,
    }
    example_medium = {
        "AccountWeeks": 140,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 3.2,
        "CustServCalls": 0,
        "DayMins": 180.0,
        "DayCalls": 120,
        "MonthlyCharge": 70.0,
        "OverageFee": 3.5,
        "RoamMins": 5.0,
    }
    example_high = {
        "AccountWeeks": 10,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 0.2,
        "CustServCalls": 5,
        "DayMins": 10.0,
        "DayCalls": 5,
        "MonthlyCharge": 250.0,
        "OverageFee": 30.0,
        "RoamMins": 5.0,
    }

    demo_col1, demo_col2, demo_col3 = st.columns(3)
    with demo_col1:
        if st.button("Low-risk customer"):
            apply_example(example_low)
    with demo_col2:
        if st.button("Medium-risk customer"):
            apply_example(example_medium)
    with demo_col3:
        if st.button("High-risk customer"):
            apply_example(example_high)

    # Ensure default values exist in session state
    defaults = {
        "AccountWeeks": 128,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 2.7,
        "CustServCalls": 1,
        "DayMins": 265.1,
        "DayCalls": 110,
        "MonthlyCharge": 89.0,
        "OverageFee": 9.87,
        "RoamMins": 10.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    col1, col2 = st.columns(2)

    with col1:
        account_weeks = st.number_input(
            FEATURE_LABELS["AccountWeeks"],
            min_value=0,
            value=st.session_state["AccountWeeks"],
            key="AccountWeeks",
        )
        contract_renewal = st.selectbox(
            FEATURE_LABELS["ContractRenewal"],
            ["Yes", "No"],
            index=0 if st.session_state["ContractRenewal"] == 1 else 1,
            key="ContractRenewalSelect",
        )
        data_plan = st.selectbox(
            FEATURE_LABELS["DataPlan"],
            ["Yes", "No"],
            index=0 if st.session_state["DataPlan"] == 1 else 1,
            key="DataPlanSelect",
        )
        data_usage = st.number_input(
            FEATURE_LABELS["DataUsage"],
            min_value=0.0,
            value=st.session_state["DataUsage"],
            step=0.1,
            key="DataUsage",
        )
        cust_serv_calls = st.number_input(
            FEATURE_LABELS["CustServCalls"],
            min_value=0,
            value=st.session_state["CustServCalls"],
            key="CustServCalls",
        )

    with col2:
        day_mins = st.number_input(
            FEATURE_LABELS["DayMins"],
            min_value=0.0,
            value=st.session_state["DayMins"],
            step=0.1,
            key="DayMins",
        )
        day_calls = st.number_input(
            FEATURE_LABELS["DayCalls"],
            min_value=0,
            value=st.session_state["DayCalls"],
            key="DayCalls",
        )
        monthly_charge = st.number_input(
            FEATURE_LABELS["MonthlyCharge"],
            min_value=0.0,
            value=st.session_state["MonthlyCharge"],
            step=0.1,
            key="MonthlyCharge",
        )
        overage_fee = st.number_input(
            FEATURE_LABELS["OverageFee"],
            min_value=0.0,
            value=st.session_state["OverageFee"],
            step=0.01,
            key="OverageFee",
        )
        roam_mins = st.number_input(
            FEATURE_LABELS["RoamMins"],
            min_value=0.0,
            value=st.session_state["RoamMins"],
            step=0.1,
            key="RoamMins",
        )

    # Map categorical inputs to numeric values
    contract_renewal_value = 1 if contract_renewal == "Yes" else 0
    data_plan_value = 1 if data_plan == "Yes" else 0

    return {
        "AccountWeeks": account_weeks,
        "ContractRenewal": contract_renewal_value,
        "DataPlan": data_plan_value,
        "DataUsage": data_usage,
        "CustServCalls": cust_serv_calls,
        "DayMins": day_mins,
        "DayCalls": day_calls,
        "MonthlyCharge": monthly_charge,
        "OverageFee": overage_fee,
        "RoamMins": roam_mins,
    }


def interpret_risk(probability: float):
    """Return risk level and recommendation based on churn probability."""
    if probability < 30:
        return "Low Risk", "No immediate action required."
    if probability <= 60:
        return "Medium Risk", "Monitor customer and consider a retention offer."
    return "High Risk", "Prioritise this customer for proactive retention."


def prediction_section(model, feature_names):
    """Render prediction controls and results."""
    input_data = build_input_form()

    if st.button("Predict Churn"):
        # Ensure column order matches the trained model
        input_df = pd.DataFrame([input_data], columns=feature_names)

        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1]) * 100

        if prediction == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is not likely to churn")

        if probability is not None:
            st.metric("Churn Probability", f"{probability:.2f}%")
            st.progress(min(max(probability / 100, 0), 1))

            risk_level, recommendation = interpret_risk(probability)
            if risk_level == "Low Risk":
                st.success(f"Risk Level: {risk_level}")
            elif risk_level == "Medium Risk":
                st.warning(f"Risk Level: {risk_level}")
            else:
                st.error(f"Risk Level: {risk_level}")

            st.info(f"Recommendation: {recommendation}")
        else:
            st.warning("This model does not provide probabilities.")

    with st.expander("Why this prediction?"):
        st.write(
            "The model uses customer behaviour and billing indicators such as customer "
            "service calls, contract renewal status, monthly charge, daily call minutes "
            "and overage fees to estimate churn risk."
        )

        feature_importance_path = PLOTS_DIR / "feature_importance.png"
        if feature_importance_path.exists():
            st.image(str(feature_importance_path), use_container_width=True)


def model_insights_section():
    """Display available plots from the training pipeline."""
    st.subheader("Model Insights")

    plot_files = [
        (
            "Churn Distribution",
            PLOTS_DIR / "churn_distribution.png",
            "Shows the balance between churned and non-churned customers.",
        ),
        (
            "F1-Score Comparison",
            PLOTS_DIR / "model_f1_comparison.png",
            "Compares model performance with an emphasis on class imbalance.",
        ),
        (
            "Confusion Matrix",
            PLOTS_DIR / "best_model_confusion_matrix.png",
            "Summarises correct and incorrect predictions for each class.",
        ),
        (
            "Feature Importance",
            PLOTS_DIR / "feature_importance.png",
            "Highlights which variables influenced the model the most.",
        ),
    ]

    for title, path, explanation in plot_files:
        if path.exists():
            with st.expander(title, expanded=False):
                st.image(str(path), use_container_width=True)
                st.caption(explanation)
        else:
            st.caption(f"{title} plot not found at {path}.")


def plot_metric_bars(results_df: pd.DataFrame, metric: str, title: str):
    """Create a bar chart for a given metric using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["Model"], results_df[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)

    for idx, value in enumerate(results_df[metric]):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrix(matrix, title: str):
    """Plot a confusion matrix using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def training_playground_section():
    """Interactive model training playground."""
    st.subheader("Model Training Playground")

    if not Path(DATASET_FILE).exists():
        st.error(f"Dataset not found: {DATASET_FILE}")
        return

    df = load_dataset(DATASET_FILE)

    # User controls for training
    test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    use_scaler = st.checkbox("Use StandardScaler", value=True)

    selected_models = st.multiselect(
        "Select models to train",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
    )

    st.markdown("**Logistic Regression settings**")
    lr_max_iter = st.number_input("max_iter", value=1000, step=100)

    st.markdown("**Decision Tree settings**")
    dt_max_depth = st.selectbox("max_depth", ["None"] + list(range(1, 21)), index=0)
    dt_min_samples_split = st.slider("min_samples_split", 2, 20, value=2)

    st.markdown("**Random Forest settings**")
    rf_n_estimators = st.slider("n_estimators", 10, 300, value=100, step=10)
    rf_max_depth = st.selectbox("rf_max_depth", ["None"] + list(range(1, 21)), index=0)
    rf_min_samples_split = st.slider("rf_min_samples_split", 2, 20, value=2)

    st.caption(
        "F1-score is useful for churn prediction because the dataset may be imbalanced, "
        "meaning there are usually fewer churned customers than non-churned customers."
    )

    if st.button("Train selected models"):
        if not selected_models:
            st.warning("Please select at least one model to train.")
            return

        feature_names = [col for col in df.columns if col != "Churn"]
        X = df[feature_names]
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )

        results = []
        trained_models = {}

        for model_name in selected_models:
            steps = []
            if use_scaler:
                steps.append(("scaler", StandardScaler()))

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=int(lr_max_iter), random_state=int(random_state))
            elif model_name == "Decision Tree":
                max_depth = None if dt_max_depth == "None" else int(dt_max_depth)
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=dt_min_samples_split,
                    random_state=int(random_state),
                )
            else:
                max_depth = None if rf_max_depth == "None" else int(rf_max_depth)
                model = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=max_depth,
                    min_samples_split=rf_min_samples_split,
                    random_state=int(random_state),
                )

            steps.append(("model", model))
            pipeline = Pipeline(steps=steps)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ConfusionMatrix": confusion_matrix(y_test, y_pred),
                "ClassificationReport": classification_report(y_test, y_pred, zero_division=0),
            }

            results.append(metrics)
            trained_models[model_name] = pipeline

        results_df = pd.DataFrame(
            [
                {
                    "Model": r["Model"],
                    "Accuracy": r["Accuracy"],
                    "Precision": r["Precision"],
                    "Recall": r["Recall"],
                    "F1": r["F1"],
                }
                for r in results
            ]
        ).sort_values(by="F1", ascending=False)

        st.dataframe(results_df, use_container_width=True)

        best_model_name = results_df.iloc[0]["Model"]
        best_metrics = next(r for r in results if r["Model"] == best_model_name)
        best_pipeline = trained_models[best_model_name]

        st.session_state["best_trained_model"] = best_pipeline
        st.session_state["best_feature_names"] = feature_names
        st.session_state["best_metrics"] = best_metrics
        st.session_state["results_df"] = results_df

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        best_metrics = st.session_state["best_metrics"]

        st.markdown("### Model Comparison")
        plot_metric_bars(results_df, "Accuracy", "Model Accuracy Comparison")
        plot_metric_bars(results_df, "F1", "Model F1-Score Comparison")

        st.markdown("### Best Model Summary")
        st.success(f"Best model: {best_metrics['Model']}")
        st.write(
            {
                "Accuracy": best_metrics["Accuracy"],
                "Precision": best_metrics["Precision"],
                "Recall": best_metrics["Recall"],
                "F1": best_metrics["F1"],
            }
        )

        plot_confusion_matrix(best_metrics["ConfusionMatrix"], "Best Model Confusion Matrix")

        st.markdown("### Classification Report")
        st.text(best_metrics["ClassificationReport"])

        if st.button("Save this model as production model"):
            st.warning(
                "This will replace the model used in the Prediction tab."
            )
            joblib.dump(st.session_state["best_trained_model"], MODEL_FILE)
            joblib.dump(st.session_state["best_feature_names"], FEATURE_NAMES_FILE)
            st.success("Production model saved successfully.")


def main():
    """Main app entry point."""
    st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

    st.title("Customer Churn Prediction Dashboard")
    st.write(
        "This app predicts whether a telecom customer is likely to cancel their subscription."
    )
    st.info(
        "This proof of concept helps telecom companies identify customers at risk of "
        "cancellation, allowing retention teams to act before losing revenue."
    )

    if not Path(MODEL_FILE).exists() or not Path(FEATURE_NAMES_FILE).exists():
        st.error("Model files not found. Please train the model before using the app.")
        st.stop()

    model, feature_names = load_artifacts()

    tabs = st.tabs(["Prediction", "Model Insights", "Model Training Playground"])

    with tabs[0]:
        prediction_section(model, feature_names)

    with tabs[1]:
        model_insights_section()

    with tabs[2]:
        training_playground_section()

    st.markdown("---")
    st.caption(
        "This model was trained on public or artificial data and is intended as a proof "
        "of concept. A production system would require real customer data, privacy "
        "validation, monitoring and regular retraining."
    )


if __name__ == "__main__":
    main()
