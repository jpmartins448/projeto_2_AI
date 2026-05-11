from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_FILE = "churn_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
PLOTS_DIR = Path("plots")


def load_artifacts():
    """Load the trained model and feature names from disk."""
    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)
    return model, feature_names


def build_input_form():
    """Render input widgets and return a dictionary of user inputs."""
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        account_weeks = st.number_input("AccountWeeks", min_value=0, value=128)
        contract_renewal = st.selectbox("ContractRenewal", ["Yes", "No"], index=0)
        data_plan = st.selectbox("DataPlan", ["Yes", "No"], index=0)
        data_usage = st.number_input("DataUsage", min_value=0.0, value=2.7, step=0.1)
        cust_serv_calls = st.number_input("CustServCalls", min_value=0, value=1)

    with col2:
        day_mins = st.number_input("DayMins", min_value=0.0, value=265.1, step=0.1)
        day_calls = st.number_input("DayCalls", min_value=0, value=110)
        monthly_charge = st.number_input("MonthlyCharge", min_value=0.0, value=89.0, step=0.1)
        overage_fee = st.number_input("OverageFee", min_value=0.0, value=9.87, step=0.01)
        roam_mins = st.number_input("RoamMins", min_value=0.0, value=10.0, step=0.1)

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
    if probability <= 70:
        return "Medium Risk", "Monitor customer and consider a retention offer."
    return "High Risk", "Prioritise this customer for proactive retention."


def prediction_section(model, feature_names):
    """Render prediction controls and results."""
    input_data = build_input_form()

    if st.button("Predict Churn"):
        # Ensure column order matches the trained model
        input_df = pd.DataFrame([input_data], columns=feature_names)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1]) * 100

        if prediction == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is not likely to churn")

        st.metric("Churn Probability", f"{probability:.2f}%")

        risk_level, recommendation = interpret_risk(probability)
        st.info(f"Risk Level: {risk_level}")
        st.write(f"Recommendation: {recommendation}")


def model_insights_section():
    """Display available plots from the training pipeline."""
    st.subheader("Model Insights")

    plot_files = [
        ("Churn Distribution", PLOTS_DIR / "churn_distribution.png"),
        ("F1 Comparison", PLOTS_DIR / "model_f1_comparison.png"),
        ("Confusion Matrix", PLOTS_DIR / "best_model_confusion_matrix.png"),
        ("Feature Importance", PLOTS_DIR / "feature_importance.png"),
    ]

    for title, path in plot_files:
        if path.exists():
            with st.expander(title, expanded=False):
                st.image(str(path), use_container_width=True)
        else:
            st.caption(f"{title} plot not found at {path}.")


def main():
    """Main app entry point."""
    st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

    st.title("Customer Churn Prediction Dashboard")
    st.write(
        "This app predicts whether a telecom customer is likely to cancel their subscription."
    )

    if not Path(MODEL_FILE).exists() or not Path(FEATURE_NAMES_FILE).exists():
        st.error("Model files not found. Please train the model before using the app.")
        st.stop()

    model, feature_names = load_artifacts()

    tabs = st.tabs(["Prediction", "Model Insights"])

    with tabs[0]:
        prediction_section(model, feature_names)

    with tabs[1]:
        model_insights_section()


if __name__ == "__main__":
    main()
