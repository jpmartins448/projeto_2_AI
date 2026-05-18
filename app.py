from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_FILE = "churn_model.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
PLOTS_DIR = Path("plots")

FEATURE_LABELS = {
    "AccountWeeks": "Account Length (weeks)",
    "ContractRenewal": "Contract Renewal",
    "DataPlan": "Data Plan",
    "DataUsage": "Data Usage (GB)",
    "CustServCalls": "Customer Service Calls",
    "DayMins": "Day Minutes",
    "DayCalls": "Day Call Count",
    "MonthlyCharge": "Monthly Charge",
    "OverageFee": "Overage Fees",
    "RoamMins": "International Minutes",
}


def apply_styles() -> None:
    """Apply minimal styling for a clean layout."""
    st.markdown(
        """
        <style>
            .section {
                background: #f7f7f8;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 16px 18px;
                margin-bottom: 16px;
            }
            .risk-low { color: #15803d; font-weight: 600; }
            .risk-medium { color: #d97706; font-weight: 600; }
            .risk-high { color: #b91c1c; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_artifacts():
    """Load the trained model and feature names from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
    except FileNotFoundError:
        st.error("Model files not found. Please run python train_model.py first.")
        st.stop()
    except Exception:
        st.error("Unable to load the model files. Please re-run python train_model.py.")
        st.stop()
    return model, feature_names


def get_example_profile(selection: str):
    """Return example profiles for the form."""
    examples = {
        "Low Risk Example": {
            "AccountWeeks": 160,
            "ContractRenewal": 1,
            "DataPlan": 1,
            "DataUsage": 3.0,
            "CustServCalls": 0,
            "DayMins": 190.0,
            "DayCalls": 105,
            "MonthlyCharge": 72.0,
            "OverageFee": 2.0,
            "RoamMins": 4.0,
        },
        "Medium Risk Example": {
            "AccountWeeks": 85,
            "ContractRenewal": 0,
            "DataPlan": 1,
            "DataUsage": 1.6,
            "CustServCalls": 2,
            "DayMins": 235.0,
            "DayCalls": 95,
            "MonthlyCharge": 88.0,
            "OverageFee": 7.0,
            "RoamMins": 9.0,
        },
        "High Risk Example": {
            "AccountWeeks": 30,
            "ContractRenewal": 0,
            "DataPlan": 0,
            "DataUsage": 0.5,
            "CustServCalls": 4,
            "DayMins": 300.0,
            "DayCalls": 80,
            "MonthlyCharge": 102.0,
            "OverageFee": 14.0,
            "RoamMins": 14.0,
        },
    }
    return examples.get(selection)


def render_customer_form():
    """Render the customer details form and return inputs with submit state."""
    defaults = {
        "AccountWeeks": 120,
        "ContractRenewal": 1,
        "DataPlan": 1,
        "DataUsage": 2.4,
        "CustServCalls": 1,
        "DayMins": 220.0,
        "DayCalls": 100,
        "MonthlyCharge": 84.0,
        "OverageFee": 7.5,
        "RoamMins": 8.0,
        "ContractRenewalSelect": "Yes",
        "DataPlanSelect": "Yes",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Customer Details")
    selection = st.selectbox(
        "Example profile",
        ["Custom", "Low Risk Example", "Medium Risk Example", "High Risk Example"],
    )
    example = get_example_profile(selection)
    if example:
        for key, value in example.items():
            st.session_state[key] = value
        st.session_state["ContractRenewalSelect"] = "Yes" if example["ContractRenewal"] == 1 else "No"
        st.session_state["DataPlanSelect"] = "Yes" if example["DataPlan"] == 1 else "No"

    with st.form("customer_form"):
        with st.expander("Customer Account", expanded=True):
            account_weeks = st.number_input(
                FEATURE_LABELS["AccountWeeks"],
                min_value=0,
                value=st.session_state["AccountWeeks"],
                key="AccountWeeks",
            )
            contract_renewal = st.selectbox(
                FEATURE_LABELS["ContractRenewal"],
                ["Yes", "No"],
                index=0 if st.session_state["ContractRenewalSelect"] == "Yes" else 1,
                key="ContractRenewalSelect",
            )
            data_plan = st.selectbox(
                FEATURE_LABELS["DataPlan"],
                ["Yes", "No"],
                index=0 if st.session_state["DataPlanSelect"] == "Yes" else 1,
                key="DataPlanSelect",
            )
            data_usage = st.number_input(
                FEATURE_LABELS["DataUsage"],
                min_value=0.0,
                value=st.session_state["DataUsage"],
                step=0.1,
                key="DataUsage",
            )

        with st.expander("Usage", expanded=True):
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
            roam_mins = st.number_input(
                FEATURE_LABELS["RoamMins"],
                min_value=0.0,
                value=st.session_state["RoamMins"],
                step=0.1,
                key="RoamMins",
            )
            cust_serv_calls = st.number_input(
                FEATURE_LABELS["CustServCalls"],
                min_value=0,
                value=st.session_state["CustServCalls"],
                key="CustServCalls",
            )

        with st.expander("Charges", expanded=True):
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
                step=0.1,
                key="OverageFee",
            )

        submitted = st.form_submit_button("Assess Risk")

    st.markdown("</div>", unsafe_allow_html=True)

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
    }, submitted


def calculate_risk_level(probability: float) -> str:
    """Return risk category based on churn probability (0-1)."""
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"


def get_recommendation(risk_level: str) -> str:
    """Return recommendation text for a risk level."""
    if risk_level == "Low Risk":
        return "Maintain regular engagement and continue monitoring customer satisfaction."
    if risk_level == "Medium Risk":
        return "Consider proactive outreach and review recent service interactions."
    return "Prioritise retention intervention and consider a personalised retention offer."


def get_possible_factors(input_data: dict) -> list:
    """Return up to three indicative risk signals."""
    factors = []
    if input_data["CustServCalls"] >= 3:
        factors.append("Several customer service calls may indicate dissatisfaction.")
    if input_data["ContractRenewal"] == 0:
        factors.append("Customer has not renewed the contract.")
    if input_data["AccountWeeks"] < 52:
        factors.append("Short account length can increase churn sensitivity.")
    if input_data["DataPlan"] == 0:
        factors.append("No data plan may reduce perceived service value.")
    if input_data["RoamMins"] >= 12:
        factors.append("Higher international usage can drive billing friction.")
    return factors[:3]


def main() -> None:
    """Main app entry point."""
    st.set_page_config(
        page_title="Telecom Churn Risk Assessment",
        page_icon="📡",
        layout="wide",
    )
    apply_styles()

    st.title("Telecom Churn Risk Assessment")
    st.caption("Estimate customer churn risk and support retention decisions.")
    st.write(
        "This tool uses a trained machine learning model to estimate the probability that a customer may churn."
    )

    model, feature_names = load_artifacts()

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        input_data, submitted = render_customer_form()

    with right_col:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Risk Assessment")
        if submitted:
            input_df = pd.DataFrame([input_data], columns=feature_names)
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(input_df)[0][1])
            else:
                probability = float(model.predict(input_df)[0])

            risk_level = calculate_risk_level(probability)
            risk_class = {
                "Low Risk": "risk-low",
                "Medium Risk": "risk-medium",
                "High Risk": "risk-high",
            }[risk_level]

            st.markdown(f"**Churn Probability:** {probability * 100:.1f}%")
            st.markdown(
                f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(probability, 0.0), 1.0))
            st.markdown(f"**Recommendation:** {get_recommendation(risk_level)}")

            factors = get_possible_factors(input_data)
            st.markdown("#### Possible contributing factors")
            st.caption("Indicative signals only.")
            if factors:
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- No dominant signals detected from the current inputs.")
        else:
            st.write("Submit customer details to see risk results.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Model information", expanded=False):
        st.markdown(
            """
            - The model was trained offline using historical telecom customer data.
            - Cross-validation and grid search were used during training.
            - Final performance is evaluated in train_model.py.
            - This app is for decision support and should be used together with business judgement.
            """
        )

        plot_paths = [
            PLOTS_DIR / "model_f1_comparison.png",
            PLOTS_DIR / "feature_importance.png",
        ]
        plotted = False
        for path in plot_paths:
            if path.exists():
                st.image(str(path), use_container_width=True)
                plotted = True
        if not plotted:
            st.caption("Run python train_model.py to generate model insight charts.")


if __name__ == "__main__":
    main()
