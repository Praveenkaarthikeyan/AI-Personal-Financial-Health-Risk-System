# ============================================================
# app.py
# Purpose: The main Streamlit web application - replaces the
#          console-based predict.py with a visual, multi-page
#          dashboard. Reuses every backend module we built
#          (database, feature_engineering, prediction, etc.)
#          without duplicating any logic.
#
# Run with:  streamlit run app.py
# (NOT python app.py - Streamlit apps need the streamlit runner)
# ============================================================

import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go

# Add src/ to Python's import path so we can import our modules
# directly (database, feature_engineering, prediction, etc.)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import database as db
from feature_engineering import compute_financial_features, compute_financial_health_score
from risk_analysis import assess_risk_level, generate_early_warnings
from recommendation_engine import generate_recommendations
from anomaly_detection import check_anomaly, load_anomaly_model
from clustering import assign_cluster, load_cluster_model
from forecasting import forecast_financial_history
import joblib
import numpy as np

# ------------------------------------------------------------
# Page configuration - must be the FIRST Streamlit command
# (moved here, before any other st.* call, which set_page_config requires)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Financial Health & Risk Management System",
    page_icon="💰",
    layout="wide"
)

# ------------------------------------------------------------
# Load the ORIGINAL trained loan model artifacts ONCE and cache
# them across reruns. Streamlit reruns this entire script on
# every click/interaction - without @st.cache_resource, these
# joblib.load() calls (especially the Random Forest) would reload
# from disk on every single interaction, which is the main cause
# of the app feeling laggy. Cached resources persist across
# reruns for the whole session, so this now loads only once.
# ------------------------------------------------------------
@st.cache_resource
def load_loan_model_artifacts():
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    label_encoders = joblib.load(os.path.join("models", "label_encoders.pkl"))
    loan_model = joblib.load(os.path.join("models", "random_forest.pkl"))
    return scaler, label_encoders, loan_model


_scaler, _label_encoders, _loan_model = load_loan_model_artifacts()


@st.cache_resource
def get_cached_anomaly_model():
    return load_anomaly_model()


@st.cache_resource
def get_cached_cluster_artifacts():
    return load_cluster_model()


def predict_loan_approval(applicant: dict) -> dict:
    """Runs the original Unit 1-3 loan model on a full applicant
    profile (demographic + loan fields), returning the prediction,
    probability, and top contributing factors (explainability)."""
    input_dict = {
        "Gender": applicant["gender"],
        "Married": applicant["married"],
        "Dependents": str(applicant["dependents"]).replace("3+", "3"),
        "Education": applicant["education"],
        "Self_Employed": applicant["self_employed"],
        "Loan_Amount_Term": applicant["loan_amount_term"],
        "Credit_History": applicant["credit_history"],
    }
    df_input = pd.DataFrame([input_dict])

    for col in ["Gender", "Married", "Education", "Self_Employed", "Dependents"]:
        le = _label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    df_input["Property_Area_Semiurban"] = 1 if applicant["property_area"] == "Semiurban" else 0
    df_input["Property_Area_Urban"] = 1 if applicant["property_area"] == "Urban" else 0

    total_income = applicant["applicant_income"] + applicant["coapplicant_income"]
    df_input["TotalIncome_log"] = np.log1p(total_income)
    df_input["LoanAmount_log"] = np.log1p(applicant["loan_amount"])

    feature_order = _scaler.feature_names_in_
    df_input = df_input[feature_order]
    X_scaled = _scaler.transform(df_input)

    prediction = _loan_model.predict(X_scaled)[0]
    probabilities = _loan_model.predict_proba(X_scaled)[0]

    importances = _loan_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_order, "Importance": importances
    }).sort_values("Importance", ascending=False).head(5)

    return {
        "loan_prediction": "Approved" if prediction == 1 else "Rejected",
        "loan_approval_probability": round(float(probabilities[1]) * 100, 1),
        "loan_rejection_probability": round(float(probabilities[0]) * 100, 1),
        "top_factors": importance_df,
    }

# ------------------------------------------------------------
# Custom CSS: card styling, bigger nav buttons, hover + fade
# transitions. Keeps everything native Streamlit components
# underneath (no extra libraries) - this just restyles them.
# ------------------------------------------------------------
st.markdown("""
<style>
/* Card-style bordered containers (customer rows, forms, etc.) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.30);
}

/* Highlight the selected customer's card. Relies on a hidden
   .selected-marker span placed inside that specific card - modern
   CSS :has() lets us style the PARENT card based on a child marker,
   which Streamlit doesn't otherwise let us do conditionally.
   (Supported in current Chrome, Edge, Safari; falls back to the
   plain card style above on older browsers.) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.selected-marker) {
    background-color: rgba(46, 125, 91, 0.15) !important;
    border: 2px solid #2E7D5B !important;
    box-shadow: 0 4px 14px rgba(46, 125, 91, 0.35) !important;
}

/* KPI metric cards */
div[data-testid="stMetric"] {
    background-color: rgba(135, 135, 135, 0.08);
    border: 1px solid rgba(135, 135, 135, 0.18);
    border-radius: 14px;
    padding: 14px 16px;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.20);
}

/* Bigger, card-like sidebar navigation buttons */
section[data-testid="stSidebar"] button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    padding: 0.85rem 1.1rem !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    transition: all 0.18s ease-in-out !important;
}
section[data-testid="stSidebar"] button:hover {
    transform: translateX(4px);
}

/* Regular buttons elsewhere (Select, Delete, form submits) */
button[kind="primary"], button[kind="secondary"] {
    border-radius: 10px !important;
    transition: all 0.18s ease-in-out !important;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Initialize the database (creates tables if they don't exist -
# safe to call every time the app starts)
# ------------------------------------------------------------
db.init_db()

# ------------------------------------------------------------
# Session state initialization
# Streamlit reruns the whole script on every interaction, so we
# use session_state to remember which customer is selected AND
# which page is active across those reruns.
# ------------------------------------------------------------
if "selected_user_id" not in st.session_state:
    st.session_state.selected_user_id = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


# ============================================================
# SIDEBAR NAVIGATION
# Built with real buttons (not a radio) so each one can be
# styled as a big, clickable card via the CSS above. The active
# page's button is rendered with type="primary" so it's visually
# highlighted - Streamlit's native way of marking an active state.
# ============================================================
st.sidebar.title("💰 Financial Health System")

NAV_ITEMS = [
    ("📊  Dashboard", "Dashboard"),
    ("👥  Customer Management", "Customer Management"),
    ("❤️  Financial Health", "Financial Health"),
    ("🏦  Loan Prediction", "Loan Prediction"),
    ("⚠️  Risk Analysis", "Risk Analysis"),
    ("🔮  What-If Simulator", "What-If Simulator"),
    ("📈  Financial History", "Financial History"),
    ("📉  Analytics", "Analytics"),
    ("🤖  Model Comparison", "Model Comparison"),
]

for label, page_name in NAV_ITEMS:
    is_active = st.session_state.current_page == page_name
    # Prefix the active page's label so it's visually distinct even on
    # older Streamlit versions where the 'type' argument isn't available.
    display_label = f"➤ {label}" if is_active else label
    try:
        clicked = st.sidebar.button(
            display_label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
    except TypeError:
        # Fallback for older Streamlit versions without the 'type' argument
        clicked = st.sidebar.button(
            display_label, key=f"nav_{page_name}", use_container_width=True
        )
    if clicked:
        st.session_state.current_page = page_name
        st.rerun()

page = st.session_state.current_page

# ------------------------------------------------------------
# Quick Customer Switcher - lets you change the selected customer
# from ANY page, not just Customer Management. Convenience feature
# on top of the existing selection mechanism - doesn't replace it.
#
# IMPORTANT: because this selectbox has an explicit key, Streamlit
# uses its persisted session_state value on every rerun and IGNORES
# the 'index' argument after the first render. Without special care,
# this would silently overwrite selected_user_id (e.g. right after
# clicking "Select" in Customer Management) with the dropdown's own
# stale value. We fix this by syncing the widget's session_state
# value to match selected_user_id BEFORE creating the widget, then
# only treating a difference AFTER creation as a genuine user change.
# ------------------------------------------------------------
st.sidebar.divider()
_all_users_for_switcher = db.get_all_users()
if _all_users_for_switcher:
    _user_names = ["— none —"] + [u["name"] for u in _all_users_for_switcher]

    _current_name = "— none —"
    if st.session_state.selected_user_id:
        _current_user = db.get_user(st.session_state.selected_user_id)
        if _current_user:
            _current_name = _current_user["name"]

    # Sync BEFORE creating the widget - safe to set a keyed widget's
    # session_state value before it's instantiated in this run.
    if st.session_state.get("quick_switch_customer") != _current_name:
        st.session_state["quick_switch_customer"] = _current_name

    _chosen_name = st.sidebar.selectbox(
        "Quick switch customer:", _user_names,
        key="quick_switch_customer",
    )

    # Only act if the user actually changed the dropdown THIS run -
    # i.e. it now differs from what we just synced it to above.
    if _chosen_name != _current_name:
        if _chosen_name == "— none —":
            st.session_state.selected_user_id = None
        else:
            _match = next((u for u in _all_users_for_switcher if u["name"] == _chosen_name), None)
            if _match:
                st.session_state.selected_user_id = _match["id"]
        st.rerun()
else:
    st.sidebar.caption("No customers yet — add one in Customer Management.")



def render_gauge(value: float, title: str, max_value: float = 100,
                  low_threshold: float = 50, high_threshold: float = 70,
                  reverse_colors: bool = False) -> None:
    """
    Renders a Plotly gauge chart - used for the Financial Health
    Score and Risk Probability visuals. reverse_colors=True flips
    the color bands (used for Risk, where LOW numbers are good,
    unlike Health Score where HIGH numbers are good).
    """
    if reverse_colors:
        bands = [
            {"range": [0, low_threshold], "color": "rgba(46, 125, 91, 0.35)"},
            {"range": [low_threshold, high_threshold], "color": "rgba(230, 180, 60, 0.35)"},
            {"range": [high_threshold, max_value], "color": "rgba(200, 60, 60, 0.35)"},
        ]
    else:
        bands = [
            {"range": [0, low_threshold], "color": "rgba(200, 60, 60, 0.35)"},
            {"range": [low_threshold, high_threshold], "color": "rgba(230, 180, 60, 0.35)"},
            {"range": [high_threshold, max_value], "color": "rgba(46, 125, 91, 0.35)"},
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#2E7D5B" if not reverse_colors else "#C83C3C"},
            "steps": bands,
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)



# Show which customer is currently selected, in the sidebar, on every page
st.sidebar.divider()
if st.session_state.selected_user_id:
    user = db.get_user(st.session_state.selected_user_id)
    if user:
        st.sidebar.success(f"Selected customer:\n**{user['name']}**")
    else:
        st.session_state.selected_user_id = None
else:
    st.sidebar.info("No customer selected.\nGo to Customer Management.")


# ============================================================
# SAMPLE DATA SEEDING
# Adds a handful of varied demo customers with financial records,
# so the dashboard/app doesn't look empty during a demo. Skips
# adding duplicates if sample customers already exist.
# ============================================================
SAMPLE_CUSTOMERS = [
    {
        "name": "Ravi Kumar", "email": "ravi.kumar@example.com", "phone": "9876500001",
        "record": {"monthly_income": 55000, "monthly_expenses": 22000, "monthly_savings": 15000,
                   "existing_emi": 4000, "other_debt": 0, "loan_amount": 150,
                   "loan_tenure_months": 60, "credit_history": 1.0, "dependents": 1,
                   "gender": "Male", "married": "Yes", "education": "Graduate",
                   "self_employed": "No", "property_area": "Urban", "employment_status": "Salaried"}
    },
    {
        "name": "Anita Sharma", "email": "anita.sharma@example.com", "phone": "9876500002",
        "record": {"monthly_income": 32000, "monthly_expenses": 18000, "monthly_savings": 4000,
                   "existing_emi": 6000, "other_debt": 2000, "loan_amount": 200,
                   "loan_tenure_months": 36, "credit_history": 1.0, "dependents": 2,
                   "gender": "Female", "married": "Yes", "education": "Graduate",
                   "self_employed": "No", "property_area": "Semiurban", "employment_status": "Salaried"}
    },
    {
        "name": "Mohammed Faisal", "email": "faisal@example.com", "phone": "9876500003",
        "record": {"monthly_income": 20000, "monthly_expenses": 19000, "monthly_savings": 200,
                   "existing_emi": 8000, "other_debt": 5000, "loan_amount": 250,
                   "loan_tenure_months": 36, "credit_history": 0.0, "dependents": 3,
                   "gender": "Male", "married": "Yes", "education": "Not Graduate",
                   "self_employed": "Yes", "property_area": "Rural", "employment_status": "Self-Employed"}
    },
    {
        "name": "Priya Nair", "email": "priya.nair@example.com", "phone": "9876500004",
        "record": {"monthly_income": 75000, "monthly_expenses": 25000, "monthly_savings": 25000,
                   "existing_emi": 5000, "other_debt": 0, "loan_amount": 300,
                   "loan_tenure_months": 84, "credit_history": 1.0, "dependents": 0,
                   "gender": "Female", "married": "No", "education": "Graduate",
                   "self_employed": "No", "property_area": "Urban", "employment_status": "Salaried"}
    },
    {
        "name": "Arjun Verma", "email": "arjun.verma@example.com", "phone": "9876500005",
        "record": {"monthly_income": 15000, "monthly_expenses": 8000, "monthly_savings": 500,
                   "existing_emi": 0, "other_debt": 0, "loan_amount": 80,
                   "loan_tenure_months": 24, "credit_history": 0.0, "dependents": 0,
                   "gender": "Male", "married": "No", "education": "Graduate",
                   "self_employed": "No", "property_area": "Urban", "employment_status": "Student"}
    },
]


def seed_sample_customers():
    """Adds the sample customers (with one financial record + full
    analysis each) if they don't already exist, matched by email."""
    existing = {u["email"] for u in db.get_all_users() if u["email"]}
    added = 0

    for sample in SAMPLE_CUSTOMERS:
        if sample["email"] in existing:
            continue   # skip if already seeded

        user_id = db.add_user(sample["name"], sample["email"], sample["phone"])
        record = sample["record"]
        record_id = db.add_financial_record(user_id, record)

        # Run the same analysis a real submission would trigger, so
        # sample customers show up properly in KPIs/history/etc.
        features = compute_financial_features(
            monthly_income=record["monthly_income"],
            monthly_expenses=record["monthly_expenses"],
            monthly_savings=record["monthly_savings"],
            existing_emi=record["existing_emi"],
            other_debt=record["other_debt"],
            loan_amount=record["loan_amount"],
            loan_tenure_months=record["loan_tenure_months"],
            credit_history=record["credit_history"],
            dependents=record["dependents"],
        )
        health = compute_financial_health_score(features)
        risk = assess_risk_level(features)
        cluster = assign_cluster(features, get_cached_cluster_artifacts())
        anomaly = check_anomaly(features, get_cached_anomaly_model())

        db.add_prediction(record_id, {
            "loan_prediction": "Approved" if record["credit_history"] == 1.0 else "Rejected",
            "loan_approval_probability": 0.8 if record["credit_history"] == 1.0 else 0.3,
            "health_score": health["financial_health_score"],
            "health_category": health["category"],
            "risk_level": risk["risk_level"],
            "risk_probability": risk["risk_probability"],
            "cluster_label": cluster["cluster_label"],
            "is_anomaly": anomaly["is_anomaly"],
        })
        added += 1

    return added


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def render_dashboard():
    st.title("📊 Dashboard")
    st.caption("Overview of all customers and system activity")

    all_users = db.get_all_users()

    # -------------------- KPI cards --------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(all_users))

    # Gather every prediction across every user for aggregate KPIs
    all_predictions = []
    for u in all_users:
        all_predictions.extend(db.get_predictions_for_user(u["id"]))

    with col2:
        if all_predictions:
            avg_health = sum(p["health_score"] for p in all_predictions if p["health_score"]) / len(all_predictions)
            st.metric("Avg. Health Score", f"{avg_health:.1f}")
        else:
            st.metric("Avg. Health Score", "—")

    with col3:
        approved_count = sum(1 for p in all_predictions if p["loan_prediction"] == "Approved")
        st.metric("Loans Approved", approved_count)

    with col4:
        high_risk_count = sum(1 for p in all_predictions if p["risk_level"] == "High Risk")
        st.metric("High Risk Customers", high_risk_count)

    st.divider()

    # -------------------- Recent activity table --------------------
    st.subheader("Recent Customers")
    if all_users:
        df_users = pd.DataFrame(all_users)[["id", "name", "email", "phone", "created_at"]]
        st.dataframe(df_users, use_container_width=True, hide_index=True)
    else:
        st.warning("No customers yet. Add one from the Customer Management page.")


# ============================================================
# PAGE: CUSTOMER MANAGEMENT
# ============================================================
def render_customer_management():
    st.title("👥 Customer Management")
    st.caption("Add, search, update, delete, and select customers")

    tab1, tab2 = st.tabs(["📋 All Customers", "➕ Add New Customer"])

    # -------------------- TAB 1: Browse / Search / Select / Delete --------------------
    with tab1:
        col_search, col_seed = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search by name or email")
        with col_seed:
            st.write("")  # small vertical spacer to align button with the text input
            if st.button("🌱 Add Sample Customers"):
                added = seed_sample_customers()
                if added > 0:
                    st.success(f"Added {added} sample customer(s).")
                    st.rerun()
                else:
                    st.info("Sample customers already exist.")

        if search_term:
            users = db.search_users(search_term)
        else:
            users = db.get_all_users()

        if not users:
            st.info("No customers found.")
        else:
            for user in users:
                is_selected = (st.session_state.selected_user_id == user["id"])

                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                    with col1:
                        if is_selected:
                            # Fully self-contained colored badge (inline HTML/CSS
                            # we control directly) instead of relying on reaching
                            # into Streamlit's internal container structure, which
                            # varies across Streamlit versions and wasn't reliably
                            # matching. This is guaranteed to render regardless
                            # of the installed Streamlit version.
                            st.markdown(
                                f"""<div style="background-color:rgba(46,125,91,0.18);
                                     border:2px solid #2E7D5B; border-radius:10px;
                                     padding:8px 12px;">
                                     <strong>✅ {user['name']}</strong><br>
                                     <span style="font-size:0.85em; opacity:0.75;">
                                     {user['email'] or 'no email'} · {user['phone'] or 'no phone'}
                                     </span></div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"**{user['name']}**")
                            st.caption(f"{user['email'] or 'no email'} · {user['phone'] or 'no phone'}")

                    with col2:
                        record_count = len(db.get_records_for_user(user["id"]))
                        st.caption(f"{record_count} financial record(s)")

                    with col3:
                        if is_selected:
                            st.button("✅ Selected", key=f"select_{user['id']}", disabled=True,
                                      use_container_width=True)
                        else:
                            if st.button("Select", key=f"select_{user['id']}", use_container_width=True):
                                st.session_state.selected_user_id = user["id"]
                                st.rerun()

                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{user['id']}", use_container_width=True):
                            db.delete_user(user["id"])
                            if st.session_state.selected_user_id == user["id"]:
                                st.session_state.selected_user_id = None
                            st.rerun()

    # -------------------- TAB 2: Add New Customer --------------------
    with tab2:
        with st.form("add_customer_form", clear_on_submit=True):
            name = st.text_input("Full Name *")
            email = st.text_input("Email")
            phone = st.text_input("Phone")

            submitted = st.form_submit_button("Add Customer")
            if submitted:
                if not name.strip():
                    st.error("Name is required.")
                else:
                    new_id = db.add_user(name.strip(), email.strip(), phone.strip())
                    st.success(f"Customer '{name}' added successfully (ID: {new_id}).")
                    st.session_state.selected_user_id = new_id


# ============================================================
# PAGE: FINANCIAL HEALTH
# ============================================================
def render_financial_health():
    st.title("❤️ Financial Health")
    st.caption("Enter financial details to compute a Health Score, warnings, and recommendations")

    if not st.session_state.selected_user_id:
        st.warning("No customer selected. Go to **Customer Management** and click **Select** on a customer first.")
        return

    user = db.get_user(st.session_state.selected_user_id)
    if not user:
        st.session_state.selected_user_id = None
        st.warning("Selected customer no longer exists.")
        return

    st.subheader(f"Customer: {user['name']}")

    # -------------------- Input form --------------------
    with st.form("financial_health_form"):
        col1, col2 = st.columns(2)

        with col1:
            monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=35000, step=1000)
            monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=18000, step=1000)
            monthly_savings = st.number_input("Monthly Savings (₹)", min_value=0, value=5000, step=500)
            existing_emi = st.number_input("Existing EMI (₹)", min_value=0, value=3000, step=500)
            other_debt = st.number_input("Other Debt Payments (₹)", min_value=0, value=0, step=500)

        with col2:
            loan_amount = st.number_input("Loan Amount Requested (in thousands ₹)", min_value=0, value=150, step=10)
            loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, value=60, step=6)
            credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                           format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor/None (0.0)")
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Analyze Financial Health", use_container_width=True)

    if not submitted:
        return

    # -------------------- Run analysis --------------------
    features = compute_financial_features(
        monthly_income=monthly_income, monthly_expenses=monthly_expenses,
        monthly_savings=monthly_savings, existing_emi=existing_emi,
        other_debt=other_debt, loan_amount=loan_amount,
        loan_tenure_months=loan_tenure_months, credit_history=credit_history,
        dependents=dependents,
    )
    health = compute_financial_health_score(features)
    risk = assess_risk_level(features)
    warnings_list = generate_early_warnings(features)
    recommendations = generate_recommendations(features, risk)
    cluster = assign_cluster(features, get_cached_cluster_artifacts())
    anomaly = check_anomaly(features, get_cached_anomaly_model())

    # -------------------- Save to database --------------------
    record_id = db.add_financial_record(user["id"], {
        "monthly_income": monthly_income, "monthly_expenses": monthly_expenses,
        "monthly_savings": monthly_savings, "existing_emi": existing_emi,
        "other_debt": other_debt, "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months, "credit_history": credit_history,
        "dependents": dependents, "gender": None, "married": None,
        "education": None, "self_employed": None, "property_area": None,
        "employment_status": None,
    })
    db.add_prediction(record_id, {
        "loan_prediction": None, "loan_approval_probability": None,
        "health_score": health["financial_health_score"], "health_category": health["category"],
        "risk_level": risk["risk_level"], "risk_probability": risk["risk_probability"],
        "cluster_label": cluster["cluster_label"], "is_anomaly": anomaly["is_anomaly"],
    })

    st.divider()

    # -------------------- Results: KPI row --------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Health Score", f"{health['financial_health_score']}/100", health["category"])
    with col2:
        st.metric("Risk Level", risk["risk_level"], f"{risk['risk_probability']}% risk")
    with col3:
        st.metric("Spending Profile", cluster["cluster_label"])
    with col4:
        st.metric("Anomaly Check", "⚠️ Unusual" if anomaly["is_anomaly"] else "✅ Typical")

    # -------------------- Health score gauge chart --------------------
    st.subheader("Financial Health Score")
    render_gauge(health["financial_health_score"], "Health Score (0-100)",
                 low_threshold=50, high_threshold=70)

    # -------------------- Sub-score breakdown --------------------
    with st.expander("📊 See score breakdown (how the 0-100 score was calculated)"):
        sub_df = pd.DataFrame(
            list(health["sub_scores"].items()), columns=["Component", "Sub-score (0-100)"]
        )
        st.bar_chart(sub_df.set_index("Component"))
        st.caption(
            "Weights used: Savings 20%, EMI burden 20%, Expense ratio 20%, "
            "Credit history 20%, Debt burden 10%, Loan-to-income 10%."
        )

    # -------------------- Warnings --------------------
    st.subheader("⚠️ Early Warnings")
    if warnings_list:
        for w in warnings_list:
            st.error(w)
    else:
        st.success("No warnings triggered — this financial profile looks stable.")

    # -------------------- Recommendations --------------------
    st.subheader("💡 Personalized Recommendations")
    for rec in recommendations:
        st.info(rec)

    # -------------------- Key derived numbers --------------------
    with st.expander("🔍 See detailed derived figures"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Monthly EMI:** ₹{features['total_emi']:.0f}")
            st.write(f"**EMI-to-Income Ratio:** {features['emi_to_income_ratio']*100:.1f}%")
            st.write(f"**Savings Rate:** {features['savings_rate']*100:.1f}%")
        with col2:
            st.write(f"**Disposable Income:** ₹{features['disposable_income']:.0f}/month")
            st.write(f"**Loan-to-Income Ratio:** {features['loan_to_income_ratio']:.2f}x")
            st.write(f"**Debt Burden Ratio:** {features['debt_burden_ratio']*100:.1f}%")

    st.caption("This assessment has been saved to the customer's Financial History.")


# ============================================================
# PAGE: LOAN PREDICTION
# ============================================================
def render_loan_prediction():
    st.title("🏦 Loan Prediction")
    st.caption("Uses the original Unit 1-3 trained Random Forest model")

    if not st.session_state.selected_user_id:
        st.warning("No customer selected. Go to **Customer Management** and select a customer first.")
        return

    user = db.get_user(st.session_state.selected_user_id)
    if not user:
        st.session_state.selected_user_id = None
        return

    st.subheader(f"Customer: {user['name']}")

    with st.form("loan_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

        with col2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col3:
            applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=150, step=10)

        col4, col5 = st.columns(2)
        with col4:
            loan_amount_term = st.selectbox("Loan Amount Term (days)", [360, 180, 120, 84, 60, 36, 12, 300, 240, 480])
        with col5:
            credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                           format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor/None (0.0)")

        submitted = st.form_submit_button("Predict Loan Eligibility", use_container_width=True)

    if not submitted:
        return

    applicant = {
        "gender": gender, "married": married, "dependents": dependents,
        "education": education, "self_employed": self_employed,
        "property_area": property_area, "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income, "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term, "credit_history": credit_history,
    }

    result = predict_loan_approval(applicant)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if result["loan_prediction"] == "Approved":
            st.success(f"✅ Loan Status: **APPROVED**")
        else:
            st.error(f"❌ Loan Status: **REJECTED**")
        st.metric("Approval Probability", f"{result['loan_approval_probability']}%")
        st.metric("Rejection Probability", f"{result['loan_rejection_probability']}%")

    with col2:
        st.subheader("Why this decision? (Explainable AI)")
        chart_df = result["top_factors"].set_index("Feature")
        st.bar_chart(chart_df)
        st.caption("Top 5 factors ranked by the Random Forest model's feature importance.")

    if applicant["dependents"] == "0" and coapplicant_income == 0 and credit_history == 0.0:
        st.info(
            "ℹ️ Note: applicants with no co-applicant income and poor credit history "
            "are historically underrepresented in the training data — treat this "
            "prediction with extra caution."
        )


# ============================================================
# PAGE: RISK ANALYSIS
# ============================================================
def render_risk_analysis():
    st.title("⚠️ Risk Analysis")
    st.caption("Detailed risk classification and contributing factors")

    if not st.session_state.selected_user_id:
        st.warning("No customer selected. Go to **Customer Management** and select a customer first.")
        return

    user = db.get_user(st.session_state.selected_user_id)
    if not user:
        st.session_state.selected_user_id = None
        return

    st.subheader(f"Customer: {user['name']}")

    # Reuse the customer's most recent financial record if one exists,
    # so risk analysis doesn't require re-entering everything from scratch.
    records = db.get_records_for_user(user["id"])
    latest = records[-1] if records else None

    with st.form("risk_analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input(
                "Monthly Income (₹)", min_value=0,
                value=int(latest["monthly_income"]) if latest else 35000, step=1000)
            monthly_expenses = st.number_input(
                "Monthly Expenses (₹)", min_value=0,
                value=int(latest["monthly_expenses"]) if latest else 18000, step=1000)
            monthly_savings = st.number_input(
                "Monthly Savings (₹)", min_value=0,
                value=int(latest["monthly_savings"]) if latest else 5000, step=500)
        with col2:
            existing_emi = st.number_input(
                "Existing EMI (₹)", min_value=0,
                value=int(latest["existing_emi"]) if latest else 3000, step=500)
            other_debt = st.number_input(
                "Other Debt (₹)", min_value=0,
                value=int(latest["other_debt"]) if latest else 0, step=500)
            credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                           index=0 if (not latest or latest["credit_history"] == 1.0) else 1,
                                           format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor/None (0.0)")

        col3, col4 = st.columns(2)
        with col3:
            loan_amount = st.number_input(
                "Loan Amount (thousands ₹)", min_value=0,
                value=int(latest["loan_amount"]) if latest else 150, step=10)
        with col4:
            loan_tenure_months = st.number_input(
                "Loan Tenure (months)", min_value=1,
                value=int(latest["loan_tenure_months"]) if latest else 60, step=6)

        dependents = st.number_input("Dependents", min_value=0, max_value=10,
                                      value=int(latest["dependents"]) if latest else 0, step=1)

        submitted = st.form_submit_button("Run Risk Analysis", use_container_width=True)

    if not submitted:
        if latest:
            st.caption("Form pre-filled with the customer's most recent financial record. Adjust and submit if needed.")
        return

    features = compute_financial_features(
        monthly_income=monthly_income, monthly_expenses=monthly_expenses,
        monthly_savings=monthly_savings, existing_emi=existing_emi,
        other_debt=other_debt, loan_amount=loan_amount,
        loan_tenure_months=loan_tenure_months, credit_history=credit_history,
        dependents=dependents,
    )
    risk = assess_risk_level(features)
    warnings_list = generate_early_warnings(features)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if risk["risk_level"] == "Low Risk":
            st.success(f"🟢 Risk Level: **{risk['risk_level']}**")
        elif risk["risk_level"] == "Medium Risk":
            st.warning(f"🟡 Risk Level: **{risk['risk_level']}**")
        else:
            st.error(f"🔴 Risk Level: **{risk['risk_level']}**")
        render_gauge(risk["risk_probability"], "Risk Probability (%)",
                     low_threshold=30, high_threshold=60, reverse_colors=True)

    with col2:
        st.subheader("Contributing Factors")
        for factor in risk["contributing_factors"]:
            st.write(f"• {factor}")

    st.subheader("Early Warnings")
    if warnings_list:
        for w in warnings_list:
            st.error(w)
    else:
        st.success("No warnings triggered for this profile.")


# ============================================================
# PAGE: WHAT-IF SIMULATOR
# ============================================================
def render_what_if_simulator():
    st.title("🔮 What-If Simulator")
    st.caption("Adjust income, expenses, or loan details and instantly see the impact — "
               "nothing here is saved to the database.")

    if not st.session_state.selected_user_id:
        st.warning("No customer selected. Go to **Customer Management** and select a customer first.")
        return

    user = db.get_user(st.session_state.selected_user_id)
    if not user:
        st.session_state.selected_user_id = None
        return

    st.subheader(f"Customer: {user['name']}")

    # Pre-fill baseline values from the customer's latest saved record, if any
    records = db.get_records_for_user(user["id"])
    latest = records[-1] if records else None

    baseline_defaults = {
        "monthly_income": int(latest["monthly_income"]) if latest else 35000,
        "monthly_expenses": int(latest["monthly_expenses"]) if latest else 18000,
        "monthly_savings": int(latest["monthly_savings"]) if latest else 5000,
        "existing_emi": int(latest["existing_emi"]) if latest else 3000,
        "other_debt": int(latest["other_debt"]) if latest else 0,
        "loan_amount": int(latest["loan_amount"]) if latest else 150,
        "loan_tenure_months": int(latest["loan_tenure_months"]) if latest else 60,
        "credit_history": latest["credit_history"] if latest else 1.0,
        "dependents": int(latest["dependents"]) if latest else 0,
    }

    st.info("👇 Adjust any value below. The **baseline** (current) and **simulated** (adjusted) "
            "results are compared side-by-side.")

    col1, col2 = st.columns(2)
    with col1:
        sim_income = st.slider("Monthly Income (₹)", 0, 150000, baseline_defaults["monthly_income"], step=1000)
        sim_expenses = st.slider("Monthly Expenses (₹)", 0, 100000, baseline_defaults["monthly_expenses"], step=1000)
        sim_savings = st.slider("Monthly Savings (₹)", 0, 50000, baseline_defaults["monthly_savings"], step=500)
        sim_emi = st.slider("Existing EMI (₹)", 0, 50000, baseline_defaults["existing_emi"], step=500)
    with col2:
        sim_debt = st.slider("Other Debt (₹)", 0, 30000, baseline_defaults["other_debt"], step=500)
        sim_loan = st.slider("Loan Amount (thousands ₹)", 0, 1000, baseline_defaults["loan_amount"], step=10)
        sim_tenure = st.slider("Loan Tenure (months)", 6, 480, baseline_defaults["loan_tenure_months"], step=6)
        sim_credit = st.selectbox("Credit History", [1.0, 0.0],
                                   index=0 if baseline_defaults["credit_history"] == 1.0 else 1,
                                   format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor/None (0.0)")

    # -------------------- Compute BOTH baseline and simulated results --------------------
    def compute_snapshot(income, expenses, savings, emi, debt, loan, tenure, credit, dependents):
        features = compute_financial_features(
            monthly_income=income, monthly_expenses=expenses, monthly_savings=savings,
            existing_emi=emi, other_debt=debt, loan_amount=loan,
            loan_tenure_months=tenure, credit_history=credit, dependents=dependents,
        )
        health = compute_financial_health_score(features)
        risk = assess_risk_level(features)
        return features, health, risk

    baseline_features, baseline_health, baseline_risk = compute_snapshot(
        baseline_defaults["monthly_income"], baseline_defaults["monthly_expenses"],
        baseline_defaults["monthly_savings"], baseline_defaults["existing_emi"],
        baseline_defaults["other_debt"], baseline_defaults["loan_amount"],
        baseline_defaults["loan_tenure_months"], baseline_defaults["credit_history"],
        baseline_defaults["dependents"],
    )
    sim_features, sim_health, sim_risk = compute_snapshot(
        sim_income, sim_expenses, sim_savings, sim_emi, sim_debt,
        sim_loan, sim_tenure, sim_credit, baseline_defaults["dependents"],
    )

    st.divider()
    st.subheader("Baseline vs Simulated Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        delta = round(sim_health["financial_health_score"] - baseline_health["financial_health_score"], 1)
        st.metric("Financial Health Score", f"{sim_health['financial_health_score']}/100",
                   delta=f"{delta:+.1f} vs baseline")
        st.caption(f"Baseline: {baseline_health['financial_health_score']}/100 ({baseline_health['category']}) "
                    f"→ Simulated: {sim_health['category']}")

    with col2:
        st.metric("Risk Probability", f"{sim_risk['risk_probability']}%",
                   delta=f"{sim_risk['risk_probability'] - baseline_risk['risk_probability']:+.1f}% vs baseline",
                   delta_color="inverse")
        st.caption(f"Baseline: {baseline_risk['risk_level']} → Simulated: {sim_risk['risk_level']}")

    with col3:
        delta_disp = round(sim_features["disposable_income"] - baseline_features["disposable_income"], 0)
        st.metric("Disposable Income", f"₹{sim_features['disposable_income']:.0f}",
                   delta=f"₹{delta_disp:+.0f} vs baseline")

    # -------------------- Side-by-side chart --------------------
    st.subheader("Health Score Comparison")
    comparison_df = pd.DataFrame({
        "Scenario": ["Baseline", "Simulated"],
        "Health Score": [baseline_health["financial_health_score"], sim_health["financial_health_score"]],
    }).set_index("Scenario")
    st.bar_chart(comparison_df)

    # -------------------- Loan approval probability, if fields are complete enough --------------------
    with st.expander("🏦 Also simulate Loan Approval Probability"):
        st.caption("Requires a few extra demographic fields not covered by the sliders above.")
        col1, col2 = st.columns(2)
        with col1:
            sim_gender = st.selectbox("Gender", ["Male", "Female"], key="whatif_gender")
            sim_married = st.selectbox("Married", ["Yes", "No"], key="whatif_married")
            sim_education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="whatif_education")
        with col2:
            sim_self_employed = st.selectbox("Self Employed", ["No", "Yes"], key="whatif_self_employed")
            sim_property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], key="whatif_property")
            sim_coapplicant = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=500, key="whatif_coapp")

        if st.button("Compare Loan Approval Probability"):
            baseline_applicant = {
                "gender": sim_gender, "married": sim_married,
                "dependents": str(baseline_defaults["dependents"]),
                "education": sim_education, "self_employed": sim_self_employed,
                "property_area": sim_property_area,
                "applicant_income": baseline_defaults["monthly_income"],
                "coapplicant_income": sim_coapplicant,
                "loan_amount": baseline_defaults["loan_amount"],
                "loan_amount_term": baseline_defaults["loan_tenure_months"],
                "credit_history": baseline_defaults["credit_history"],
            }
            sim_applicant = {
                "gender": sim_gender, "married": sim_married,
                "dependents": str(baseline_defaults["dependents"]),
                "education": sim_education, "self_employed": sim_self_employed,
                "property_area": sim_property_area,
                "applicant_income": sim_income, "coapplicant_income": sim_coapplicant,
                "loan_amount": sim_loan, "loan_amount_term": sim_tenure,
                "credit_history": sim_credit,
            }
            baseline_loan = predict_loan_approval(baseline_applicant)
            sim_loan_result = predict_loan_approval(sim_applicant)

            colA, colB = st.columns(2)
            with colA:
                st.write("**Baseline**")
                st.write(f"{baseline_loan['loan_prediction']} ({baseline_loan['loan_approval_probability']}% approval)")
            with colB:
                st.write("**Simulated**")
                st.write(f"{sim_loan_result['loan_prediction']} ({sim_loan_result['loan_approval_probability']}% approval)")


# ============================================================
# PAGE: FINANCIAL HISTORY
# ============================================================
def render_financial_history():
    st.title("📈 Financial History")
    st.caption("Trends across all past assessments for this customer, plus a simple forecast")

    if not st.session_state.selected_user_id:
        st.warning("No customer selected. Go to **Customer Management** and select a customer first.")
        return

    user = db.get_user(st.session_state.selected_user_id)
    if not user:
        st.session_state.selected_user_id = None
        return

    st.subheader(f"Customer: {user['name']}")

    records = db.get_records_for_user(user["id"])          # oldest first
    predictions = db.get_predictions_for_user(user["id"])  # newest first

    if not records:
        st.info(
            "No financial history yet for this customer. Submit an assessment on the "
            "**Financial Health** page to start building their history."
        )
        return

    # predictions come back newest-first from the DB; reverse for a
    # chronological (oldest-first) trend, matching the records order
    predictions_chrono = list(reversed(predictions))

    # -------------------- Build a trend dataframe --------------------
    trend_rows = []
    for i, rec in enumerate(records):
        pred = predictions_chrono[i] if i < len(predictions_chrono) else {}
        trend_rows.append({
            "Date": rec["created_at"][:10],   # just the date part, not the full timestamp
            "Income": rec["monthly_income"],
            "Expenses": rec["monthly_expenses"],
            "Savings": rec["monthly_savings"],
            "Health Score": pred.get("health_score"),
            "Risk Level": pred.get("risk_level"),
        })
    trend_df = pd.DataFrame(trend_rows)

    st.metric("Total Assessments Recorded", len(records))

    # -------------------- Trend charts --------------------
    st.subheader("Financial Health Score Trend")
    if trend_df["Health Score"].notna().any():
        st.line_chart(trend_df.set_index("Date")[["Health Score"]])
    else:
        st.info("No health score history recorded yet.")

    st.subheader("Income, Expense & Savings Trend")
    st.line_chart(trend_df.set_index("Date")[["Income", "Expenses", "Savings"]])

    st.subheader("Risk Level Over Time")
    st.dataframe(trend_df[["Date", "Risk Level"]], use_container_width=True, hide_index=True)

    # -------------------- Trend interpretation --------------------
    if len(trend_df) >= 2 and trend_df["Health Score"].notna().sum() >= 2:
        scores = trend_df["Health Score"].dropna().tolist()
        change = scores[-1] - scores[0]
        if change > 5:
            st.success(f"📈 Overall trend: financial health has **improved** by {change:.1f} points "
                       f"since the first recorded assessment.")
        elif change < -5:
            st.error(f"📉 Overall trend: financial health has **declined** by {abs(change):.1f} points "
                     f"since the first recorded assessment.")
        else:
            st.info("➡️ Overall trend: financial health has stayed **roughly stable** over time.")

    st.divider()

    # -------------------- Forecasting --------------------
    st.subheader("🔮 Simple Forecast (Next Period)")
    forecast_result = forecast_financial_history(records)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Expenses Forecast**")
        exp_forecast = forecast_result["expenses_forecast"]
        if exp_forecast["forecast_possible"]:
            st.metric("Projected Next Expenses", f"₹{exp_forecast['forecast_value']:.0f}",
                       delta=exp_forecast["trend_direction"])
        else:
            st.info(exp_forecast["message"])

    with col2:
        st.markdown("**Savings Forecast**")
        sav_forecast = forecast_result["savings_forecast"]
        if sav_forecast["forecast_possible"]:
            st.metric("Projected Next Savings", f"₹{sav_forecast['forecast_value']:.0f}",
                       delta=sav_forecast["trend_direction"])
        else:
            st.info(sav_forecast["message"])

    st.caption(
        "This forecast uses simple linear regression on past records - a straight-line "
        "trend projection, not a guarantee. At least 3 historical records are required."
    )

    # -------------------- Full raw history table --------------------
    with st.expander("📋 See full raw history table"):
        st.dataframe(trend_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download history as CSV",
            data=trend_df.to_csv(index=False),
            file_name=f"{user['name'].replace(' ', '_')}_financial_history.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE: ANALYTICS
# ============================================================
def render_analytics():
    st.title("📉 Analytics")
    st.caption("Aggregate insights across all customers in the system")

    all_users = db.get_all_users()
    if not all_users:
        st.info("No customers yet. Add some from Customer Management to see analytics here.")
        return

    all_predictions = []
    for u in all_users:
        for p in db.get_predictions_for_user(u["id"]):
            p["customer_name"] = u["name"]
            all_predictions.append(p)

    if not all_predictions:
        st.info("No assessments recorded yet. Run some Financial Health analyses first.")
        return

    df = pd.DataFrame(all_predictions)

    # -------------------- Top KPIs --------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(all_users))
    with col2:
        st.metric("Total Assessments", len(df))
    with col3:
        avg_score = df["health_score"].dropna().mean()
        st.metric("Avg. Health Score", f"{avg_score:.1f}" if not pd.isna(avg_score) else "—")
    with col4:
        anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df else 0
        st.metric("Anomalies Flagged", anomaly_count)

    st.divider()

    col1, col2 = st.columns(2)

    # -------------------- Health score distribution --------------------
    with col1:
        st.subheader("Health Score Category Distribution")
        if "health_category" in df and df["health_category"].notna().any():
            cat_counts = df["health_category"].value_counts()
            st.bar_chart(cat_counts)
        else:
            st.info("No health category data yet.")

    # -------------------- Risk level distribution --------------------
    with col2:
        st.subheader("Risk Level Distribution")
        if "risk_level" in df and df["risk_level"].notna().any():
            risk_counts = df["risk_level"].value_counts()
            st.bar_chart(risk_counts)
        else:
            st.info("No risk level data yet.")

    col3, col4 = st.columns(2)

    # -------------------- Cluster distribution --------------------
    with col3:
        st.subheader("Spending Profile (Cluster) Distribution")
        if "cluster_label" in df and df["cluster_label"].notna().any():
            cluster_counts = df["cluster_label"].value_counts()
            st.bar_chart(cluster_counts)
            st.caption(
                "Reminder: clustering is unsupervised and descriptive - it groups similar "
                "customers together, it does not predict approval or risk on its own."
            )
        else:
            st.info("No cluster data yet.")

    # -------------------- Loan approval breakdown --------------------
    with col4:
        st.subheader("Loan Prediction Outcomes")
        if "loan_prediction" in df and df["loan_prediction"].notna().any():
            loan_counts = df["loan_prediction"].value_counts()
            st.bar_chart(loan_counts)
        else:
            st.info("No loan predictions recorded yet (run some from the Loan Prediction page).")

    st.divider()

    # -------------------- Full table --------------------
    with st.expander("📋 See all assessment records"):
        display_cols = [c for c in ["customer_name", "health_score", "health_category",
                                     "risk_level", "cluster_label", "loan_prediction",
                                     "is_anomaly", "created_at"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download all records as CSV",
            data=df[display_cols].to_csv(index=False),
            file_name="all_customer_assessments.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
def render_model_comparison():
    st.title("🤖 Model Comparison")
    st.caption("Results from the original Unit 1-3 training scripts (data_preprocessing.py "
               "through train_ann.py) - this page visualizes existing results, it does not retrain anything.")

    outputs_dir = "outputs"

    # -------------------- Unit 2: Decision Tree vs Random Forest --------------------
    st.subheader("Unit 2: Decision Tree vs Random Forest")
    unit2_path = os.path.join(outputs_dir, "unit2_model_comparison.csv")
    if os.path.exists(unit2_path):
        unit2_df = pd.read_csv(unit2_path)
        st.dataframe(unit2_df, use_container_width=True, hide_index=True)
        metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1-score"] if c in unit2_df.columns]
        if "Model" in unit2_df.columns and metrics:
            st.bar_chart(unit2_df.set_index("Model")[metrics])
    else:
        st.warning(f"'{unit2_path}' not found. Run `python src/train_tree_models.py` first.")

    st.divider()

    # -------------------- Novelty: Log transform comparison --------------------
    st.subheader("Novelty: Before vs After Log Transformation")
    log_path = os.path.join(outputs_dir, "log_transform_comparison.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        metrics = [c for c in ["Accuracy", "F1-score"] if c in log_df.columns]
        if "Version" in log_df.columns and metrics:
            st.bar_chart(log_df.set_index("Version")[metrics])
        st.caption(
            "This proves the log transformation itself improves model performance, "
            "using identical models and settings otherwise - the project's novelty."
        )
    else:
        st.warning(f"'{log_path}' not found. Run `python src/compare_models.py` first.")

    st.divider()

    # -------------------- Unit 3: ANN vs Random Forest --------------------
    st.subheader("Unit 3: ANN vs Random Forest")
    ann_path = os.path.join(outputs_dir, "unit3_ann_vs_rf_comparison.csv")
    if os.path.exists(ann_path):
        ann_df = pd.read_csv(ann_path)
        st.dataframe(ann_df, use_container_width=True, hide_index=True)
        metrics = [c for c in ["Accuracy", "Precision", "Recall", "F1-score"] if c in ann_df.columns]
        if "Model" in ann_df.columns and metrics:
            st.bar_chart(ann_df.set_index("Model")[metrics])
    else:
        st.warning(f"'{ann_path}' not found. Run `python src/train_ann.py` first.")

    st.divider()

    # -------------------- ANN training curves image, if available --------------------
    st.subheader("ANN Training Curves")
    curves_path = os.path.join(outputs_dir, "ann_accuracy_loss_curves.png")
    if os.path.exists(curves_path):
        st.image(curves_path, use_container_width=True)
    else:
        st.info("Training curve image not found yet - run `python src/train_ann.py` to generate it.")


# ============================================================
# ROUTING: render whichever page is selected in the sidebar
# ============================================================
if page == "Dashboard":
    render_dashboard()
elif page == "Customer Management":
    render_customer_management()
elif page == "Financial Health":
    render_financial_health()
elif page == "Loan Prediction":
    render_loan_prediction()
elif page == "Risk Analysis":
    render_risk_analysis()
elif page == "What-If Simulator":
    render_what_if_simulator()
elif page == "Financial History":
    render_financial_history()
elif page == "Analytics":
    render_analytics()
elif page == "Model Comparison":
    render_model_comparison()