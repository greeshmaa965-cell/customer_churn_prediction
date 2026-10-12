"""
app/streamlit_app.py
--------------------
Streamlit dashboard for Customer Churn Prediction with SHAP explanations.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow importing from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.predict import predict_single, predict_batch
from src.train import load_model, BEST_THRESHOLD
from src.preprocessor import load_preprocessor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient header */
.hero-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #56ccf2, #2f80ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-header p {
    font-size: 1rem;
    color: #a0aec0;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: white;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .value { font-size: 2rem; font-weight: 700; }
.metric-card .label { font-size: 0.85rem; color: #a0aec0; margin-top: 0.3rem; }

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.risk-low    { background: #1a4731; color: #68d391; border: 1px solid #68d391; }
.risk-medium { background: #4a3320; color: #f6ad55; border: 1px solid #f6ad55; }
.risk-high   { background: #4a1820; color: #fc8181; border: 1px solid #fc8181; }

/* Section headers */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #2d3748;
}

/* Gauge container */
.gauge-container { text-align: center; padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Caching ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_preprocessor():
    return load_preprocessor()


@st.cache_resource
def get_shap_explainer():
    model = get_model()
    preprocessor = get_preprocessor()

    # Load training data for background
    processed_dir = ROOT / "data" / "processed"
    with open(processed_dir / "x_train_processed.pkl", "rb") as f:
        X_train = pickle.load(f)

    # Use a small background dataset for speed
    background = shap.sample(X_train, 100, random_state=42)
    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
    return explainer


# ── Helper functions ───────────────────────────────────────────────────────────
def build_gauge(probability: float):
    """Draw a simple gauge chart for the churn probability."""
    fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta = np.linspace(0, np.pi, 200)
    r = 1.0

    # Background arc
    ax.plot(r * np.cos(theta), r * np.sin(theta), lw=12, color="#2d3748", solid_capstyle="round")

    # Coloured arc (proportional to probability)
    angle = np.pi * (1 - probability)
    theta_fill = np.linspace(angle, np.pi, 200)
    color = "#fc8181" if probability >= 0.6 else ("#f6ad55" if probability >= 0.35 else "#68d391")
    ax.plot(r * np.cos(theta_fill), r * np.sin(theta_fill), lw=12, color=color, solid_capstyle="round")

    # Needle
    ax.annotate(
        "", xy=(0.7 * np.cos(angle), 0.7 * np.sin(angle)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2)
    )

    ax.text(0, -0.25, f"{probability * 100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis("off")
    return fig


def shap_bar_chart(shap_values, feature_names, top_n=10):
    """Return a Matplotlib figure of the top-N SHAP feature importances."""
    vals = np.array(shap_values).flatten()
    indices = np.argsort(np.abs(vals))[-top_n:]

    feat = [feature_names[i] for i in indices]
    sv = vals[indices]
    colors = ["#fc8181" if v > 0 else "#68d391" for v in sv]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f2027")
    ax.set_facecolor("#0f2027")
    bars = ax.barh(feat, sv, color=colors)
    ax.axvline(0, color="#4a5568", lw=1)
    ax.set_xlabel("SHAP value (impact on churn prediction)", color="#a0aec0")
    ax.set_title("Feature Contributions (SHAP)", color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#a0aec0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    fig.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    page = st.radio("", ["🔍 Single Prediction", "📁 Batch Prediction", "📈 Model Info"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown(f"**Threshold:** `{BEST_THRESHOLD}`")
    st.markdown("**Dataset:** Telco Customer Churn")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>📊 Customer Churn Predictor</h1>
    <p>End-to-end ML system with Explainable AI — powered by Logistic Regression & SHAP</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":
    st.subheader("Customer Profile")
    st.markdown("Fill in the customer details below and click **Predict** to see the churn risk.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", key="senior")
            partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")

        with c2:
            st.markdown('<p class="section-title">Account</p>', unsafe_allow_html=True)
            tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="payment")
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5, key="monthly")
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly), step=1.0, key="total")

        with c3:
            st.markdown('<p class="section-title">Services</p>', unsafe_allow_html=True)
            phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multi")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
            online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="online_sec")
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="online_backup")
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="sm")

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Analysing customer profile..."):
            result = predict_single(input_dict)

        prob = result["churn_probability"]
        risk = result["risk_label"]
        pred = result["churn_prediction"]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### Churn Probability")
            fig_gauge = build_gauge(prob)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

            risk_class = f"risk-{risk.lower()}"
            verdict = "Will Churn" if pred == 1 else "Will Stay"
            st.markdown(f"""
            <div style='text-align:center;margin-top:1rem;'>
                <span class='risk-badge {risk_class}'>{risk} Risk</span>
                <p style='color:#a0aec0;margin-top:0.5rem;font-size:0.9rem;'>{verdict}</p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("### Why this prediction? (SHAP)")
            try:
                preprocessor = get_preprocessor()
                model = get_model()
                explainer = get_shap_explainer()

                X_raw = pd.DataFrame([input_dict])
                X_raw["TotalCharges"] = pd.to_numeric(X_raw["TotalCharges"], errors="coerce").fillna(0)
                X_processed = preprocessor.transform(X_raw)
                feature_names = preprocessor.get_feature_names_out()
                X_df = pd.DataFrame(X_processed, columns=feature_names)

                shap_vals = explainer.shap_values(X_df)
                fig_shap = shap_bar_chart(shap_vals[0], list(feature_names), top_n=10)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Retention Recommendations")
        recs = []
        if contract == "Month-to-month":
            recs.append("Offer a discounted annual or two-year contract to increase retention.")
        if internet == "Fiber optic" and tech_support == "No":
            recs.append("Provide complimentary Tech Support for the first 3 months.")
        if monthly > 80:
            recs.append("Consider a loyalty discount to reduce monthly charges.")
        if tenure < 12:
            recs.append("Enroll customer in new-customer loyalty programme.")
        if not recs:
            recs.append("Customer profile looks stable — continue monitoring.")

        for r in recs:
            st.markdown(f"- {r}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.subheader("Batch Churn Prediction")
    st.markdown("Upload a CSV file with customer data (same schema as the Telco dataset, without the `Churn` column).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)

        # Drop ID column if present
        if "customerID" in df_raw.columns:
            df_raw = df_raw.drop("customerID", axis=1)
        if "Churn" in df_raw.columns:
            df_raw = df_raw.drop("Churn", axis=1)

        st.markdown(f"**{len(df_raw):,} customers uploaded**")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Predicting..."):
                results_df = predict_batch(df_raw.copy())

            st.success("Done!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(results_df))
            col2.metric("Predicted Churners", int(results_df["churn_prediction"].sum()))
            col3.metric("Avg Churn Probability", f"{results_df['churn_probability'].mean():.1%}")

            st.dataframe(
                results_df[["churn_probability", "churn_prediction", "risk_label"] + list(df_raw.columns[:5])],
                use_container_width=True,
            )

            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv_out, "churn_predictions.csv", "text/csv")

    else:
        st.info("Upload a CSV file to begin batch prediction.")
        st.markdown("**Expected columns:** gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.subheader("Model Performance Dashboard")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("Accuracy", "74.95%"),
        ("Precision", "56.3%"),
        ("Recall", "78.9%"),
        ("F1-Score", "65.8%"),
        ("ROC-AUC", "84.2%"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4, c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show saved figures if they exist
    figures_dir = ROOT / "reports" / "figures"
    fig_paths = {
        "Confusion Matrix": figures_dir / "confusion_matrix.png",
        "ROC Curve": figures_dir / "roc_curve.png",
        "Threshold Analysis": figures_dir / "threshold_analysis.png",
    }

    available = {k: v for k, v in fig_paths.items() if v.exists()}

    if available:
        tabs = st.tabs(list(available.keys()))
        for tab, (title, path) in zip(tabs, available.items()):
            with tab:
                st.image(str(path), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate` to generate evaluation plots.")

    st.markdown("---")
    st.markdown("### About the Model")
    st.markdown("""
    | Attribute | Value |
    |-----------|-------|
    | Algorithm | Logistic Regression |
    | Dataset | Telco Customer Churn (IBM) |
    | Train / Test Split | 80% / 20% (stratified) |
    | Decision Threshold | 0.3 (tuned for high recall) |
    | Features | 19 raw → 30 encoded |
    | Encoding | OneHotEncoder + StandardScaler |
    | Explainability | SHAP LinearExplainer |
    """)
