# -*- coding: utf-8 -*-
"""
SBA 7(a) Loan Default Predictor — Streamlit App
Converted from: SMB loans Trees.ipynb (Google Colab)
"""

import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SBA 7(a) Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a78bfa;
        border-left: 4px solid #7c3aed;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(90deg, rgba(124,58,237,0.3), rgba(59,130,246,0.3));
        border: 1px solid rgba(124,58,237,0.4);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        text-align: center;
    }
    .hero-banner h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 0; }
    .hero-banner p  { font-size: 1.05rem; color: #c4b5fd; margin: 8px 0 0; }

    /* Step badge */
    .step-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 6px;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(124,58,237,0.5);
        border-radius: 12px;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 28px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #c4b5fd;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
        color: white !important;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }

    /* Info/success/warning boxes */
    .stAlert { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🏦 SBA 7(a) Loan Default Predictor</h1>
        <p>End-to-end ML pipeline · Random Forest & XGBoost · Portfolio Risk Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Navigation & File Upload
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload SBA 7(a) CSV file",
        type=["csv"],
        help="Upload the FOIA 7(a) dataset (e.g. foia-7a-fy2010-fy2019-asof-260331.csv)",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    run_rf = st.checkbox("Train Random Forest", value=True)
    run_xgb = st.checkbox("Train XGBoost (base)", value=True)
    run_tuned = st.checkbox("Train Tuned XGBoost", value=False)
    st.caption("⚠️ Tuned XGBoost runs RandomizedSearchCV — may take several minutes.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#9ca3af'>SBA FY2010–FY2019 · sklearn · XGBoost</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Status labels / helpers
# ─────────────────────────────────────────────────────────────────────────────
PIF_STATUSES = ["P I F", "CLSLN"]
DEFAULT_STATUSES = ["CHGOFF", "LIQUID", "PURCH(NOT C/O)"]
NON_PERFORMING = ["DELINQ", "PSTDUE", "DEFERD"]
OUTSTANDING_STATUSES = ["CURR", "COMMIT", "SOLDNC"]

NAICS_MAP = {
    "11": "Agriculture, Forestry, Fishing & Hunting",
    "21": "Mining, Quarrying, Oil & Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade", "45": "Retail Trade",
    "48": "Transportation & Warehousing", "49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate, Rental & Leasing",
    "54": "Professional, Scientific & Technical Services",
    "55": "Management of Companies & Enterprises",
    "56": "Admin, Support & Waste Management",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts, Entertainment & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services (except Public Admin)",
    "92": "Public Administration",
}

NUMERIC_FEATURES = [
    "grossapproval", "sbaguaranteedapproval", "initialinterestrate",
    "terminmonths", "loan_age_days", "disbursement_delay_days",
    "jobssupported", "sba_guarantee_pct", "capital_per_job",
    "approvalfy", "approval_month",
]
CATEGORICAL_FEATURES = [
    "naics_sector", "projectstate", "projectcounty", "processingmethod",
    "subprogram", "fixedorvariableinterestind", "businesstype",
    "businessage", "soldsecmrktind",
]
PASSTHROUGH_FEATURES = ["collateralind", "revolverstatus", "is_out_of_state_lender", "is_franchise"]

COLUMNS_TO_DROP = [
    "asofdate", "approvaldate", "firstdisbursementdate",
    "loanstatus", "paidinfulldate", "chargeoffdate", "grosschargeoffamount",
    "borrname", "borrstreet", "borrcity", "borrstate", "borrzip",
    "bankname", "bankstreet", "bankcity", "bankstate", "bankzip",
    "bankfdicnumber", "bankncuanumber",
    "naicscode", "naicsdescription", "franchisename", "franchisecode",
    "program", "locationid", "sbadistrictoffice", "congressionaldistrict",
    "portfolio_category",
]


# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline functions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Drop soft duplicates (keep latest asofdate)
    core = ["borrname", "bankname", "approvaldate", "grossapproval"]
    df = df.sort_values("asofdate")
    df = df.drop_duplicates(subset=core, keep="last")

    # Drop cancelled loans
    df = df[df["loanstatus"] != "CANCLD"].copy()

    # Parse dates
    for col in ["asofdate", "approvaldate", "firstdisbursementdate", "paidinfulldate", "chargeoffdate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def build_portfolio_summary(_df: pd.DataFrame):
    is_pif  = _df["loanstatus"].isin(PIF_STATUSES)
    is_def  = _df["loanstatus"].isin(DEFAULT_STATUSES)
    is_out  = _df["loanstatus"].isin(OUTSTANDING_STATUSES)
    is_np   = _df["loanstatus"].isin(NON_PERFORMING)

    totals = {
        "Paid in Full": (_df[is_pif]["grossapproval"].sum(),  is_pif.sum()),
        "Default":      (_df[is_def]["grossapproval"].sum(),  is_def.sum()),
        "Non Performing": (_df[is_np]["grossapproval"].sum(), is_np.sum()),
        "Outstanding":  (_df[is_out]["grossapproval"].sum(),  is_out.sum()),
    }
    overall = _df["grossapproval"].sum()

    rows = []
    for name, (amt, cnt) in totals.items():
        rows.append({"Category": name, "Loan Count": cnt,
                     "Total Amount ($)": amt, "% of Portfolio": amt / overall * 100})
    rows.append({"Category": "Total", "Loan Count": len(_df),
                 "Total Amount ($)": overall, "% of Portfolio": 100.0})

    summary = pd.DataFrame(rows).set_index("Category")
    return summary, totals, overall


@st.cache_data(show_spinner=False)
def build_state_summary(_df: pd.DataFrame):
    def _cat(s):
        if s in PIF_STATUSES:        return "Paid in Full"
        if s in DEFAULT_STATUSES:    return "Default"
        if s in NON_PERFORMING:      return "Non Performing"
        if s in OUTSTANDING_STATUSES: return "Outstanding"
        return "Unknown"

    df2 = _df.copy()
    df2["portfolio_category"] = df2["loanstatus"].apply(_cat)

    pivot = df2.groupby(["projectstate", "portfolio_category"]).agg(
        Loan_Count=("grossapproval", "count"),
        Total_Amount_USD=("grossapproval", "sum"),
    ).unstack(fill_value=0)
    pivot.columns = [f"{c[1]}_{c[0]}" for c in pivot.columns]

    state_cnt = df2.groupby("projectstate")["grossapproval"].count()
    state_amt = df2.groupby("projectstate")["grossapproval"].sum()

    for cat in ["Paid in Full", "Default", "Non Performing", "Outstanding"]:
        pivot[f"{cat}_Count_%"] = (pivot.get(f"{cat}_Loan_Count", 0) / state_cnt) * 100
        pivot[f"{cat}_Dollar_%"] = (pivot.get(f"{cat}_Total_Amount_USD", 0) / state_amt) * 100

    pivot["Total_State_Loans"]      = state_cnt
    pivot["Total_State_Amount_USD"] = state_amt

    return pivot.sort_values("Default_Dollar_%", ascending=False).reset_index()


@st.cache_data(show_spinner=False)
def engineer_features(_df: pd.DataFrame):
    df2 = _df.copy()

    df2["loan_age_days"]          = (df2["asofdate"] - df2["firstdisbursementdate"]).dt.days
    df2["disbursement_delay_days"]= (df2["firstdisbursementdate"] - df2["approvaldate"]).dt.days
    df2["approval_month"]         = df2["approvaldate"].dt.month
    df2["sba_guarantee_pct"]      = df2["sbaguaranteedapproval"] / df2["grossapproval"]
    df2["capital_per_job"]        = df2["grossapproval"] / (df2["jobssupported"] + 1)
    df2["naics_sector"]           = df2["naicscode"].fillna(0).astype(int).astype(str).str[:2].replace("0", "Unknown")
    df2["is_out_of_state_lender"] = (df2["bankstate"] != df2["projectstate"]).astype(int)
    df2["is_franchise"]           = df2["franchisecode"].notna().astype(int)

    # Target
    df2["is_default"] = np.nan
    df2.loc[df2["loanstatus"].isin(DEFAULT_STATUSES + NON_PERFORMING), "is_default"] = 1
    df2.loc[df2["loanstatus"].isin(PIF_STATUSES + OUTSTANDING_STATUSES), "is_default"] = 0

    # Drop rows without a defined target
    df2 = df2.dropna(subset=["is_default"])

    X = df2.drop(columns=[c for c in COLUMNS_TO_DROP if c in df2.columns])

    # Impute
    X["loan_age_days"]           = X["loan_age_days"].fillna(X["loan_age_days"].median())
    X["disbursement_delay_days"] = X["disbursement_delay_days"].fillna(X["disbursement_delay_days"].median())
    X["soldsecmrktind"]          = X["soldsecmrktind"].fillna("N")
    X["businesstype"]            = X["businesstype"].fillna("Unknown")
    X["businessage"]             = X["businessage"].fillna("Unknown")

    y = X.pop("is_default").astype(int)  # ensure int keys in classification_report
    return X, y


def build_preprocessor():
    return ColumnTransformer(
        transformers=[("cat", TargetEncoder(cv=5, random_state=42), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )


@st.cache_resource(show_spinner=False)
def train_rf(_X_train, _y_train):
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(_X_train, _y_train)
    return pipe


@st.cache_resource(show_spinner=False)
def train_xgb(_X_train, _y_train):
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=11, random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(_X_train, _y_train)
    return pipe


@st.cache_resource(show_spinner=False)
def train_tuned_xgb(_X_train, _y_train):
    base = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(scale_pos_weight=11, random_state=42, n_jobs=-1)),
    ])
    params = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [4, 6, 8],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9],
    }
    search = RandomizedSearchCV(base, params, n_iter=10, scoring="roc_auc",
                                cv=3, random_state=42, n_jobs=-1)
    search.fit(_X_train, _y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


# ─────────────────────────────────────────────────────────────────────────────
# Safe metric extractor — handles any key format sklearn may produce
# ─────────────────────────────────────────────────────────────────────────────
def _get_class_metrics(report: dict, metric: str) -> float:
    """Try every possible key sklearn uses for the positive class."""
    for key in [1, 1.0, "1", "1.0", True]:
        if key in report and metric in report[key]:
            return report[key][metric]
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def show_model_results(label: str, pipeline, X_test, y_test, color: str):
    y_pred  = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_probs)
    y_test_int = y_test.astype(int)  # normalise to int so report keys are always 0/1
    report  = classification_report(y_test_int, y_pred.astype(int), output_dict=True)
    cm      = confusion_matrix(y_test_int, y_pred.astype(int))

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",  f"{auc:.4f}")
    c2.metric("Precision (Default)", f"{_get_class_metrics(report, 'precision'):.3f}")
    c3.metric("Recall (Default)",    f"{_get_class_metrics(report, 'recall'):.3f}")
    c4.metric("F1 (Default)",        f"{_get_class_metrics(report, 'f1-score'):.3f}")

    col_a, col_b = st.columns(2)

    # Confusion matrix heatmap
    with col_a:
        st.markdown("**Confusion Matrix**")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Default", "Default"],
            y=["Not Default", "Default"],
            color_continuous_scale=color,
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    with col_b:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{label} (AUC={auc:.4f})",
            line=dict(color="#a78bfa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.15)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Full classification report table
    with st.expander("📋 Full Classification Report"):
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    return y_probs


# ─────────────────────────────────────────────────────────────────────────────
# Optimal threshold section
# ─────────────────────────────────────────────────────────────────────────────
def show_threshold_analysis(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    optimal_threshold = thresholds[best_idx]

    st.info(f"**Mathematically optimal threshold (G-Mean / J-Statistic): `{optimal_threshold:.4f}`**")

    custom_preds = (y_probs >= optimal_threshold).astype(int)
    y_test_int   = np.array(y_test).astype(int)  # normalise to int so report keys are always 0/1
    report = classification_report(y_test_int, custom_preds, output_dict=True)
    cm     = confusion_matrix(y_test_int, custom_preds)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision (Default)", f"{_get_class_metrics(report, 'precision'):.3f}")
    c2.metric("Recall (Default)",    f"{_get_class_metrics(report, 'recall'):.3f}")
    c3.metric("F1 (Default)",        f"{_get_class_metrics(report, 'f1-score'):.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Custom Threshold Confusion Matrix**")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Default", "Default"],
            y=["Not Default", "Default"],
            color_continuous_scale="Purp",
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_b:
        st.markdown("**G-Mean by Threshold**")
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=thresholds, y=gmeans, mode="lines",
            line=dict(color="#34d399", width=2),
        ))
        fig_g.add_vline(x=optimal_threshold, line_dash="dash",
                        line_color="#f472b6",
                        annotation_text=f"Optimal: {optimal_threshold:.3f}",
                        annotation_font_color="#f472b6")
        fig_g.update_layout(
            xaxis_title="Threshold", yaxis_title="G-Mean",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    return optimal_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Main App Flow
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 20px;'>
            <div style='font-size:4rem'>📤</div>
            <h3 style='color:#a78bfa; margin-top:12px'>Upload your SBA 7(a) CSV to get started</h3>
            <p style='color:#9ca3af'>Use the sidebar to upload the FOIA 7(a) dataset CSV file.</p>
            <p style='color:#6b7280; font-size:0.85rem'>Expected: <code>foia-7a-fy2010-fy2019-asof-*.csv</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Load & Clean ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading and cleaning data…"):
    file_bytes = uploaded_file.read()
    df_clean = load_and_clean(file_bytes)

st.success(f"✅ Data loaded: **{len(df_clean):,} loans** after deduplication and filtering.")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_portfolio, tab_geo, tab_model, tab_features = st.tabs([
    "📊 Portfolio Overview",
    "🗺️ Geographic Risk",
    "🤖 ML Models",
    "📈 Feature Importance",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown('<div class="section-header">Portfolio Summary (FY2010–FY2019)</div>', unsafe_allow_html=True)

    summary, totals, overall = build_portfolio_summary(df_clean)

    # KPI row
    kpi_cols = st.columns(4)
    colors_kpi = {"Paid in Full": "#4ade80", "Default": "#f87171",
                  "Non Performing": "#fb923c", "Outstanding": "#facc15"}
    for col, (name, (amt, cnt)) in zip(kpi_cols, totals.items()):
        col.metric(
            label=name,
            value=f"${amt/1e9:.2f}B",
            delta=f"{cnt:,} loans",
            delta_color="off",
        )

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("**Portfolio Breakdown Table**")
        fmt_summary = summary.copy()
        fmt_summary["Total Amount ($)"] = fmt_summary["Total Amount ($)"].apply("${:,.0f}".format)
        fmt_summary["% of Portfolio"]   = fmt_summary["% of Portfolio"].apply("{:.2f}%".format)
        fmt_summary["Loan Count"]       = fmt_summary["Loan Count"].apply("{:,}".format)
        st.dataframe(fmt_summary, use_container_width=True)

    with col_right:
        st.markdown("**Dollar Volume by Status**")
        labels = list(totals.keys())
        sizes  = [v[0] for v in totals.values()]
        pie_colors = ["#4ade80", "#f87171", "#fb923c", "#facc15"]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=sizes,
            marker=dict(colors=pie_colors, line=dict(color="#1e1b4b", width=2)),
            textinfo="percent+label",
            pull=[0, 0.1, 0.1, 0],
            hole=0.35,
        ))
        fig_pie.update_layout(
            title="SBA 7(a) Portfolio Breakdown",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Sector default rate chart
    st.markdown('<div class="section-header">Default Rates by Industry Sector</div>', unsafe_allow_html=True)

    df2 = df_clean.copy()
    df2.loc[df2["loanstatus"].isin(DEFAULT_STATUSES + NON_PERFORMING), "is_default"] = 1
    df2.loc[df2["loanstatus"].isin(PIF_STATUSES + OUTSTANDING_STATUSES), "is_default"] = 0
    df2["naics_sector"] = df2["naicscode"].fillna(0).astype(int).astype(str).str[:2].replace("0", "Unknown")
    df2["Sector_Description"] = df2["naics_sector"].map(NAICS_MAP)
    df2 = df2.dropna(subset=["Sector_Description"])

    sector_totals   = df2.groupby("Sector_Description")["grossapproval"].sum().reset_index(name="Total_Volume_USD")
    sector_defaults = df2[df2["is_default"] == 1].groupby("Sector_Description")["grossapproval"].sum().reset_index(name="Default_Volume_USD")
    sector_summary  = pd.merge(sector_totals, sector_defaults, on="Sector_Description", how="left").fillna(0)
    sector_summary["Default_Rate_Dollar_%"] = (sector_summary["Default_Volume_USD"] / sector_summary["Total_Volume_USD"]) * 100
    sector_summary  = sector_summary.sort_values("Default_Rate_Dollar_%", ascending=True).reset_index(drop=True)

    fig_sector = px.bar(
        sector_summary,
        x="Default_Rate_Dollar_%", y="Sector_Description",
        orientation="h",
        title="<b>Dollar Volume Default Rate by Industry Sector</b>",
        labels={"Sector_Description": "Industry Sector", "Default_Rate_Dollar_%": "Default Rate (%)"},
        color="Default_Rate_Dollar_%",
        color_continuous_scale="YlOrRd",
    )
    fig_sector.update_layout(
        title_x=0.5, title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis={"categoryorder": "array", "categoryarray": sector_summary["Sector_Description"].tolist()},
        coloraxis_showscale=False,
        margin=dict(l=340, r=20, t=60, b=40),
        height=650,
    )
    st.plotly_chart(fig_sector, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geographic Risk
# ═══════════════════════════════════════════════════════════════════════════════
with tab_geo:
    st.markdown('<div class="section-header">State-Level Default Risk</div>', unsafe_allow_html=True)

    with st.spinner("Building state-level analysis…"):
        state_data = build_state_summary(df_clean)

    # Choropleth maps
    fig_geo = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=("<b>Count Default Rate</b>", "<b>Dollar Default Rate</b>"),
    )
    for col_idx, metric, cbar_x in [
        (1, "Default_Count_%",  0.44),
        (2, "Default_Dollar_%", 1.02),
    ]:
        fig_geo.add_trace(
            go.Choropleth(
                locations=state_data["projectstate"],
                z=state_data[metric],
                locationmode="USA-states",
                colorscale="Reds",
                colorbar_x=cbar_x,
                colorbar_title="Rate %",
            ),
            row=1, col=col_idx,
        )
    fig_geo.update_geos(scope="usa")
    fig_geo.update_layout(
        title_text="<b>SBA Portfolio: Frequency vs. Severity by State</b>",
        title_font_size=18, title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)", font_color="white",
        height=520,
    )
    st.plotly_chart(fig_geo, use_container_width=True)

    # Table — top risky states
    st.markdown('<div class="section-header">Top 10 Highest Default Rate States</div>', unsafe_allow_html=True)
    disp_cols = ["projectstate", "Default_Count_%", "Default_Dollar_%",
                 "Total_State_Loans", "Total_State_Amount_USD"]
    disp = state_data[disp_cols].head(10).rename(columns={
        "projectstate":        "State",
        "Default_Count_%":     "Count Default %",
        "Default_Dollar_%":    "Dollar Default %",
        "Total_State_Loans":   "Total Loans",
        "Total_State_Amount_USD": "Total Volume ($)",
    })
    disp["Count Default %"]  = disp["Count Default %"].round(2)
    disp["Dollar Default %"] = disp["Dollar Default %"].round(2)
    disp["Total Volume ($)"] = disp["Total Volume ($)"].apply("${:,.0f}".format)
    st.dataframe(disp.set_index("State"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML Models
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">Machine Learning Pipeline</div>', unsafe_allow_html=True)

    with st.spinner("🔧 Engineering features…"):
        X, y = engineer_features(df_clean)

    st.info(f"**Feature matrix:** {X.shape[0]:,} rows × {X.shape[1]} columns | "
            f"**Default rate:** {y.mean()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_probs_for_threshold = None
    best_pipeline_for_download = None

    # ── Random Forest ────────────────────────────────────────────────────────
    if run_rf:
        st.markdown("---")
        st.markdown("### 🌲 Random Forest")
        with st.spinner("Training Random Forest (150 trees, max_depth=12)…"):
            rf_pipeline = train_rf(X_train, y_train)
        rf_probs = show_model_results("Random Forest", rf_pipeline, X_test, y_test, "Greens")

    # ── XGBoost (base) ───────────────────────────────────────────────────────
    if run_xgb:
        st.markdown("---")
        st.markdown("### ⚡ XGBoost (Base)")
        with st.spinner("Training XGBoost (200 rounds, lr=0.1)…"):
            xgb_pipeline = train_xgb(X_train, y_train)
        xgb_probs = show_model_results("XGBoost", xgb_pipeline, X_test, y_test, "Blues")
        best_probs_for_threshold = xgb_probs
        best_pipeline_for_download = xgb_pipeline

    # ── Tuned XGBoost ────────────────────────────────────────────────────────
    if run_tuned:
        st.markdown("---")
        st.markdown("### 🔬 Tuned XGBoost (RandomizedSearchCV)")
        with st.spinner("Running hyperparameter search (this may take several minutes)…"):
            tuned_pipe, best_params, best_cv_score = train_tuned_xgb(X_train, y_train)
        st.success(f"Best CV ROC-AUC: **{best_cv_score:.4f}**")
        with st.expander("Best Hyperparameters"):
            st.json({k.replace("classifier__", ""): v for k, v in best_params.items()})
        tuned_probs = show_model_results("Tuned XGBoost", tuned_pipe, X_test, y_test, "Purp")
        best_probs_for_threshold = tuned_probs
        best_pipeline_for_download = tuned_pipe

    # ── Threshold Analysis ───────────────────────────────────────────────────
    if best_probs_for_threshold is not None:
        st.markdown("---")
        st.markdown("### 🎯 Optimal Decision Threshold")
        opt_threshold = show_threshold_analysis(y_test, best_probs_for_threshold)

    # ── Download Model ───────────────────────────────────────────────────────
    if best_pipeline_for_download is not None:
        st.markdown("---")
        st.markdown("### 💾 Save Model")
        buffer = io.BytesIO()
        joblib.dump(best_pipeline_for_download, buffer)
        buffer.seek(0)
        st.download_button(
            label="⬇️ Download Trained Pipeline (.pkl)",
            data=buffer,
            file_name="sba_loan_xgboost_pipeline.pkl",
            mime="application/octet-stream",
        )

    if not run_rf and not run_xgb and not run_tuned:
        st.warning("Enable at least one model in the sidebar to train.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
with tab_features:
    st.markdown('<div class="section-header">XGBoost Feature Importance</div>', unsafe_allow_html=True)

    if not run_xgb:
        st.warning("Enable **Train XGBoost (base)** in the sidebar to see feature importance.")
    else:
        with st.spinner("Training XGBoost for feature importance…"):
            _X, _y = engineer_features(df_clean)
            _Xtr, _Xte, _ytr, _yte = train_test_split(_X, _y, test_size=0.2, stratify=_y, random_state=42)
            xgb_pipe = train_xgb(_Xtr, _ytr)

        trained_xgb = xgb_pipe.named_steps["classifier"]
        importance_df = pd.DataFrame({
            "Feature":    NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES,
            "Importance": trained_xgb.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            importance_df,
            x="Importance", y="Feature",
            orientation="h",
            title="<b>XGBoost Feature Importance (Gain)</b>",
            labels={"Importance": "Relative Importance (Gain)", "Feature": "Loan Feature"},
            color="Importance",
            color_continuous_scale="Viridis",
            height=700,
        )
        fig_imp.update_layout(
            title_x=0.5, title_font_size=18,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
            margin=dict(l=220, r=20, t=60, b=40),
            yaxis={"categoryorder": "array", "categoryarray": importance_df["Feature"].tolist()},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("📋 Raw Importance Scores"):
            st.dataframe(
                importance_df.sort_values("Importance", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
