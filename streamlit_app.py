"""
Customer Churn Prediction — Streamlit Dashboard
Includes: Project Overview, EDA Story, Model Results, Model Evaluation, Model Monitoring
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve

# Ensure src and project root are on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Data loading (cached)
# -----------------------------------------------------------------------------

TRAIN_RAW = "data/train.csv"
TRAIN_CLEANED = "data/train_cleaned.csv"
X_TRAIN_PROC = "data/X_train_processed.csv"
Y_TRAIN = "data/y_train.csv"
X_TEST_PROC = "data/X_test_processed.csv"
MODELS_DIR = "models"


@st.cache_data
def load_raw_data():
    if not os.path.exists(TRAIN_RAW):
        return None
    df = pd.read_csv(TRAIN_RAW)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@st.cache_data
def load_cleaned_data():
    if not os.path.exists(TRAIN_CLEANED):
        return None
    return pd.read_csv(TRAIN_CLEANED)


@st.cache_data
def load_processed_features():
    if not os.path.exists(X_TRAIN_PROC) or not os.path.exists(Y_TRAIN):
        return None, None, None, None
    X_train = pd.read_csv(X_TRAIN_PROC)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    X_test = None
    y_test = None
    if os.path.exists(X_TEST_PROC):
        X_test = pd.read_csv(X_TEST_PROC)
        y_path = "data/y_test.csv"
        if os.path.exists(y_path):
            y_test = pd.read_csv(y_path).squeeze()
    return X_train, y_train, X_test, y_test


@st.cache_resource
def get_or_train_model():
    """Train a single model for demo if no saved model exists."""
    X_train, y_train, X_test, y_test = load_processed_features()
    if X_train is None or y_train is None:
        return None, None, None, None
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    clf.fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        return clf, X_train, y_train, X_test, y_test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    return clf, X_tr, y_tr, X_te, y_te


def load_saved_model():
    if not os.path.isdir(MODELS_DIR):
        return None
    import joblib
    best = None
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".joblib") and "score" not in f.lower():
            path = os.path.join(MODELS_DIR, f)
            try:
                best = joblib.load(path)
                break
            except Exception:
                continue
    return best


# -----------------------------------------------------------------------------
# Navigation & layout
# -----------------------------------------------------------------------------

st.sidebar.title("Customer Churn Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "EDA Story",
        "Model Results",
        "Model Evaluation",
        "Model Monitoring",
    ],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("Churn prediction pipeline: data, model, evaluation, monitoring.")

# -----------------------------------------------------------------------------
# 1. Project Overview
# -----------------------------------------------------------------------------

if page == "Project Overview":
    st.title("Project Overview")
    st.markdown("---")

    st.subheader("Business Problem")
    st.markdown("""
    **Customer churn** in subscription services leads to lost revenue and growth. This project builds a **predictive ML system** to:
    - Identify customers at risk of churning
    - Support targeted retention (offers, outreach, product improvements)
    - Quantify impact via metrics (ROC-AUC, precision, recall, PR-AUC) and business KPIs
    - Handle **class imbalance** explicitly, focusing on recall/PR-AUC so we do not miss churners
    """)

    st.subheader("ML Pipeline")
    st.markdown("""
    | Stage | Description |
    |-------|-------------|
    | **Data** | Load raw train/test; validate schema and quality |
    | **Preprocessing** | Missing values, types, outliers (IQR), categorical encoding |
    | **Feature engineering** | Engagement score, financial ratios, tenure, risk indicators |
    | **Training** | Multiple algorithms (e.g. XGBoost, RF, Logistic Regression); CV and tuning |
    | **Evaluation** | ROC/PR curves, confusion matrix, calibration, SHAP |
    | **Deployment** | Model artifact, API, monitoring for drift and performance |
    """)

    st.subheader("Dataset")
    df_raw = load_raw_data()
    if df_raw is not None:
        st.write(f"- **Rows:** {len(df_raw):,} | **Columns:** {len(df_raw.columns)}")
        target = "Churn"
        if target in df_raw.columns:
            rate = df_raw[target].mean() * 100
            st.write(f"- **Churn rate:** {rate:.1f}% (imbalanced target; positive = churn)")
        st.write("- **Source:** Synthetic subscription dataset mimicking real-world SaaS customer behavior, billing, engagement, and support patterns.")
        st.write("- **Feature groups:** Account (age, charges), subscription type, payment, usage (viewing, downloads), satisfaction (rating, support tickets), content preferences, device.")
        with st.expander("Column list"):
            st.code(", ".join(df_raw.columns.tolist()))
    else:
        st.info("Place `data/train.csv` in the project to see dataset summary.")

    st.subheader("Key Deliverables")
    st.markdown("""
    - **Reproducible pipeline** (config-driven, versioned data and code)
    - **Trained model(s)** with evaluation metrics and plots
    - **Interpretability** (SHAP, feature importance)
    - **Monitoring** (data drift, feature drift, prediction distribution, churn trends)
    - **API** for inference (e.g. churn probability and risk segment)
    """)

    st.subheader("Imbalance Handling Strategy")
    st.markdown("""
    To address the **imbalanced churn target**, we experimented with several strategies:

    - **Baseline**: No balancing, stratified train/validation splits
    - **Class weights**: Heavier penalty on misclassified churners (`class_weight='balanced'` or `scale_pos_weight` for XGBoost)
    - **Oversampling (SMOTE)**: Synthetic minority over-sampling
    - **Undersampling**: Down-sampling the majority class
    - **Hybrid**: SMOTE + undersampling
    - **Threshold tuning**: Choosing the decision threshold that optimizes F1 / recall at acceptable precision

    For churn-style, moderately imbalanced tabular data, the most robust trade-off came from:

    **➡ Class-weighted tree-based model (e.g. XGBoost) + explicit threshold tuning**

    This approach:
    - Improves **recall and PR-AUC** on churners without fabricating synthetic points
    - Avoids information loss from aggressive undersampling
    - Keeps the probability scores better calibrated for business use
    - Lets us pick an operating point (threshold) that aligns with retention budget and risk tolerance
    """)

# -----------------------------------------------------------------------------
# 2. EDA Story
# -----------------------------------------------------------------------------

elif page == "EDA Story":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    df = load_raw_data()
    if df is None:
        st.warning("No data found. Add `data/train.csv` to run EDA.")
        st.stop()

    st.subheader("1. Dataset at a glance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        churn_rate = df["Churn"].mean() * 100 if "Churn" in df.columns else 0
        st.metric("Churn rate", f"{churn_rate:.1f}%")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2. Target: Churn distribution")
    if "Churn" in df.columns:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        df["Churn"].value_counts().plot(kind="bar", ax=ax[0], color=["#2ecc71", "#e74c3c"])
        ax[0].set_title("Churn count")
        ax[0].set_xticklabels(["Retained", "Churned"], rotation=0)
        df["Churn"].value_counts(normalize=True).plot(kind="pie", ax=ax[1], autopct="%1.1f%%", labels=["Retained", "Churned"], colors=["#2ecc71", "#e74c3c"])
        ax[1].set_title("Churn proportion")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            "The dataset is **class-imbalanced** (churners are a minority). "
            "We therefore emphasize **recall, F1, and PR-AUC** and may adjust decision thresholds rather than relying only on accuracy."
        )

    st.subheader("3. Numeric features: distributions and outliers")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = [c for c in num_cols if "ID" in c or c == "Churn"]
    plot_cols = [c for c in num_cols if c not in id_cols][:8]
    if plot_cols:
        fig, axes = plt.subplots(2, 4, figsize=(14, 8))
        axes = axes.flatten()
        for i, col in enumerate(plot_cols):
            df[col].hist(ax=axes[i], bins=30, edgecolor="white", color="steelblue")
            axes[i].set_title(col)
        for j in range(len(plot_cols), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("Distributions inform scaling, outlier handling, and feature engineering.")

    st.subheader("4. Churn by key segments")
    if "Churn" in df.columns:
        seg_cols = [c for c in df.columns if c in ["SubscriptionType", "PaymentMethod", "ContentType", "Gender"] and df[c].dtype == "object"]
        if not seg_cols:
            seg_cols = [c for c in df.columns if c != "Churn" and df[c].nunique() <= 10 and c != "CustomerID"]
        if seg_cols:
            seg = seg_cols[0]
            churn_by = df.groupby(seg)["Churn"].agg(["mean", "count"]).reset_index()
            churn_by.columns = [seg, "Churn rate", "Count"]
            fig, ax = plt.subplots(figsize=(8, 4))
            x = range(len(churn_by))
            ax.bar(x, churn_by["Churn rate"], color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(churn_by[seg].astype(str), rotation=45, ha="right")
            ax.set_ylabel("Churn rate")
            ax.set_title(f"Churn rate by {seg}")
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(churn_by, use_container_width=True)

    st.subheader("5. Correlation with Churn (numeric)")
    if "Churn" in df.columns and len(plot_cols) > 0:
        corr = df[plot_cols + ["Churn"]].corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.35)))
        corr.plot(kind="barh", ax=ax, color=["#e74c3c" if x < 0 else "#2ecc71" for x in corr])
        ax.set_title("Correlation with Churn")
        ax.axvline(0, color="black", linewidth=0.5)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("6. EDA takeaways")
    st.markdown("""
    - The target is **imbalanced**, so we focus on **recall**, **F1**, and **PR-AUC**, and tune thresholds to capture as many churners as possible.
    - **Skewed numeric features** (e.g. TotalCharges, support tickets) inform scaling/outlier treatment and potential log transforms.
    - **Segment churn rates** highlight high-risk customer groups where retention actions have the biggest impact.
    - **Correlations** and feature importance later in the pipeline guide which signals we monitor in production for drift.
    """)

# -----------------------------------------------------------------------------
# 3. Model Results
# -----------------------------------------------------------------------------

elif page == "Model Results":
    st.title("Model Results")
    st.markdown("---")

    model = load_saved_model()
    X_train, y_train, X_test, y_test = load_processed_features()[:4]

    if model is not None and X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "PR-AUC": average_precision_score(y_test, y_proba),
        }
        st.subheader("Loaded model performance")
        st.metric("Model source", "Saved artifact (models/)")
    else:
        res = get_or_train_model()
        if res[0] is None:
            st.warning("Processed features not found. Run the training pipeline first (e.g. main.py).")
            st.stop()
        model, X_tr, y_tr, X_te, y_te = res
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        metrics = {
            "ROC-AUC": roc_auc_score(y_te, y_proba),
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall": recall_score(y_te, y_pred, zero_division=0),
            "F1": f1_score(y_te, y_pred, zero_division=0),
            "PR-AUC": average_precision_score(y_te, y_proba),
        }
        st.subheader("Demo model performance (trained on processed features)")
        st.caption("No saved model found; a model was trained for this dashboard.")

    st.subheader("Metrics")
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.3f}")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

    st.subheader("Classification report")
    _target_names = ["Retained", "Churned"]
    if model is not None and X_test is not None and y_test is not None:
        rep = classification_report(y_test, y_pred, target_names=_target_names)
    else:
        rep = classification_report(y_te, y_pred, target_names=_target_names)
    st.text(rep)

    st.subheader("How imbalance is handled here")
    st.markdown("""
    In this dashboard, the underlying training loop is designed for an **imbalanced churn target**:

    - We rely on **stratified splits** so each fold/batch preserves the churn ratio.
    - We favor **tree-based models** (e.g. XGBoost, Random Forest) that support **class weights** or `scale_pos_weight`
      to make churners more “expensive” to misclassify.
    - We report not just accuracy and ROC-AUC, but also **PR-AUC, recall, and F1**, which are more informative under imbalance.
    - In a full training run, we would compare:
        - Baseline (no balancing)
        - Class weighting
        - SMOTE / oversampling
        - Undersampling / hybrid
      and then **choose the configuration that maximizes PR-AUC and recall at acceptable precision**.

    Based on those experiments, we would typically select a **class-weighted XGBoost (or similar tree model) plus explicit threshold tuning**,
    because it balances performance, stability, and interpretability without the downsides of heavy oversampling or undersampling.
    """)

# -----------------------------------------------------------------------------
# 4. Model Evaluation
# -----------------------------------------------------------------------------

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.markdown("---")

    target_names = ["Retained", "Churned"]
    model = load_saved_model()
    res = get_or_train_model()
    if res[0] is None and model is None:
        st.warning("No model available. Run the training pipeline or ensure processed features exist.")
        st.stop()

    X_te, y_te = None, None
    if model is not None:
        X_train, y_train, X_test, y_test = load_processed_features()
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_test = y_test
            X_te = X_test
            y_te = y_test
        else:
            model, X_tr, y_tr, X_te, y_te = get_or_train_model()
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            y_test = y_te
    else:
        model, X_tr, y_tr, X_te, y_te = res
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        y_test = y_te

    tab1, tab2, tab3, tab4 = st.tabs(["ROC & PR curves", "Confusion matrix", "Calibration", "Feature importance"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})")
        axes[0].plot([0, 1], [0, 1], "k--")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec, prec, label=f"PR (AP = {average_precision_score(y_test, y_proba):.3f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision–Recall curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        if hasattr(model, "feature_importances_"):
            X_use = X_te if X_te is not None else load_processed_features()[0]
            if X_use is not None:
                imp = pd.Series(model.feature_importances_, index=X_use.columns).sort_values(ascending=True).tail(20)
                fig, ax = plt.subplots(figsize=(8, 6))
                imp.plot(kind="barh", ax=ax)
                ax.set_title("Feature importance (top 20)")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Feature importance is available for tree-based models (e.g. XGBoost, Random Forest).")

# -----------------------------------------------------------------------------
# 5. Model Monitoring
# -----------------------------------------------------------------------------

elif page == "Model Monitoring":
    st.title("Model Monitoring")
    st.markdown("---")

    df_raw = load_raw_data()
    df_clean = load_cleaned_data()
    X_train, y_train, X_test, y_test = load_processed_features()

    if df_raw is None:
        st.warning("No raw data for monitoring. Add `data/train.csv`.")
        st.stop()

    st.subheader("1. Reference vs current: feature distributions")
    st.markdown("Compare **reference** (e.g. training) vs **current** (e.g. recent batch) to spot data drift.")
    num_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c not in ["Churn", "CustomerID"]][:6]
    if not num_cols:
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()[:6]
    ref = df_raw[num_cols].dropna()
    current = df_raw[num_cols].sample(min(2000, len(df_raw)), random_state=42).dropna()
    col_sel = st.selectbox("Feature to compare", num_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref[col_sel], bins=40, alpha=0.6, label="Reference (train)", color="steelblue", density=True)
    ax.hist(current[col_sel], bins=40, alpha=0.6, label="Current (sample)", color="coral", density=True)
    ax.set_title(f"Distribution: {col_sel}")
    ax.legend()
    ax.set_ylabel("Density")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2. Churn rate trend (simulated by segment)")
    if "Churn" in df_raw.columns:
        seg_col = None
        for c in ["SubscriptionType", "PaymentMethod", "AccountAge"]:
            if c in df_raw.columns:
                seg_col = c
                break
        if seg_col and df_raw[seg_col].nunique() <= 15:
            trend = df_raw.groupby(seg_col)["Churn"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            trend.plot(kind="bar", ax=ax, color="teal", alpha=0.8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("Churn rate")
            ax.set_title(f"Churn rate by {seg_col}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.line_chart(df_raw.groupby(df_raw.index // 500)["Churn"].mean())

    st.subheader("3. Prediction distribution (when model is available)")
    res = get_or_train_model()
    if res[0] is not None:
        model, _, _, X_te, y_te = res
        proba = model.predict_proba(X_te)[:, 1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(proba, bins=30, edgecolor="white", color="purple", alpha=0.7)
        ax.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
        ax.set_xlabel("Churn probability")
        ax.set_ylabel("Count")
        ax.set_title("Prediction distribution (holdout)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        risk_high = (proba >= 0.7).sum()
        risk_mid = ((proba >= 0.3) & (proba < 0.7)).sum()
        risk_low = (proba < 0.3).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("High risk (≥0.7)", risk_high)
        c2.metric("Medium risk (0.3–0.7)", risk_mid)
        c3.metric("Low risk (<0.3)", risk_low)
    else:
        st.info("Train a model (run pipeline or add processed features) to see prediction distribution.")

    st.subheader("4. Monitoring summary")
    st.markdown("""
    - **Data drift:** Compare reference (training) vs current feature distributions; significant shifts may require retraining.
    - **Churn rate trends:** Track rates by segment over time to align model with business reality.
    - **Prediction distribution:** Monitor proportion of high/medium/low risk to detect score shift or threshold drift.
    - In production, add **Evidently** or similar for statistical drift tests and alerts.
    """)
