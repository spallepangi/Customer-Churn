import os
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.clustering import (
    basic_clean,
    engineer_behavior_features,
    build_feature_matrix,
    scale_and_project,
    elbow_scores,
    silhouette_scores_k,
    fit_kmeans,
    fit_dbscan,
    profile_clusters,
    export_cluster_profiles_csv,
)

st.set_page_config(page_title="Superstore Customer Segmentation", layout="wide")

# Sidebar - Data source
st.sidebar.header("Data Source")
options = ["Use bundled sample (data/train.csv)", "Upload CSV"]
if os.path.exists("superstore_data.csv"):
    options.insert(0, "Use superstore_data.csv")
option = st.sidebar.radio("Choose data:", options)

@st.cache_data
def load_sample_data():
    path = os.path.join("data", "train.csv")
    df = pd.read_csv(path)
    return df

uploaded_df = None
if option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"]) 
    if file is not None:
        uploaded_df = pd.read_csv(file)

# Main title
st.title("Superstore Customer Segmentation")
st.markdown("Identify meaningful customer segments from demographics and purchasing behavior to inform marketing strategies.")

# Load data
if option == "Use superstore_data.csv":
    raw_df = pd.read_csv("superstore_data.csv")
elif option == "Use bundled sample (data/train.csv)":
    if not os.path.exists("data/train.csv"):
        st.error("data/train.csv not found. Please upload a CSV instead.")
        st.stop()
    raw_df = load_sample_data()
else:
    if uploaded_df is None:
        st.info("Upload a CSV to begin.")
        st.stop()
    raw_df = uploaded_df

st.subheader("Preview")
st.dataframe(raw_df.head(10))

# Cleaning and feature engineering
with st.expander("Step 1 - Prepare data", expanded=True):
    df_clean = basic_clean(raw_df)
    df_fe = engineer_behavior_features(df_clean)

    st.write("Rows:", len(df_fe))
    st.write("Columns:", len(df_fe.columns))

    include_cats = st.checkbox("Include categorical variables (one-hot)", value=False)
    X, features = build_feature_matrix(df_fe, include_categorical=include_cats)

    st.write("Feature matrix shape:", X.shape)

# Scaling and PCA for visualization
with st.expander("Step 2 - Scale and project", expanded=True):
    n_pca = st.slider("PCA components for visualization", min_value=0, max_value=3, value=2)
    Xs, scaler, pca_model = scale_and_project(X, n_components=n_pca)
    if n_pca >= 2 and pca_model is not None:
        st.write("Explained variance ratio:", getattr(pca_model, "explained_variance_ratio_", None))

# Clustering controls
st.subheader("Clustering")
method = st.selectbox("Algorithm", ["KMeans", "DBSCAN"], index=0)

if method == "KMeans":
    k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=4)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Elbow (2-12)"):
            scores = elbow_scores(Xs, k_min=2, k_max=12)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(scores["k"], scores["inertia"], marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia (SSE)")
            ax.set_title("Elbow method")
            st.pyplot(fig)
    with col2:
        if st.button("Run Silhouette (2-12)"):
            ks = list(range(2, 13))
            sils = silhouette_scores_k(Xs, ks)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(sils["k"], sils["silhouette"], marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette score")
            ax.set_title("Silhouette vs k")
            st.pyplot(fig)

    if st.button("Cluster with KMeans"):
        result = fit_kmeans(Xs, k=k, pca_model=pca_model, scaler=scaler)
        labels = result.labels
        st.success(f"KMeans done. Silhouette: {result.silhouette if result.silhouette is not None else 'NA'}")
        # Plot in PCA space
        if result.pca_components is not None:
            fig, ax = plt.subplots(figsize=(5,4))
            scatter = ax.scatter(result.pca_components[:,0], result.pca_components[:,1], c=labels, cmap="tab10", s=15, alpha=0.8)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Clusters (PCA space)")
            st.pyplot(fig)
        # Profiling
        summary, labeled = profile_clusters(df_fe, Xs, labels)
        st.subheader("Cluster Summary")
        st.dataframe(summary)
        st.subheader("Labeled Customers (head)")
        st.dataframe(labeled.head())
        # Downloads
        b1, b2 = export_cluster_profiles_csv(summary, labeled)
        st.download_button("Download Cluster Summary CSV", data=b1, file_name="cluster_summary.csv")
        st.download_button("Download Labeled Customers CSV", data=b2, file_name="customers_with_clusters.csv")

else:
    eps = st.slider("eps", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
    min_samples = st.slider("min_samples", min_value=3, max_value=50, value=10)
    if st.button("Cluster with DBSCAN"):
        result = fit_dbscan(Xs, eps=eps, min_samples=min_samples, pca_model=pca_model, scaler=scaler)
        labels = result.labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.success(f"DBSCAN done. Clusters: {n_clusters}. Silhouette: {result.silhouette if result.silhouette is not None else 'NA'}")
        if result.pca_components is not None:
            fig, ax = plt.subplots(figsize=(5,4))
            scatter = ax.scatter(result.pca_components[:,0], result.pca_components[:,1], c=labels, cmap="tab20", s=15, alpha=0.8)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("DBSCAN clusters (PCA space)")
            st.pyplot(fig)
        summary, labeled = profile_clusters(df_fe, Xs, labels)
        st.subheader("Cluster Summary")
        st.dataframe(summary)
        st.subheader("Labeled Customers (head)")
        st.dataframe(labeled.head())
        b1, b2 = export_cluster_profiles_csv(summary, labeled)
        st.download_button("Download Cluster Summary CSV", data=b1, file_name="cluster_summary.csv")
        st.download_button("Download Labeled Customers CSV", data=b2, file_name="customers_with_clusters.csv")

