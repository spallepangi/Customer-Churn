"""
Clustering utilities for customer segmentation.

This module provides end-to-end preprocessing, feature engineering for segmentation,
modeling with K-Means and DBSCAN, evaluation helpers (elbow, silhouette), and
cluster profiling utilities to derive marketing insights.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


NUMERIC_IMPUTE_STRATEGY = "median"


@dataclass
class ClusteringResult:
    model_name: str
    labels: np.ndarray
    inertia_: Optional[float]
    silhouette: Optional[float]
    centroids: Optional[np.ndarray]
    pca_components: Optional[np.ndarray]
    pca_explained_variance: Optional[np.ndarray]
    features_used: List[str]
    scaler: StandardScaler
    pca_model: Optional[PCA]


def _infer_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "CustomerID", "customer_id", "Customer Id", "Customer_ID", "CustID",
        "ID", "Id", "user_id"
    ]
    return [c for c in candidates if c in df.columns]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # Convert obvious numeric columns stored as strings
    for col in df.columns:
        if df[col].dtype == object:
            # Try coercing if many values look numeric
            sample = df[col].dropna().astype(str).str.replace(",", "", regex=False)
            if len(sample) and sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.6:
                df[col] = pd.to_numeric(sample, errors="coerce")

    # Impute numeric missing
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if NUMERIC_IMPUTE_STRATEGY == "median":
            df[col] = df[col].fillna(df[col].median())
        elif NUMERIC_IMPUTE_STRATEGY == "mean":
            df[col] = df[col].fillna(df[col].mean())

    # Impute categorical missing with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def engineer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example generic features if present
    if {"OrderAmount", "OrderCount"}.issubset(df.columns):
        df["AvgOrderValue"] = (df["OrderAmount"] / (df["OrderCount"].replace(0, np.nan))).fillna(0)
    if {"TotalPurchaseAmount", "TotalOrders"}.issubset(df.columns):
        df["AvgOrderValue"] = (df["TotalPurchaseAmount"] / (df["TotalOrders"].replace(0, np.nan))).fillna(0)
    if {"ViewingHoursPerWeek", "AverageViewingDuration"}.issubset(df.columns):
        df["ViewingIntensity"] = (df["ViewingHoursPerWeek"] * 60) / (df["AverageViewingDuration"].replace(0, np.nan))
        df["ViewingIntensity"] = df["ViewingIntensity"].fillna(0)
    if {"MonthlyCharges", "TotalCharges", "AccountAge"}.issubset(df.columns):
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["AccountAge"].replace(0, np.nan))
        df["AvgMonthlySpend"] = df["AvgMonthlySpend"].replace([np.inf, -np.inf], np.nan).fillna(df["AvgMonthlySpend"].median())
    return df


def build_feature_matrix(df: pd.DataFrame, include_categorical: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    id_cols = _infer_id_columns(df)

    # Drop clear target columns if present (churn problems)
    drop_cols = [c for c in ["Churn", "Target", "label"] if c in df.columns]

    work = df.drop(columns=id_cols + drop_cols, errors="ignore")

    # Optionally one-hot encode categoricals
    if include_categorical:
        work = pd.get_dummies(work, drop_first=True)
    else:
        # Keep only numeric
        work = work.select_dtypes(include=[np.number])

    # Remove all-zero or all-NaN columns
    work = work.loc[:, work.notna().any()]
    work = work.fillna(0)

    # Remove constant columns
    nunique = work.nunique()
    work = work.loc[:, nunique > 1]

    features = work.columns.tolist()
    return work, features


def scale_and_project(X: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, StandardScaler, Optional[PCA]]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca_model = None
    Xp = None
    if n_components and n_components >= 2:
        pca_model = PCA(n_components=n_components, random_state=42)
        Xp = pca_model.fit_transform(Xs)
    return pd.DataFrame(Xs, columns=X.columns, index=X.index), scaler, pca_model if Xp is not None else None


def elbow_scores(X: pd.DataFrame, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def silhouette_scores_k(X: pd.DataFrame, k_values: List[int], random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = np.nan
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(
    X: pd.DataFrame,
    k: int,
    pca_model: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
    random_state: int = 42,
) -> ClusteringResult:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    sil = None
    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None

    pca_components = None
    pca_explained = None
    if pca_model is not None:
        pca_components = pca_model.transform(X)
        pca_explained = getattr(pca_model, "explained_variance_ratio_", None)

    return ClusteringResult(
        model_name=f"KMeans(k={k})",
        labels=labels,
        inertia_=float(km.inertia_),
        silhouette=sil,
        centroids=km.cluster_centers_,
        pca_components=pca_components,
        pca_explained_variance=pca_explained,
        features_used=X.columns.tolist(),
        scaler=scaler or StandardScaler().fit(X),
        pca_model=pca_model,
    )


def fit_dbscan(
    X: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 10,
    pca_model: Optional[PCA] = None,
    scaler: Optional[StandardScaler] = None,
) -> ClusteringResult:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    # DBSCAN inertia is not defined; silhouette only if more than 1 cluster and no -1 only
    sil = None
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
        try:
            sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            sil = None

    pca_components = None
    pca_explained = None
    if pca_model is not None:
        pca_components = pca_model.transform(X)
        pca_explained = getattr(pca_model, "explained_variance_ratio_", None)

    return ClusteringResult(
        model_name=f"DBSCAN(eps={eps}, min_samples={min_samples})",
        labels=labels,
        inertia_=None,
        silhouette=sil,
        centroids=None,
        pca_components=pca_components,
        pca_explained_variance=pca_explained,
        features_used=X.columns.tolist(),
        scaler=scaler or StandardScaler().fit(X),
        pca_model=pca_model,
    )


def profile_clusters(
    df_original: pd.DataFrame,
    X_features: pd.DataFrame,
    labels: np.ndarray,
    id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cluster_summary, labeled_customers).
    cluster_summary: per-cluster means for numeric features and size.
    labeled_customers: original df with Cluster label appended.
    """
    labeled = df_original.copy()
    labeled["Cluster"] = labels

    # Select numeric features for profiling
    prof = X_features.copy()
    prof["Cluster"] = labels
    summary = prof.groupby("Cluster").agg(["mean", "median", "count"]).T

    # Flatten columns
    summary.index = ["__".join(idx).strip("_") for idx in summary.index]
    summary = summary.T.reset_index().rename(columns={"index": "Cluster"})

    # Add size
    size = pd.Series(labels).value_counts().rename_axis("Cluster").reset_index(name="Size")
    summary = size.merge(summary, on="Cluster", how="left")

    if id_column and id_column in labeled.columns:
        # Nothing additional, but keep id for downstream
        pass

    return summary.sort_values("Cluster"), labeled


def export_cluster_profiles_csv(summary: pd.DataFrame, labeled: pd.DataFrame) -> Tuple[bytes, bytes]:
    buf1, buf2 = io.StringIO(), io.StringIO()
    summary.to_csv(buf1, index=False)
    labeled.to_csv(buf2, index=False)
    return buf1.getvalue().encode(), buf2.getvalue().encode()

