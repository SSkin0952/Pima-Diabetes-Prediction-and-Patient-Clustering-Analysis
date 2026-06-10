"""
analysis_diabetes_pycharm.py

PyCharm-ready version of the Pima diabetes analysis notebook.

What this script does:
1. Loads diabetes.csv from KaggleHub or a local path.
2. Trains Linear Regression, Logistic Regression, Gradient Boosting, Random Forest, and MLP models.
3. Prints metrics to console and saves them to output/results_summary.txt.
4. Saves all figures as PNG files in the output/figures folder.
5. Saves important CSV outputs in the output folder.

How to run in PyCharm:
- Put this file in your project.
- Install packages:
  pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm kagglehub
- Run this file directly.

Optional:
- If KaggleHub download fails, manually place diabetes.csv in the same folder as this script.
"""

import os
from pathlib import Path
from contextlib import redirect_stdout

# Keep this before sklearn imports to avoid some Windows/MKL threading issues.
os.environ["OMP_NUM_THREADS"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.calibration import calibration_curve


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str) -> None:
    """Save current matplotlib figure and close it."""
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def load_data() -> pd.DataFrame:
    """Load Pima diabetes dataset from KaggleHub, or from local diabetes.csv."""
    local_csv = BASE_DIR / "diabetes.csv"
    if local_csv.exists():
        print(f"Using local dataset: {local_csv}")
        return pd.read_csv(local_csv)

    try:
        import kagglehub

        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        csv_path = Path(path) / "diabetes.csv"
        print(f"Downloaded dataset path: {csv_path}")
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not load dataset. Either install/configure kagglehub, "
            "or put diabetes.csv in the same folder as this script."
        ) from exc


def evaluate_classifier(name: str, y_test, y_pred, y_prob=None) -> dict:
    """Print classification metrics and return compact metric dict."""
    print(f"\n=== {name} ===")
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", auc)

    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return {"Model": name, "Accuracy": acc, "AUC": auc}


def summarize_clusters(dfc: pd.DataFrame, cluster_col: str, feature_cols: list[str]) -> pd.DataFrame:
    summary = []
    for c in sorted(dfc[cluster_col].unique()):
        sub = dfc[dfc[cluster_col] == c]
        row = {
            "cluster": c,
            "n": len(sub),
            "outcome_rate": sub["Outcome"].mean(),
        }
        row.update({f"mean_{k}": v for k, v in sub[feature_cols].mean().to_dict().items()})
        summary.append(row)
    return pd.DataFrame(summary)


def main() -> None:
    # ---------------------------
    # 1. Load data
    # ---------------------------
    df = load_data()
    print("Loaded data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # ---------------------------
    # 2. Data cleaning
    # ---------------------------
    # Note: Pregnancies and DiabetesPedigreeFunction can be 0 in real data.
    # I kept your original notebook logic for consistency.
    cols_zero_missing = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)
    print("\nMissing counts after replacing 0 with NaN:")
    print(df.isnull().sum())

    median_dict = {c: df[c].median() for c in cols_zero_missing}
    df.fillna(value=median_dict, inplace=True)

    print("\nAfter imputation, missing counts:")
    print(df.isnull().sum())

    # ---------------------------
    # 3. Prepare X, y and split
    # ---------------------------
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    metrics = []

    # ---------------------------
    # 4. Linear Regression baseline
    # ---------------------------
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = (lin.predict(X_test) >= 0.5).astype(int)

    print("\n=== Linear Regression baseline ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_lin))
    print(classification_report(y_test, y_pred_lin, digits=4))
    metrics.append({"Model": "Linear Regression baseline", "Accuracy": accuracy_score(y_test, y_pred_lin), "AUC": None})

    # ---------------------------
    # 5. Supervised models
    # ---------------------------
    log = LogisticRegression(max_iter=2000, solver="lbfgs")
    log.fit(X_train, y_train)
    y_pred_log = log.predict(X_test)
    y_prob_log = log.predict_proba(X_test)[:, 1]
    metrics.append(evaluate_classifier("Logistic Regression", y_test, y_pred_log, y_prob_log))

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]
    metrics.append(evaluate_classifier("Gradient Boosting", y_test, y_pred_gb, y_prob_gb))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    metrics.append(evaluate_classifier("Random Forest", y_test, y_pred_rf, y_prob_rf))

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)
    metrics.append(evaluate_classifier("MLP", y_test, y_pred_mlp, y_prob_mlp))

    # ---------------------------
    # 6. Figures for supervised models
    # ---------------------------
    models_for_cm = [
        ("Logistic Regression", y_pred_log),
        ("Gradient Boosting", y_pred_gb),
        ("Random Forest", y_pred_rf),
        ("MLP", y_pred_mlp),
    ]

    plt.figure(figsize=(10, 8))
    for i, (name, yp) in enumerate(models_for_cm, 1):
        ax = plt.subplot(2, 2, i)
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, yp))
        disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.suptitle("Confusion Matrix Comparison")
    save_current_figure("01_confusion_matrix_comparison.png")

    model_probs = [
        ("Logistic Regression", y_prob_log),
        ("Gradient Boosting", y_prob_gb),
        ("Random Forest", y_prob_rf),
        ("MLP", y_prob_mlp),
    ]

    roc_data = {}
    for name, prob in model_probs:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, prob), prob)

    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc, _) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison of All Models")
    plt.legend()
    plt.grid(True)
    save_current_figure("02_roc_curve_comparison.png")

    plt.figure(figsize=(12, 10))
    for i, (name, (fpr, tpr, auc, _)) in enumerate(roc_data.items(), 1):
        ax = plt.subplot(2, 2, i)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"{name} ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True)
    save_current_figure("03_individual_roc_curves.png")

    auc_df = pd.DataFrame(
        [(name, auc) for name, (_, _, auc, _) in roc_data.items()],
        columns=["Model", "AUC"],
    ).sort_values("AUC", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison AUC ===")
    print(auc_df)
    auc_df.to_csv(OUTPUT_DIR / "model_auc_comparison.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="AUC", y="Model", data=auc_df, orient="h")
    plt.title("ROC AUC Comparison Across Models")
    plt.xlim(0.5, 1.0)
    save_current_figure("04_auc_barplot.png")

    plt.figure(figsize=(12, 10))
    y_test_np = y_test.to_numpy()
    for i, (name, prob) in enumerate(model_probs, 1):
        ax = plt.subplot(2, 2, i)
        sns.kdeplot(prob[y_test_np == 0], label="Outcome = 0", fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(prob[y_test_np == 1], label="Outcome = 1", fill=True, alpha=0.5, ax=ax)
        ax.set_title(f"{name}: Predicted Probability Distribution")
        ax.set_xlabel("P(Outcome = 1)")
        ax.legend()
    save_current_figure("05_predicted_probability_distribution.png")

    plt.figure(figsize=(6, 6))
    for name, prob in model_probs:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of positive")
    plt.title("Calibration Curves")
    plt.legend()
    plt.grid(True)
    save_current_figure("06_calibration_curves.png")

    # ---------------------------
    # 7. Bootstrap logistic coefficients CI
    # ---------------------------
    n_boot = 1000
    rng = np.random.RandomState(0)
    coefs = np.zeros((n_boot, X_train.shape[1]))

    X_train_np = X_train.reset_index(drop=True)
    y_train_np = y_train.reset_index(drop=True)

    for i in tqdm(range(n_boot), desc="Bootstrapping coefficients"):
        idx = rng.randint(0, len(X_train_np), len(X_train_np))
        Xb = X_train_np.iloc[idx]
        yb = y_train_np.iloc[idx]
        model_b = LogisticRegression(max_iter=2000, solver="lbfgs")
        try:
            model_b.fit(Xb, yb)
            coefs[i, :] = model_b.coef_.flatten()
        except Exception:
            coefs[i, :] = np.nan

    coef_summary = []
    for j, col in enumerate(X.columns):
        col_vals = coefs[:, j]
        col_vals = col_vals[~np.isnan(col_vals)]
        coef_summary.append(
            (
                col,
                np.mean(col_vals),
                np.percentile(col_vals, 2.5),
                np.percentile(col_vals, 97.5),
            )
        )

    coef_summary_df = pd.DataFrame(
        coef_summary,
        columns=["feature", "coef_mean", "2.5%", "97.5%"],
    ).sort_values(by="coef_mean", key=lambda s: s.abs(), ascending=False)
    print("\nLogistic coefficient bootstrap summary:")
    print(coef_summary_df)
    coef_summary_df.to_csv(OUTPUT_DIR / "logistic_coef_bootstrap_summary.csv", index=False)

    # ---------------------------
    # 8. EDA correlation heatmap
    # ---------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation after Imputation")
    save_current_figure("07_feature_correlation_heatmap.png")

    # ---------------------------
    # 9. Unsupervised clustering analysis
    # ---------------------------
    X_all = X_scaled.copy()
    ks = range(2, 7)
    inertias = []
    sil_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_all)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_all, labels))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(list(ks), inertias, "-o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow")

    plt.subplot(1, 2, 2)
    plt.plot(list(ks), sil_scores, "-o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans Silhouette")
    save_current_figure("08_kmeans_elbow_silhouette.png")

    k_best = list(ks)[int(np.argmax(sil_scores))]
    print(f"\nSelected k by max silhouette among 2..6: {k_best}")

    km_final = KMeans(n_clusters=k_best, random_state=42, n_init=20)
    km_labels = km_final.fit_predict(X_all)
    df_clusters = df.copy()
    df_clusters["cluster_kmeans"] = km_labels

    print("\nKMeans cluster sizes:")
    print(df_clusters["cluster_kmeans"].value_counts().sort_index())

    outcome_by_cluster = pd.crosstab(
        df_clusters["cluster_kmeans"],
        df_clusters["Outcome"],
        normalize="index",
    ) * 100
    print("\nOutcome percent by KMeans cluster:")
    print(outcome_by_cluster.round(2))

    centroids_scaled = pd.DataFrame(km_final.cluster_centers_, columns=X_all.columns)
    centroids_orig = pd.DataFrame(scaler.inverse_transform(centroids_scaled), columns=X_all.columns)
    print("\nCluster centroids original scale:")
    print(centroids_orig.T)

    cluster_profile = df_clusters.groupby("cluster_kmeans")[X.columns.tolist()].mean()
    print("\nCluster profiles feature means:")
    print(cluster_profile)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=km_labels, palette="tab10", legend="full")
    plt.title(f"KMeans clusters k={k_best} in PCA 2D space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="cluster")
    save_current_figure("09_kmeans_clusters_pca.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        centroids_scaled,
        yticklabels=[f"cluster_{i}" for i in range(k_best)],
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
    )
    plt.title("Cluster centroids standardized features")
    plt.xlabel("Feature")
    save_current_figure("10_cluster_centroids_heatmap.png")

    Z = linkage(X_all, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5, leaf_rotation=90.0, leaf_font_size=10.0)
    plt.title("Hierarchical Clustering Dendrogram truncated")
    plt.xlabel("Sample index or cluster size")
    plt.ylabel("Distance")
    save_current_figure("11_hierarchical_dendrogram.png")

    n_clusters_h = k_best
    agg = AgglomerativeClustering(n_clusters=n_clusters_h, linkage="ward")
    agg_labels = agg.fit_predict(X_all)
    df_clusters["cluster_hier"] = agg_labels

    print("\nHierarchical cluster sizes:")
    print(pd.Series(agg_labels).value_counts().sort_index())
    print("\nOutcome percent by Hierarchical cluster:")
    print((pd.crosstab(df_clusters["cluster_hier"], df_clusters["Outcome"], normalize="index") * 100).round(2))

    ari = adjusted_rand_score(km_labels, agg_labels)
    print(f"\nAdjusted Rand Index between KMeans and Agglomerative k={k_best}: {ari:.3f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agg_labels, palette="tab10", legend="full")
    plt.title(f"Hierarchical clusters n={n_clusters_h} in PCA 2D space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="cluster")
    save_current_figure("12_hierarchical_clusters_pca.png")

    km_summary = summarize_clusters(df_clusters, "cluster_kmeans", X.columns.tolist()).sort_values("cluster")
    hier_summary = summarize_clusters(df_clusters, "cluster_hier", X.columns.tolist()).sort_values("cluster")

    print("\nKMeans cluster summary means + outcome rate:")
    print(km_summary.round(3))
    print("\nHierarchical cluster summary means + outcome rate:")
    print(hier_summary.round(3))

    df_clusters.to_csv(OUTPUT_DIR / "pima_clusters.csv", index=False)
    km_summary.to_csv(OUTPUT_DIR / "kmeans_cluster_summary.csv", index=False)
    hier_summary.to_csv(OUTPUT_DIR / "hierarchical_cluster_summary.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    print("\n=== Compact metrics summary ===")
    print(metrics_df)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    log_path = OUTPUT_DIR / "results_summary.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            main()
    print(f"Finished. Full text results saved to: {log_path}")
    print(f"Figures saved to: {FIG_DIR}")
