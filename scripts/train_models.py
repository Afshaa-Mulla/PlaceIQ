"""
Model Training, Evaluation & Explainability Pipeline for PlaceIQ
Trains and benchmarks:
1. Placement Probability Classifier: Logistic Regression vs Random Forest vs Gradient Boosting (best kept by ROC-AUC)
2. Package Tier Classifier: Multiclass Random Forest / Gradient Boosting (Low / Mid / High / Premium)
Exports serialized pipelines, ROC/PR/Calibration curve points, confusion matrices, feature importances, and JSON metrics.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
    "python_score", "sql_score", "ml_score", "dsa_score",
    "web_score", "cloud_score", "core_eng_score",
    "communication_score", "aptitude_score", "teamwork_score",
    "projects_count", "internships_count", "certifications_count",
    "hackathons_count", "coding_rating"
]
CATEGORICAL_FEATURES = ["branch"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "cgpa": "CGPA",
    "tenth_pct": "10th Percentage",
    "twelfth_pct": "12th Percentage",
    "backlogs": "Active Backlogs",
    "python_score": "Python Skill",
    "sql_score": "SQL & Databases",
    "ml_score": "Machine Learning",
    "dsa_score": "DSA & Algorithms",
    "web_score": "Full-Stack Web",
    "cloud_score": "Cloud & DevOps",
    "core_eng_score": "Core Engineering",
    "communication_score": "Communication",
    "aptitude_score": "Quantitative Aptitude",
    "teamwork_score": "Teamwork & Leadership",
    "projects_count": "Projects Completed",
    "internships_count": "Internships",
    "certifications_count": "Certifications",
    "hackathons_count": "Hackathons",
    "coding_rating": "Coding Platform Rating",
    "branch": "Engineering Branch"
}


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def downsample_curve(x_arr, y_arr, max_pts=60):
    if len(x_arr) <= max_pts:
        return [round(float(v), 4) for v in x_arr], [round(float(v), 4) for v in y_arr]
    idx = np.linspace(0, len(x_arr) - 1, max_pts, dtype=int)
    return [round(float(x_arr[i]), 4) for i in idx], [round(float(y_arr[i]), 4) for i in idx]


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(DATA_PROC / "train.csv")
    test_df = pd.read_csv(DATA_PROC / "test.csv")

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["placed"].astype(int)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["placed"].astype(int)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=220, max_depth=10, min_samples_leaf=4, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.055, max_depth=3, subsample=0.85, random_state=42
        ),
    }

    comparison = []
    curves = {}
    trained_pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        cal_true, cal_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="uniform")

        fpr_s, tpr_s = downsample_curve(fpr, tpr)
        rec_s, prec_s = downsample_curve(pr_rec[::-1], pr_prec[::-1])

        comparison.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": cm,
        })
        curves[name] = {
            "roc": {"fpr": fpr_s, "tpr": tpr_s, "auc": round(float(auc), 4)},
            "pr": {"recall": rec_s, "precision": prec_s},
            "calibration": {
                "mean_predicted": [round(float(v), 4) for v in cal_pred],
                "fraction_positives": [round(float(v), 4) for v in cal_true],
            },
            "confusion_matrix": cm,
        }
        trained_pipelines[name] = pipe

    # Select best model by ROC-AUC
    comparison.sort(key=lambda x: x["roc_auc"], reverse=True)
    best_model_name = comparison[0]["model"]
    best_placement_pipe = trained_pipelines[best_model_name]

    # Extract feature importances aggregated to original feature level
    prep = best_placement_pipe.named_steps["prep"]
    clf = best_placement_pipe.named_steps["clf"]
    feature_names_out = prep.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
    else:
        raw_imp = np.abs(clf.coef_[0])

    agg_imp = {}
    for fname, val in zip(feature_names_out, raw_imp):
        clean = fname.replace("num__", "").replace("cat__", "")
        if clean.startswith("branch_"):
            clean = "branch"
        agg_imp[clean] = agg_imp.get(clean, 0.0) + float(val)

    total_imp = sum(agg_imp.values()) or 1.0
    feature_importances = sorted(
        [
            {
                "feature": k,
                "label": FEATURE_LABELS.get(k, k),
                "importance": round(v / total_imp, 4),
            }
            for k, v in agg_imp.items()
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Compute baseline statistics for SHAP-style local additive attribution
    num_means = train_df[NUMERIC_FEATURES].mean().to_dict()
    num_stds = train_df[NUMERIC_FEATURES].std().replace(0, 1).to_dict()
    imp_lookup = {item["feature"]: item["importance"] for item in feature_importances}

    # Train Package Tier Multiclass Model on Placed Students
    placed_train = train_df[train_df["placed"] == 1].copy()
    placed_test = test_df[test_df["placed"] == 1].copy()

    pkg_classes = ["Low (<7 LPA)", "Mid (7-11.5 LPA)", "High (11.5-16.5 LPA)", "Premium (16.5+ LPA)"]
    pkg_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.05, max_depth=3, random_state=42
        )),
    ])
    pkg_pipe.fit(placed_train[ALL_FEATURES], placed_train["package_category"])
    pkg_pred = pkg_pipe.predict(placed_test[ALL_FEATURES])
    pkg_acc = accuracy_score(placed_test["package_category"], pkg_pred)
    pkg_f1 = f1_score(placed_test["package_category"], pkg_pred, average="weighted", zero_division=0)
    pkg_cm = confusion_matrix(placed_test["package_category"], pkg_pred, labels=pkg_classes).tolist()
    pkg_report = classification_report(
        placed_test["package_category"], pkg_pred, labels=pkg_classes, output_dict=True, zero_division=0
    )

    # Save artifacts
    joblib.dump(best_placement_pipe, MODELS_DIR / "placement_model.joblib")
    joblib.dump(pkg_pipe, MODELS_DIR / "package_model.joblib")

    metadata = {
        "best_placement_model": best_model_name,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "placed_train_size": int(len(placed_train)),
        "placed_test_size": int(len(placed_test)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "model_comparison": comparison,
        "curves": curves,
        "feature_importances": feature_importances,
        "baseline_means": {k: round(float(v), 3) for k, v in num_means.items()},
        "baseline_stds": {k: round(float(v), 3) for k, v in num_stds.items()},
        "importance_weights": imp_lookup,
        "package_model": {
            "algorithm": "Gradient Boosting Multiclass",
            "classes": pkg_classes,
            "accuracy": round(float(pkg_acc), 4),
            "weighted_f1": round(float(pkg_f1), 4),
            "confusion_matrix": pkg_cm,
            "per_class": {
                c: {
                    "precision": round(float(pkg_report[c]["precision"]), 4),
                    "recall": round(float(pkg_report[c]["recall"]), 4),
                    "f1": round(float(pkg_report[c]["f1-score"]), 4),
                    "support": int(pkg_report[c]["support"]),
                }
                for c in pkg_classes
            },
        },
    }

    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best Placement Model: {best_model_name} (ROC-AUC={comparison[0]['roc_auc']}, Acc={comparison[0]['accuracy']})")
    print(f"Package Multiclass Model Acc={pkg_acc:.4f}, Weighted F1={pkg_f1:.4f}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
