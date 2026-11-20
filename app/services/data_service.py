import json
from functools import lru_cache
import numpy as np
import pandas as pd
from app.core.config import DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    auc, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

SKILL_COLS = [
    "python_score", "sql_score", "ml_score", "dsa_score",
    "web_score", "cloud_score", "core_eng_score",
    "communication_score", "aptitude_score", "teamwork_score"
]

SKILL_DISPLAY = {
    "python_score": "Python",
    "sql_score": "SQL & DB",
    "ml_score": "Machine Learning",
    "dsa_score": "DSA",
    "web_score": "Full-Stack Web",
    "cloud_score": "Cloud & DevOps",
    "core_eng_score": "Core Eng.",
    "communication_score": "Communication",
    "aptitude_score": "Aptitude",
    "teamwork_score": "Teamwork"
}


@lru_cache(maxsize=1)
def load_students() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "students_full.csv")


@lru_cache(maxsize=1)
def load_companies() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "companies.csv")


@lru_cache(maxsize=1)
def load_roles() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "job_roles.csv")


def _read_json(name: str) -> dict:
    path = MODELS_DIR / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# @lru_cache(maxsize=1)
def _build_model_lab_metrics() -> dict:
    try:
        df = load_students()
        feature_cols = [
            "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
            "python_score", "sql_score", "ml_score", "dsa_score",
            "web_score", "cloud_score", "core_eng_score",
            "communication_score", "aptitude_score", "teamwork_score",
            "projects_count", "internships_count", "certifications_count",
            "hackathons_count", "coding_rating"
        ]

        X = df[feature_cols].fillna(0)
        y_clf = df["placed"].fillna(0)
        y_reg = df["package_lpa"].fillna(0)

        X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
        _, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

        models_clf = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        classification_results = []
        best_clf_name = "Gradient Boosting"
        best_roc_auc = -1

        for name, model in models_clf.items():
            model.fit(X_train, y_train_c)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test_c, preds)
            prec = precision_score(y_test_c, preds, zero_division=0)
            rec = recall_score(y_test_c, preds, zero_division=0)
            f1 = f1_score(y_test_c, preds, zero_division=0)
            auc_score = roc_auc_score(y_test_c, probs)

            fpr, tpr, _ = roc_curve(y_test_c, probs)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test_c, probs)
            cm = confusion_matrix(y_test_c, preds).tolist()

            if auc_score > best_roc_auc:
                best_roc_auc = auc_score
                best_clf_name = name

            pr_auc_val = auc(recall_vals, precision_vals)

            classification_results.append({
                "model": name,
                "accuracy": round(float(acc), 3),
                "precision": round(float(prec), 3),
                "recall": round(float(rec), 3),
                "f1_score": round(float(f1), 3),
                "roc_auc": round(float(auc_score), 3),
                "pr_auc": round(float(pr_auc_val), 3),
                "roc_curve": {"fpr": [round(float(x), 3) for x in fpr], "tpr": [round(float(x), 3) for x in tpr]},
                "pr_curve": {"precision": [round(float(x), 3) for x in precision_vals], "recall": [round(float(x), 3) for x in recall_vals]},
                "confusion_matrix": cm
            })

        models_reg = {
            "Ridge Regression": Ridge(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        }

        regression_results = []
        best_reg_name = "Gradient Boosting Regressor"
        best_r2 = -999

        for name, model in models_reg.items():
            model.fit(X_train, y_train_r)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test_r, preds)
            rmse = np.sqrt(mean_squared_error(y_test_r, preds))
            r2 = r2_score(y_test_r, preds)

            if r2 > best_r2:
                best_r2 = r2
                best_reg_name = name

            regression_results.append({
                "model": name,
                "mae_lpa": round(float(mae), 2),
                "rmse_lpa": round(float(rmse), 2),
                "r2": round(float(r2), 3)
            })

        rf_model = models_clf["Random Forest"]
        importances = rf_model.feature_importances_
        feat_imp = sorted(
            [{"feature": col, "importance": round(float(imp), 4)} for col, imp in zip(feature_cols, importances)],
            key=lambda x: x["importance"],
            reverse=True
        )

        return {
            "champion_classifier": best_clf_name,
            "champion_regressor": best_reg_name,
            "classification": classification_results,
            "regression": regression_results,
            "feature_importance": feat_imp
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    meta = _read_json("model_metrics.json")
    if not meta:
        raise FileNotFoundError(
            "models/model_metrics.json missing - run `python scripts/train_models.py`."
        )
    return meta


@lru_cache(maxsize=1)
def load_model_lab_metrics() -> dict:
    data = _read_json("model_lab_metrics.json")
    if not data.get("classification"):
        data = _build_model_lab_metrics()
    return data


def filter_students(batch=None, branch=None, readiness=None, placed=None, search=None) -> pd.DataFrame:
    df = load_students().copy()
    if batch and str(batch) != "ALL":
        df = df[df["batch"] == int(batch)]
    if branch and branch != "ALL":
        df = df[df["branch"] == branch]
    if readiness and readiness != "ALL":
        df = df[df["readiness_band"] == readiness]
    if placed is not None and str(placed) != "ALL":
        df = df[df["placed"] == int(placed)]
    if search:
        q = str(search).strip().lower()
        mask = (
            df["student_id"].str.lower().str.contains(q, na=False)
            | df["name"].str.lower().str.contains(q, na=False)
            | df["branch"].str.lower().str.contains(q, na=False)
            | df["placed_company"].str.lower().str.contains(q, na=False)
            | df["placed_role"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]
    return df


def compute_role_matches_for_student(student_row: dict, top_n: int = 12) -> list:
    roles_df = load_roles()
    results = []
    cgpa = float(student_row["cgpa"])
    backlogs = int(student_row["backlogs"])
    branch = str(student_row["branch"])

    for _, r in roles_df.iterrows():
        pref_branches = str(r["preferred_branches"]).split(",")
        cgpa_ok = cgpa >= float(r["min_cgpa"])
        bl_ok = backlogs <= int(r["max_backlogs"])
        branch_ok = branch in pref_branches
        eligible = cgpa_ok and bl_ok

        req_skills = json.loads(r["required_skills_json"])
        ratios = []
        gaps = []
        for sk, req_val in req_skills.items():
            stu_val = float(student_row.get(sk, 0))
            ratios.append(min(1.1, stu_val / max(req_val, 1)))
            diff = round(req_val - stu_val, 1)
            gaps.append({
                "skill": sk,
                "label": SKILL_DISPLAY.get(sk, sk),
                "student": round(stu_val, 1),
                "required": req_val,
                "gap": max(0.0, diff),
                "surplus": max(0.0, -diff),
            })

        skill_match = float(np.mean(ratios)) * 100.0 if ratios else 75.0
        branch_factor = 1.0 if branch_ok else 0.78
        elig_factor = 1.0 if eligible else 0.55
        fit_score = round(min(99.5, skill_match * branch_factor * elig_factor), 1)

        reasons = []
        if not cgpa_ok:
            reasons.append(f"CGPA {cgpa} < {r['min_cgpa']}")
        if not bl_ok:
            reasons.append(f"Backlogs {backlogs} > {r['max_backlogs']}")
        if not branch_ok:
            reasons.append(f"Non-core branch ({branch})")

        gaps.sort(key=lambda x: x["gap"], reverse=True)
        results.append({
            "role_id": r["role_id"],
            "company_id": r["company_id"],
            "company_name": r["company_name"],
            "tier": r["tier"],
            "domain": r["domain"],
            "role_title": r["role_title"],
            "package_band": r["package_band"],
            "package_lpa": float(r["package_lpa"]),
            "min_cgpa": float(r["min_cgpa"]),
            "max_backlogs": int(r["max_backlogs"]),
            "preferred_branches": pref_branches,
            "eligible": bool(eligible),
            "branch_preferred": bool(branch_ok),
            "eligibility_notes": reasons if reasons else ["Meets all criteria"],
            "fit_score": fit_score,
            "skill_match_pct": round(min(100.0, skill_match), 1),
            "skill_gaps": gaps,
            "required_skills": req_skills,
        })

    results.sort(key=lambda x: (x["eligible"], x["fit_score"]), reverse=True)
    return results[:top_n]