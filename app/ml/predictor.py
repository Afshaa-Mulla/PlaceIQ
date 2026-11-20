from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from app.core.config import MODELS_DIR
from app.services.data_service import load_metrics

def _load_or_retrain(filename: str):
    path = MODELS_DIR / filename
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[PlaceIQ] Could not load {filename} ({type(exc).__name__}: {exc}). Retraining models...")
        from scripts.train_models import train_and_evaluate
        train_and_evaluate()
        load_metrics.cache_clear()
        return joblib.load(path)
    
@lru_cache(maxsize=1)
def get_placement_model():
    return _load_or_retrain("placement_model.joblib")


@lru_cache(maxsize=1)
def get_package_model():
    return _load_or_retrain("package_model.joblib")


def compute_readiness_score(profile: dict) -> tuple[float, str]:
    branch = profile.get("branch", "CSE")
    cgpa = float(profile.get("cgpa", 7.5))
    backlogs = int(profile.get("backlogs", 0))
    py = float(profile.get("python_score", 65))
    sql = float(profile.get("sql_score", 65))
    ml = float(profile.get("ml_score", 55))
    dsa = float(profile.get("dsa_score", 60))
    web = float(profile.get("web_score", 60))
    cloud = float(profile.get("cloud_score", 55))
    core = float(profile.get("core_eng_score", 60))
    comm = float(profile.get("communication_score", 68))
    apt = float(profile.get("aptitude_score", 68))
    team = float(profile.get("teamwork_score", 70))
    proj = int(profile.get("projects_count", 2))
    intern = int(profile.get("internships_count", 1))
    cert = int(profile.get("certifications_count", 2))
    hack = int(profile.get("hackathons_count", 1))
    rating = float(profile.get("coding_rating", 1300))

    if branch in ("MECH", "CIVIL"):
        tech_index = 0.45 * core + 0.20 * py + 0.20 * sql + 0.15 * apt
    elif branch in ("ECE", "EE"):
        tech_index = 0.32 * core + 0.24 * py + 0.24 * dsa + 0.20 * cloud
    elif branch == "DS":
        tech_index = 0.30 * py + 0.30 * ml + 0.22 * sql + 0.18 * dsa
    else:
        tech_index = 0.30 * dsa + 0.25 * py + 0.23 * web + 0.22 * sql

    soft_index = 0.40 * comm + 0.35 * apt + 0.25 * team
    exp_index = min(100.0, proj * 9.0 + intern * 22.0 + cert * 5.0 + hack * 6.5)

    raw = (
        0.24 * ((cgpa - 5.0) / 5.0 * 100.0)
        + 0.34 * tech_index
        + 0.20 * soft_index
        + 0.16 * exp_index
        + 0.06 * ((rating - 800) / 1550.0 * 100.0)
        - 6.5 * backlogs
    )
    score = round(float(np.clip(raw, 12.0, 98.5)), 1)
    if score >= 76:
        band = "Industry Ready"
    elif score >= 60:
        band = "Developing"
    else:
        band = "Needs Upskilling"
    return score, band


def predict_student_profile(profile: dict) -> dict:
    meta = load_metrics()
    num_feats = meta["numeric_features"]
    cat_feats = meta["categorical_features"]
    all_feats = num_feats + cat_feats

    row = {}
    for f in num_feats:
        row[f] = float(profile.get(f, meta["baseline_means"].get(f, 0)))
    row["branch"] = str(profile.get("branch", "CSE"))

    X = pd.DataFrame([row])[all_feats]
    place_model = get_placement_model()
    pkg_model = get_package_model()

    prob = float(place_model.predict_proba(X)[0, 1])
    prob_pct = round(prob * 100.0, 1)

    pkg_probs = pkg_model.predict_proba(X)[0]
    pkg_classes = list(pkg_model.classes_)
    pkg_dist = {cls: round(float(p) * 100.0, 1) for cls, p in zip(pkg_classes, pkg_probs)}
    best_pkg_idx = int(np.argmax(pkg_probs))
    predicted_pkg_band = pkg_classes[best_pkg_idx]

    midpoints = {
        "Low (<7 LPA)": 5.5,
        "Mid (7-11.5 LPA)": 8.8,
        "High (11.5-16.5 LPA)": 13.6,
        "Premium (16.5+ LPA)": 18.8,
    }
    expected_lpa = round(sum(midpoints.get(c, 7.5) * p for c, p in zip(pkg_classes, pkg_probs)), 2)

    means = meta["baseline_means"]
    stds = meta["baseline_stds"]
    weights = meta["importance_weights"]
    labels = meta["feature_labels"]

    contributions = []
    for f in num_feats:
        val = row[f]
        z = (val - means[f]) / max(stds[f], 1e-6)
        sign = -1.0 if f == "backlogs" else 1.0
        w = weights.get(f, 0.03)
        impact_pp = round(sign * z * w * 85.0, 2)
        contributions.append({
            "feature": f,
            "label": labels.get(f, f),
            "value": round(val, 2),
            "cohort_avg": round(means[f], 2),
            "impact_pp": impact_pp,
        })

    contributions.sort(key=lambda x: abs(x["impact_pp"]), reverse=True)
    readiness_score, readiness_band = compute_readiness_score(row)

    positives = [c for c in contributions if c["impact_pp"] > 0.8][:3]
    negatives = [c for c in contributions if c["impact_pp"] < -0.8][:3]

    pos_txt = ", ".join([f"{c['label']} ({c['value']})" for c in positives]) or "balanced baseline metrics"
    neg_txt = ", ".join([f"{c['label']} ({c['value']} vs avg {c['cohort_avg']})" for c in negatives]) or "no major red flags"

    plain_explanation = (
        f"Predicted placement probability is {prob_pct}% ({readiness_band}, Readiness {readiness_score}/100). "
        f"Key upward drivers: {pos_txt}. Primary drag factors: {neg_txt}."
    )

    return {
        "placement_probability": prob_pct,
        "placement_label": "Likely Placed" if prob >= 0.5 else "At Risk / Needs Boost",
        "predicted_package_band": predicted_pkg_band,
        "expected_lpa": expected_lpa,
        "package_probabilities": pkg_dist,
        "readiness_score": readiness_score,
        "readiness_band": readiness_band,
        "base_rate_pct": 48.6,
        "contributions": contributions[:12],
        "top_strengths": positives,
        "top_drags": negatives,
        "plain_explanation": plain_explanation,
    }
