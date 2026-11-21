import json
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import traceback

from app.core.config import get_gemini_key, get_gemini_model
from app.services.data_service import (
    load_students, load_companies, load_roles, load_model_lab_metrics,
    filter_students, compute_role_matches_for_student, SKILL_COLS, SKILL_DISPLAY
)
from app.ml.predictor import predict_student_profile
from app.services.gemini_service import (
    chat_with_career_assistant, generate_30_60_90_roadmap,
    ask_dashboard_nlq, generate_batch_executive_report
)
from app.services.excel_service import build_excel_report

router = APIRouter(prefix="/api")


def clean_floats(obj):
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return 0.0 if (math.isnan(val) or math.isinf(val)) else val
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_floats(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_floats(v) for v in obj]
    elif pd.isna(obj):
        return 0.0
    return obj


class PredictRequest(BaseModel):
    branch: str = "CSE"
    cgpa: float = 7.8
    tenth_pct: float = 82.0
    twelfth_pct: float = 80.0
    backlogs: int = 0
    python_score: float = 72.0
    sql_score: float = 68.0
    ml_score: float = 62.0
    dsa_score: float = 70.0
    web_score: float = 66.0
    cloud_score: float = 60.0
    core_eng_score: float = 62.0
    communication_score: float = 72.0
    aptitude_score: float = 72.0
    teamwork_score: float = 74.0
    projects_count: int = 3
    internships_count: int = 1
    certifications_count: int = 2
    hackathons_count: int = 2
    coding_rating: int = 1420


class ChatRequest(BaseModel):
    student_id: str = "STU0001"
    message: str
    history: list = []


class RoadmapRequest(BaseModel):
    student_id: str = "STU0001"
    target_role: str = ""


class NLQRequest(BaseModel):
    question: str


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "gemini_configured": bool(get_gemini_key()),
        "gemini_model": get_gemini_model(),
    }


@router.get("/overview")
def get_overview(batch: str = "ALL", branch: str = "ALL"):
    df = filter_students(batch=batch, branch=branch)
    full_df = load_students()
    placed_df = df[df["placed"] == 1]

    total = int(len(df))
    placed_cnt = int(df["placed"].sum())
    placement_rate = round(placed_cnt / total * 100, 1) if total else 0.0
    avg_pkg = round(float(placed_df["package_lpa"].mean()), 2) if len(placed_df) else 0.0
    median_pkg = round(float(placed_df["package_lpa"].median()), 2) if len(placed_df) else 0.0
    max_pkg = round(float(placed_df["package_lpa"].max()), 2) if len(placed_df) else 0.0
    avg_readiness = round(float(df["readiness_score"].mean()), 1) if total else 0.0
    ready_pct = round(float((df["readiness_band"] == "Industry Ready").mean() * 100), 1) if total else 0.0

    trend_base = filter_students(batch="ALL", branch=branch)
    batch_trends = []
    for b, g in trend_base.groupby("batch"):
        gp = g[g["placed"] == 1]
        batch_trends.append({
            "batch": int(b),
            "students": int(len(g)),
            "placed": int(g["placed"].sum()),
            "placement_rate": round(float(g["placed"].mean() * 100), 1),
            "avg_package": round(float(gp["package_lpa"].mean()), 2) if len(gp) else 0.0,
            "avg_readiness": round(float(g["readiness_score"].mean()), 1),
        })

    branch_base = filter_students(batch=batch, branch="ALL")
    branch_summary = []
    for br, g in branch_base.groupby("branch"):
        gp = g[g["placed"] == 1]
        branch_summary.append({
            "branch": br,
            "branch_name": g["branch_name"].iloc[0],
            "students": int(len(g)),
            "placed": int(g["placed"].sum()),
            "placement_rate": round(float(g["placed"].mean() * 100), 1),
            "avg_package": round(float(gp["package_lpa"].mean()), 2) if len(gp) else 0.0,
            "avg_readiness": round(float(g["readiness_score"].mean()), 1),
        })
    branch_summary.sort(key=lambda x: x["placement_rate"], reverse=True)

    sankey_nodes = []
    node_idx = {}

    def get_node(label):
        if label not in node_idx:
            node_idx[label] = len(sankey_nodes)
            sankey_nodes.append(label)
        return node_idx[label]

    links_map = {}

    def add_link(src_lbl, tgt_lbl, val):
        if val <= 0:
            return
        s_i = get_node(src_lbl)
        t_i = get_node(tgt_lbl)
        links_map[(s_i, t_i)] = links_map.get((s_i, t_i), 0) + int(val)

    for (br, rb), cnt in df.groupby(["branch", "readiness_band"]).size().items():
        add_link(f"Branch: {br}", rb, cnt)

    for (rb, pl), cnt in df.groupby(["readiness_band", "placed"]).size().items():
        status = "Placed" if int(pl) == 1 else "Unplaced"
        add_link(rb, status, cnt)

    for (pl, pc), cnt in df.groupby(["placed", "package_category"]).size().items():
        status = "Placed" if int(pl) == 1 else "Unplaced"
        dest = pc if int(pl) == 1 else "Seeking Offer"
        add_link(status, dest, cnt)

    sankey_links = [
        {"source": s, "target": t, "value": v}
        for (s, t), v in links_map.items()
    ]

    funnel = [
        {"stage": "Total Applications", "count": int(df["applications_sent"].sum())},
        {"stage": "Eligible Drives", "count": int(df["eligible_drives"].sum())},
        {"stage": "Shortlisted Rounds", "count": int(df["shortlisted_count"].sum())},
        {"stage": "Final Offers (Placed)", "count": placed_cnt},
    ]

    sunburst_items = []
    if len(placed_df):
        roles_df = load_roles()
        tier_map = dict(zip(roles_df["company_name"], roles_df["tier"]))
        grp_cr = (
            placed_df.groupby(["placed_company", "placed_role"])
            .agg(hires=("student_id", "count"), avg_lpa=("package_lpa", "mean"))
            .reset_index()
        )
        for _, r in grp_cr.iterrows():
            sunburst_items.append({
                "tier": tier_map.get(r["placed_company"], "Enterprise"),
                "company": r["placed_company"],
                "role": r["placed_role"],
                "hires": int(r["hires"]),
                "avg_lpa": round(float(r["avg_lpa"]), 2),
            })

    return clean_floats({
        "kpis": {
            "total_students": total,
            "placed_students": placed_cnt,
            "placement_rate": placement_rate,
            "avg_package_lpa": avg_pkg,
            "median_package_lpa": median_pkg,
            "max_package_lpa": max_pkg,
            "avg_readiness": avg_readiness,
            "industry_ready_pct": ready_pct,
            "companies_count": int(load_companies().shape[0]),
            "roles_count": int(load_roles().shape[0]),
        },
        "batch_trends": batch_trends,
        "branch_summary": branch_summary,
        "sankey": {"nodes": sankey_nodes, "links": sankey_links},
        "funnel": funnel,
        "sunburst": sunburst_items,
        "all_batches": sorted([int(x) for x in full_df["batch"].unique()]),
        "all_branches": sorted(list(full_df["branch"].unique())),
    })


@router.get("/batch-analytics")
def get_batch_analytics(batch: str = "ALL", branch: str = "ALL"):
    df = filter_students(batch=batch, branch=branch)
    placed_df = df[df["placed"] == 1]

    pkg_by_branch = {}
    for br, g in placed_df.groupby("branch"):
        pkg_by_branch[br] = [round(float(v), 2) for v in g["package_lpa"].tolist()]

    pkg_by_readiness = {}
    for rb, g in placed_df.groupby("readiness_band"):
        pkg_by_readiness[rb] = [round(float(v), 2) for v in g["package_lpa"].tolist()]

    sample_df = df.sample(n=min(180, len(df)), random_state=42) if len(df) else df
    parallel_rows = sample_df[[
        "student_id", "branch", "cgpa", "dsa_score", "python_score", "sql_score",
        "communication_score", "readiness_score", "package_lpa", "placed"
    ]].to_dict(orient="records")

    scatter_df = df.sample(n=min(350, len(df)), random_state=7) if len(df) else df
    scatter_rows = scatter_df[[
        "student_id", "name", "branch", "batch", "cgpa", "tech_index", "soft_index",
        "readiness_score", "internships_count", "package_lpa", "placed", "placed_company"
    ]].to_dict(orient="records")

    corr_cols = [
        "cgpa", "backlogs", "python_score", "sql_score", "dsa_score", "ml_score",
        "cloud_score", "core_eng_score", "communication_score", "aptitude_score",
        "internships_count", "projects_count", "coding_rating", "readiness_score", "placed", "package_lpa"
    ]
    corr_labels = [
        "CGPA", "Backlogs", "Python", "SQL", "DSA", "ML",
        "Cloud", "Core Eng", "Comm.", "Aptitude",
        "Internships", "Projects", "Coding Rating", "Readiness", "Placed", "Package"
    ]
    corr_mat = df[corr_cols].corr().fillna(0).round(2).values.tolist() if len(df) > 2 else []

    branch_skills = []
    for br, g in df.groupby("branch"):
        row = {"branch": br}
        for sk in SKILL_COLS:
            row[sk] = round(float(g[sk].mean()), 1)
        branch_skills.append(row)

    return clean_floats({
        "pkg_by_branch": pkg_by_branch,
        "pkg_by_readiness": pkg_by_readiness,
        "parallel_rows": parallel_rows,
        "scatter_rows": scatter_rows,
        "correlation": {"labels": corr_labels, "matrix": corr_mat},
        "branch_skills": branch_skills,
    })


@router.get("/students")
def list_students(
    batch: str = "ALL",
    branch: str = "ALL",
    readiness: str = "ALL",
    placed: str = "ALL",
    search: str = "",
    sort_by: str = "readiness_score",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 25,
):
    df = filter_students(batch=batch, branch=branch, readiness=readiness, placed=placed, search=search)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=(sort_dir.lower() == "asc"))
    total = int(len(df))
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    subset = df.iloc[start:end].fillna("")
    return clean_floats({
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
        "items": subset.to_dict(orient="records"),
    })


@router.get("/students/{student_id}")
def get_student_360(student_id: str):
    df = load_students()
    match = df[df["student_id"] == student_id.upper()]
    if match.empty:
        row = df.iloc[0].fillna("").to_dict()
    else:
        row = match.iloc[0].fillna("").to_dict()

    branch_df = df[df["branch"] == row["branch"]]
    peer_avg = {sk: round(float(branch_df[sk].mean()), 1) for sk in SKILL_COLS}

    prediction = predict_student_profile(row)

    matches = compute_role_matches_for_student(row, top_n=12)

    return clean_floats({
        "student": row,
        "peer_avg": peer_avg,
        "skill_labels": SKILL_DISPLAY,
        "prediction": prediction,
        "role_matches": matches,
    })


@router.post("/predict")
def run_prediction(req: PredictRequest):
    payload = req.model_dump()
    prediction = predict_student_profile(payload)
    top_matches = compute_role_matches_for_student(payload, top_n=6)
    return clean_floats({"prediction": prediction, "top_matches": top_matches})


@router.get("/matcher")
def company_matcher(
    student_id: str = "STU0001",
    company_id: str = "",
    tier: str = "ALL",
):
    df = load_students()
    companies_df = load_companies()
    roles_df = load_roles()

    stu_match = df[df["student_id"] == student_id.upper()]
    stu_row = (stu_match.iloc[0] if not stu_match.empty else df.iloc[0]).fillna("").to_dict()
    student_matches = compute_role_matches_for_student(stu_row, top_n=60)
    if tier != "ALL":
        student_matches = [m for m in student_matches if m["tier"] == tier]

    if not company_id:
        company_id = companies_df.iloc[0]["company_id"]
    comp_row = companies_df[companies_df["company_id"] == company_id]
    shortlist = []
    selected_company = {}
    if not comp_row.empty:
        c = comp_row.iloc[0].to_dict()
        selected_company = c
        pref_branches = str(c["preferred_branches"]).split(",")
        elig_df = df[(df["cgpa"] >= float(c["min_cgpa"])) & (df["backlogs"] <= int(c["max_backlogs"]))].copy()
        elig_df["branch_bonus"] = np.where(elig_df["branch"].isin(pref_branches), 12.0, 0.0)
        elig_df["drive_rank_score"] = (elig_df["readiness_score"] * 0.75 + elig_df["cgpa"] * 2.2 + elig_df["branch_bonus"]).round(1)
        elig_df = elig_df.sort_values(by="drive_rank_score", ascending=False).head(30)
        shortlist = elig_df[[
            "student_id", "name", "batch", "branch", "cgpa", "backlogs",
            "readiness_score", "tech_index", "soft_index", "drive_rank_score", "placed", "placed_company"
        ]].to_dict(orient="records")

    return clean_floats({
        "student": stu_row,
        "role_matches": student_matches,
        "companies": companies_df.to_dict(orient="records"),
        "selected_company": selected_company,
        "eligible_shortlist": shortlist,
        "total_roles": int(len(roles_df)),
    })


@router.get("/skill-gaps")
def get_skill_gaps(batch: str = "ALL", branch: str = "ALL", student_id: str = "STU0001"):
    df = filter_students(batch=batch, branch=branch)
    roles_df = load_roles()

    demand = {}
    for sk in SKILL_COLS:
        col = f"req_{sk}"
        vals = roles_df[roles_df[col] > 0][col]
        demand[sk] = round(float(vals.mean()), 1) if len(vals) and not pd.isna(vals.mean()) else 70.0

    batch_skill_gaps = []
    for sk in SKILL_COLS:
        cohort_avg = round(float(df[sk].mean()), 1) if len(df) and not pd.isna(df[sk].mean()) else 0.0
        target = demand[sk]
        gap = round(max(0.0, target - cohort_avg), 1)
        batch_skill_gaps.append({
            "skill": sk,
            "label": SKILL_DISPLAY[sk],
            "cohort_avg": cohort_avg,
            "industry_benchmark": target,
            "gap": gap,
            "priority": "Critical" if gap >= 12 else ("Moderate" if gap >= 5 else "On Track"),
        })
    batch_skill_gaps.sort(key=lambda x: x["gap"], reverse=True)

    top_comps = load_companies().head(18)
    heatmap_companies = []
    heatmap_matrix = []
    for _, c in top_comps.iterrows():
        c_roles = roles_df[roles_df["company_id"] == c["company_id"]]
        heatmap_companies.append(c["company_name"])
        row_vals = []
        for sk in SKILL_COLS:
            v = float(c_roles[f"req_{sk}"].max()) if len(c_roles) and not pd.isna(c_roles[f"req_{sk}"].max()) else 0.0
            row_vals.append(round(v, 0))
        heatmap_matrix.append(row_vals)

    full_df = load_students()
    s_match = full_df[full_df["student_id"] == student_id.upper()]
    stu_row = (s_match.iloc[0] if not s_match.empty else full_df.iloc[0]).fillna("").to_dict()
    matches = compute_role_matches_for_student(stu_row, top_n=3)

    return clean_floats({
        "batch_skill_gaps": batch_skill_gaps,
        "heatmap": {
            "companies": heatmap_companies,
            "skills": [SKILL_DISPLAY[s] for s in SKILL_COLS],
            "matrix": heatmap_matrix,
        },
        "student": stu_row,
        "student_top_matches": matches,
    })


@router.get("/model-lab")
def get_model_lab():
    return clean_floats(load_model_lab_metrics())

#adding to check error
def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        traceback.print_exc()
        err_msg = f"Backend error: {type(exc).__name__}: {exc}"
        return {
            "success": False,
            "configured": True,
            "reply": err_msg,
            "roadmap": err_msg,
            "answer": err_msg,
            "report": err_msg,
        }

@router.post("/ai/chat")
def ai_chat(req: ChatRequest):
    return _safe(chat_with_career_assistant, req.student_id, req.message, req.history)


@router.post("/ai/roadmap")
def ai_roadmap(req: RoadmapRequest):
    return _safe(generate_30_60_90_roadmap, req.student_id, req.target_role)


@router.post("/ai/nlq")
def ai_nlq(req: NLQRequest):
    return ask_dashboard_nlq(req.question)


@router.get("/ai/batch-summary")
def ai_batch_summary(batch: str = "ALL", branch: str = "ALL"):
    return generate_batch_executive_report(batch=batch, branch=branch)


@router.get("/reports/export")
def export_report(
    report_type: str = Query("student", pattern="^(student|batch|company)$"),
    batch: str = "ALL",
    branch: str = "ALL",
    company_id: str = "",
):
    buf = build_excel_report(report_type=report_type, batch=batch, branch=branch, company_id=company_id)
    filename = f"PlaceIQ_{report_type}_report_{batch}_{branch}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )