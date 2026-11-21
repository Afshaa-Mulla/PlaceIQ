import json
import traceback
from app.core.config import get_gemini_key, get_gemini_model, get_gemini_model_candidates
from app.services.data_service import load_students, compute_role_matches_for_student
from app.ml.predictor import predict_student_profile


def _call_gemini(prompt: str, system_instruction: str = "") -> tuple[bool, str]:
    api_key = get_gemini_key()
    if not api_key:
        return False, (
            "Gemini API key not configured. Paste your key into the `.env` file "
            "(`GEMINI_API_KEY=AIza...`) and restart Uvicorn to unlock live AI responses. "
            "All ML predictions, dashboards, and reports remain 100% functional without a key."
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            temperature=0.45,
            system_instruction=system_instruction or None,
        )

        last_exc = None

        for model_name in get_gemini_model_candidates():
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

                text = (resp.text or "").strip()
                return True, text or "No response returned from Gemini."

            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()

                if not any(
                    k in msg
                    for k in (
                        "404",
                        "not found",
                        "not_found",
                        "503",
                        "unavailable",
                        "overloaded",
                        "429",
                        "quota",
                        "resource_exhausted",
                    )
                ):
                    break

        return False, (
            f"Gemini API error: {last_exc}. "
            "Verify GEMINI_API_KEY / GEMINI_MODEL in `.env` "
            "and your internet connection."
        )

    except ImportError:
        return False, (
            "The `google-genai` package is not installed. "
            "Run `pip install -U google-genai` and restart the server."
        )

    except Exception as exc:
        return False, (
            f"Gemini API error: {exc}. "
            "Verify your GEMINI_API_KEY in `.env` and internet connection."
        )


def get_student_context(student_id: str) -> dict:
    df = load_students()
    clean_id = (student_id or "STU0001").strip().upper()
    match = df[df["student_id"].astype(str).str.upper() == clean_id]
    if match.empty:
        row = df.iloc[0].to_dict()
    else:
        row = match.iloc[0].to_dict()
    pred = predict_student_profile(row)
    roles = compute_role_matches_for_student(row, top_n=5)
    return {"student": row, "prediction": pred, "top_roles": roles}


def chat_with_career_assistant(student_id: str, message: str, history: list | None = None) -> dict:
    try:
        ctx = get_student_context(student_id)
        s = ctx["student"]
        p = ctx["prediction"]
        top_roles_summary = [
            f"{r['company_name']} - {r['role_title']} ({r['package_lpa']} LPA, Fit {r['fit_score']}%, Gaps: {[g['label']+': -'+str(g['gap']) for g in r['skill_gaps'] if g['gap']>0]})"
            for r in ctx["top_roles"][:3]
        ]
        prompt = f"""Student Profile Context:
- ID & Name: {s['student_id']} ({s['name']}) | Branch: {s['branch']} ({s['branch_name']}) | Batch: {s['batch']}
- Academics: CGPA {s['cgpa']}/10, 10th {s['tenth_pct']}%, 12th {s['twelfth_pct']}%, Active Backlogs: {s['backlogs']}
- Technical Skills (0-100): Python {s['python_score']}, SQL {s['sql_score']}, ML {s['ml_score']}, DSA {s['dsa_score']}, Web {s['web_score']}, Cloud {s['cloud_score']}, Core Eng {s['core_eng_score']}
- Soft Skills (0-100): Communication {s['communication_score']}, Aptitude {s['aptitude_score']}, Teamwork {s['teamwork_score']}
- Experience: {s['projects_count']} projects, {s['internships_count']} internships, {s['certifications_count']} certifications, {s['hackathons_count']} hackathons, Coding Rating {s['coding_rating']}
- ML Readiness Score: {p['readiness_score']}/100 ({p['readiness_band']})
- ML Placement Probability: {p['placement_probability']}% | Predicted Package Tier: {p['predicted_package_band']} (~{p['expected_lpa']} LPA)
- Top Matched Roles: {'; '.join(top_roles_summary)}

Conversation History: {json.dumps((history or [])[-6:])}
Student Question: {message}

Provide a clear, structured, personalized answer grounded in the exact numbers above."""
        system_inst = (
            "You are PlaceIQ Career Intelligence AI, an expert campus placement advisor and "
            "analytics assistant. Ground every recommendation strictly in the student and cohort "
            "metrics provided. Be concise, specific, actionable, and encouraging."
        )
        ok, reply = _call_gemini(prompt, system_instruction=system_inst)
        return {"configured": bool(get_gemini_key()), "success": ok, "reply": reply, "student_id": s["student_id"]}
    except Exception as exc:
        traceback.print_exc()
        return {
            "configured": bool(get_gemini_key()),
            "success": False,
            "reply": f"Assistant error ({type(exc).__name__}): {exc}",
            "student_id": (student_id or "STU0001").strip().upper(),
        }


def generate_30_60_90_roadmap(student_id: str, target_role: str = "") -> dict:
    try:
        ctx = get_student_context(student_id)
        s = ctx["student"]
        p = ctx["prediction"]
        chosen_role = target_role or (
            ctx["top_roles"][0]["role_title"] + " at " + ctx["top_roles"][0]["company_name"]
            if ctx["top_roles"]
            else "Software / Core Engineer"
        )

        prompt = f"""Create a concrete, personalized 30-60-90 Day Placement Preparation Roadmap for:
- Student: {s['name']} ({s['student_id']}) | Branch: {s['branch']} | Batch: {s['batch']}
- Target Role: {chosen_role}
- Current Metrics: CGPA {s['cgpa']}, Backlogs {s['backlogs']}, Readiness {p['readiness_score']}/100, Placement Prob {p['placement_probability']}%
- Skills: Python {s['python_score']}, SQL {s['sql_score']}, ML {s['ml_score']}, DSA {s['dsa_score']}, Web {s['web_score']}, Cloud {s['cloud_score']}, Core Eng {s['core_eng_score']}, Communication {s['communication_score']}, Aptitude {s['aptitude_score']}
- Top Drags: {json.dumps(p['top_drags'])}

Format using clean Markdown sections:
### Days 1–30: Foundation & Immediate Skill Gaps
### Days 31–60: Projects, Mock Drives & Portfolio Proof
### Days 61–90: Company-Specific Sprints & Interview Mastery
Include weekly targets, measurable milestones, and specific topics based on their lowest scores."""
        system_inst = (
            "You are PlaceIQ Career Intelligence AI, an expert campus placement advisor and "
            "analytics assistant. Ground every recommendation strictly in the student and cohort "
            "metrics provided. Be concise, specific, actionable, and encouraging."
        )
        ok, reply = _call_gemini(prompt, system_instruction=system_inst)
        return {"configured": bool(get_gemini_key()), "success": ok, "roadmap": reply, "target_role": chosen_role}
    except Exception as exc:
        traceback.print_exc()
        return {
            "configured": bool(get_gemini_key()),
            "success": False,
            "roadmap": f"Roadmap error ({type(exc).__name__}): {exc}",
            "target_role": target_role,
        }


def ask_dashboard_nlq(question: str) -> dict:
    df = load_students()
    branch_stats = (
        df.groupby("branch")
        .agg(
            students=("student_id", "count"),
            placement_rate=("placed", lambda x: round(x.mean() * 100, 1)),
            avg_cgpa=("cgpa", lambda x: round(x.mean(), 2)),
            avg_readiness=("readiness_score", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    placed_df = df[df["placed"] == 1]
    pkg_by_branch = (
        placed_df.groupby("branch")["package_lpa"].mean().round(2).to_dict()
        if not placed_df.empty else {}
    )
    batch_stats = (
        df.groupby("batch")
        .agg(
            students=("student_id", "count"),
            placement_rate=("placed", lambda x: round(x.mean() * 100, 1)),
            avg_readiness=("readiness_score", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary_context = {
        "total_students": int(len(df)),
        "overall_placement_rate_pct": round(float(df["placed"].mean() * 100), 1),
        "avg_placed_package_lpa": round(float(placed_df["package_lpa"].mean()), 2) if not placed_df.empty else 0,
        "max_package_lpa": round(float(placed_df["package_lpa"].max()), 2) if not placed_df.empty else 0,
        "branch_breakdown": branch_stats,
        "avg_package_lpa_by_branch": pkg_by_branch,
        "batch_trends": batch_stats,
    }
    prompt = f"""You are a Placement Cell Data Analyst. Answer the placement officer's question strictly using this cohort data:
{json.dumps(summary_context, indent=2)}

Question: {question}
Give a crisp, executive answer with exact figures."""
    system_inst = (
        "You are PlaceIQ Career Intelligence AI, an expert campus placement advisor and "
        "analytics assistant. Ground every recommendation strictly in the student and cohort "
        "metrics provided. Be concise, specific, actionable, and encouraging."
    )
    ok, reply = _call_gemini(prompt, system_instruction=system_inst)
    return {"configured": bool(get_gemini_key()), "success": ok, "answer": reply, "context_snapshot": summary_context}


def generate_batch_executive_report(batch: str = "ALL", branch: str = "ALL") -> dict:
    df = load_students()
    if batch != "ALL":
        df = df[df["batch"] == int(batch)]
    if branch != "ALL":
        df = df[df["branch"] == branch]
    placed_df = df[df["placed"] == 1]
    stats = {
        "filter_batch": batch,
        "filter_branch": branch,
        "cohort_size": int(len(df)),
        "placed_count": int(df["placed"].sum()),
        "placement_rate_pct": round(float(df["placed"].mean() * 100), 1) if len(df) else 0,
        "avg_package_lpa": round(float(placed_df["package_lpa"].mean()), 2) if len(placed_df) else 0,
        "readiness_distribution": df["readiness_band"].value_counts().to_dict(),
        "avg_skills": {
            "Python": round(float(df["python_score"].mean()), 1),
            "SQL": round(float(df["sql_score"].mean()), 1),
            "DSA": round(float(df["dsa_score"].mean()), 1),
            "ML": round(float(df["ml_score"].mean()), 1),
            "Cloud": round(float(df["cloud_score"].mean()), 1),
            "Core Eng": round(float(df["core_eng_score"].mean()), 1),
            "Communication": round(float(df["communication_score"].mean()), 1),
        },
        "top_recruiters": placed_df["placed_company"].value_counts().head(6).to_dict(),
    }
    prompt = f"""Write an executive Placement Cell Briefing Report based on these metrics:
{json.dumps(stats, indent=2)}

Include:
1. Executive Headline Summary
2. Key Strengths & Recruiter Traction
3. Critical Skill Bottlenecks & At-Risk Cohort
4. 3 Recommended Training Interventions for the Training & Placement (T&P) Cell"""
    system_inst = (
        "You are PlaceIQ Career Intelligence AI, an expert campus placement advisor and "
        "analytics assistant. Ground every recommendation strictly in the student and cohort "
        "metrics provided. Be concise, specific, actionable, and encouraging."
    )
    ok, reply = _call_gemini(prompt, system_instruction=system_inst)
    return {"configured": bool(get_gemini_key()), "success": ok, "report": reply, "stats": stats}