from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import (
    APP_NAME, APP_TAGLINE, SYNTHETIC_NOTICE,
    TEMPLATES_DIR, STATIC_DIR, get_gemini_key
)
from app.api.routes import router as api_router

app = FastAPI(title=APP_NAME, description=APP_TAGLINE, version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.include_router(api_router)

NAV_PAGES = [
    {"slug": "overview", "path": "/", "label": "Overview", "icon": "layout-dashboard", "group": "Analytics"},
    {"slug": "batch", "path": "/batch-analytics", "label": "Batch & Branch", "icon": "layers", "group": "Analytics"},
    {"slug": "explorer", "path": "/students", "label": "Student Explorer", "icon": "users", "group": "Students"},
    {"slug": "profile", "path": "/student-360", "label": "Student 360°", "icon": "user-check", "group": "Students"},
    {"slug": "predictor", "path": "/predictor", "label": "What-If Predictor", "icon": "sliders", "group": "Intelligence"},
    {"slug": "matcher", "path": "/matcher", "label": "Company Matcher", "icon": "briefcase", "group": "Intelligence"},
    {"slug": "gaps", "path": "/skill-gaps", "label": "Skill Gap Center", "icon": "target", "group": "Intelligence"},
    {"slug": "modellab", "path": "/model-lab", "label": "Model Lab & XAI", "icon": "cpu", "group": "Science"},
    {"slug": "assistant", "path": "/ai-assistant", "label": "AI Career Copilot", "icon": "sparkles", "group": "Science"},
    {"slug": "reports", "path": "/reports", "label": "Excel Reports", "icon": "file-spreadsheet", "group": "Operations"},
]


def render_page(request: Request, template_name: str, active_slug: str, page_title: str, extra: dict | None = None):
    ctx = {
        "request": request,
        "app_name": APP_NAME,
        "app_tagline": APP_TAGLINE,
        "synthetic_notice": SYNTHETIC_NOTICE,
        "nav_pages": NAV_PAGES,
        "active_slug": active_slug,
        "page_title": page_title,
        "gemini_ready": bool(get_gemini_key()),
    }
    if extra:
        ctx.update(extra)
    return templates.TemplateResponse(request, template_name, ctx)


@app.get("/", response_class=HTMLResponse)
def page_overview(request: Request):
    return render_page(request, "overview.html", "overview", "Executive Overview Dashboard")


@app.get("/batch-analytics", response_class=HTMLResponse)
def page_batch(request: Request):
    return render_page(request, "batch_analytics.html", "batch", "Multi-Branch & Batch Analytics")


@app.get("/students", response_class=HTMLResponse)
def page_explorer(request: Request):
    return render_page(request, "student_explorer.html", "explorer", "Student Cohort Explorer")


@app.get("/student-360", response_class=HTMLResponse)
def page_profile(request: Request, id: str = "STU0001"):
    return render_page(request, "student_360.html", "profile", "Student 360° Readiness Profile", {"default_student_id": id})


@app.get("/predictor", response_class=HTMLResponse)
def page_predictor(request: Request):
    return render_page(request, "predictor.html", "predictor", "Live What-If Placement Simulator")


@app.get("/matcher", response_class=HTMLResponse)
def page_matcher(request: Request, id: str = "STU0001"):
    return render_page(request, "matcher.html", "matcher", "Company & Role Fit Matcher", {"default_student_id": id})


@app.get("/skill-gaps", response_class=HTMLResponse)
def page_gaps(request: Request, id: str = "STU0001"):
    return render_page(request, "skill_gaps.html", "gaps", "Skill Gap & Curriculum Center", {"default_student_id": id})


@app.get("/model-lab", response_class=HTMLResponse)
def page_modellab(request: Request):
    return render_page(request, "model_lab.html", "modellab", "ML Benchmark & Explainability Lab")


@app.get("/ai-assistant", response_class=HTMLResponse)
def page_assistant(request: Request, id: str = "STU0001"):
    return render_page(request, "ai_assistant.html", "assistant", "Gemini AI Career Copilot", {"default_student_id": id})


@app.get("/reports", response_class=HTMLResponse)
def page_reports(request: Request):
    return render_page(request, "reports.html", "reports", "Placement Cell Excel Export Center")


@app.get("/about", response_class=HTMLResponse)
def page_about(request: Request):
    return render_page(request, "about.html", "about", "System Architecture & Methodology")
