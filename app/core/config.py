import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

APP_NAME = "PlaceIQ"
APP_TAGLINE = "AI-Powered Campus Placement Intelligence & Career Readiness Suite"
SYNTHETIC_NOTICE = (
    "Dataset Notice: 100% synthetic multi-branch dataset (1,800 students across "
    "4 batches & 7 engineering branches, 45 companies, 57 roles) engineered for "
    "academic demonstration."
)

DEFAULT_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


def get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        return ""
    return key


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODELS[0]


def get_gemini_model_candidates() -> list:
    out = []
    for m in [get_gemini_model()] + DEFAULT_GEMINI_MODELS:
        if m and m not in out:
            out.append(m)
    return out