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
SYNTHETIC_NOTICE = "Dataset Notice: 100% synthetic multi-branch dataset (1,800 students across 4 batches & 7 engineering branches, 45 companies, 57 roles) engineered for academic demonstration."

def get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        return ""
    return key

def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
