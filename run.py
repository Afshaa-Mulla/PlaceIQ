"""
PlaceIQ — Campus Placement Analytics, Readiness & Career Intelligence Platform
Run with: python run.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ensure_artifacts():
    data_file = ROOT / "data" / "raw" / "students_full.csv"
    model_file = ROOT / "models" / "placement_model.joblib"
    if not data_file.exists():
        print("[PlaceIQ] Synthetic dataset missing — generating 1,800 multi-branch records...")
        from scripts.generate_data import generate_all
        generate_all()
    if not model_file.exists():
        print("[PlaceIQ] Trained models missing — running benchmark & training pipeline...")
        from scripts.train_models import train_and_evaluate
        train_and_evaluate()


if __name__ == "__main__":
    ensure_artifacts()
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"\n>>> PlaceIQ running at http://{host}:{port} <<<\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
