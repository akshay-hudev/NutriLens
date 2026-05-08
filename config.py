import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent
_load_dotenv(BASE_DIR / ".env")


def _project_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


class Config:
    BASE_DIR = BASE_DIR
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATE_DIR = BASE_DIR / "templates"

    UPLOAD_ROOT = _project_path(os.environ.get("NUTRILENS_UPLOAD_ROOT", STATIC_DIR / "uploads"))
    RUNS_ROOT = UPLOAD_ROOT / "runs"
    MAX_IMAGES = int(os.environ.get("NUTRILENS_MAX_IMAGES", "5"))
    MAX_IMAGE_BYTES = int(os.environ.get("NUTRILENS_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "nutrilens-dev-secret")
    MAX_CONTENT_LENGTH = (MAX_IMAGE_BYTES * MAX_IMAGES) + (1024 * 1024)

    MODEL_PATH = _project_path(
        os.environ.get("NUTRILENS_MODEL_PATH", BASE_DIR / "model_trained_101class.hdf5")
    )
    NUTRITION_CSV = _project_path(os.environ.get("NUTRILENS_NUTRITION_CSV", BASE_DIR / "nutrition101.csv"))

    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.environ.get("NUTRILENS_GEMINI_MODEL", "gemini-2.0-flash")
    ENABLE_GEMINI = _env_bool("NUTRILENS_ENABLE_GEMINI", bool(GOOGLE_API_KEY))

    DEFAULT_PLATE_DIAMETER_CM = float(os.environ.get("NUTRILENS_PLATE_DIAMETER_CM", "26"))
    DEFAULT_FRAME_DIAMETER_CM = float(os.environ.get("NUTRILENS_FRAME_DIAMETER_CM", "32"))
    DEFAULT_DENSITY_G_PER_CM3 = float(os.environ.get("NUTRILENS_DEFAULT_DENSITY", "0.85"))

    RECONSTRUCTION_BACKEND = os.environ.get("NUTRILENS_RECON_BACKEND", "gsplat").lower()
    COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
    ENABLE_COLMAP = _env_bool("NUTRILENS_ENABLE_COLMAP", False)
    ENABLE_GSPLAT = _env_bool("NUTRILENS_ENABLE_GSPLAT", False)
    ENABLE_NERF = _env_bool("NUTRILENS_ENABLE_NERF", False)
    RECONSTRUCTION_TIMEOUT_SECONDS = int(
        os.environ.get("NUTRILENS_RECON_TIMEOUT_SECONDS", "90")
    )

    _BENCHMARK_ROOT_RAW = os.environ.get("NUTRILENS_BENCHMARK_DATA_ROOT", "")
    BENCHMARK_DATA_ROOT = _project_path(_BENCHMARK_ROOT_RAW) if _BENCHMARK_ROOT_RAW else None
    BENCHMARK_NAME = os.environ.get("NUTRILENS_BENCHMARK_NAME", "MetaFood3D-style")

    @classmethod
    def init_app(cls, app) -> None:
        cls.UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        cls.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        app.config.from_object(cls)
