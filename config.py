import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Ensure environment variables are loaded
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "development-key-change-me")

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'instance' / 'attendance.db'}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {
            "timeout": 30  # Wait up to 30s to resolve locks with DB Browser
        }
    }

    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload limit

    TEMP_DIR = BASE_DIR / "instance" / "tmp"
