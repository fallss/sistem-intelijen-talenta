"""
Configuration module for Talent Match Intelligence System.
Safely loads environment variables, Streamlit secrets, and defines domain constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Load .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


def get_secret(key: str, default: str = "") -> str:
    """Safely retrieves a configuration key from environment variables or Streamlit secrets."""
    # 1. Check environment variables
    val = os.getenv(key)
    if val:
        return val.strip()

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    return default


# API & Database Credentials
DATABASE_URL: str = get_secret("DATABASE_URL")
OPENROUTER_API_KEY: str = get_secret("OPENROUTER_API_KEY")
OPENROUTER_MODEL: str = get_secret("OPENROUTER_MODEL", "openai/gpt-4o-mini")
GOOGLE_API_KEY: str = get_secret("GOOGLE_API_KEY")
SUPABASE_URL: str = get_secret("SUPABASE_URL")
SUPABASE_KEY: str = get_secret("SUPABASE_KEY")

# Domain Metric Mapping for Talent Intelligence
# Separates true psychological & assessment scores from structural entity IDs
TALENT_METRIC_COLUMNS = [
    "avg_competency_score",
    "iq",
    "gtq",
    "tiki",
    "pauli",
    "faxtor",
    "years_of_service_months",
    "rating",
    "leadership_score",
    "communication_score",
    "cognitive_score",
]

# Structural ID / categorical columns to exclude from mathematical scoring
EXCLUDED_SCORING_COLUMNS = [
    "employee_id",
    "fullname",
    "nip",
    "company_id",
    "area_id",
    "position_id",
    "department_id",
    "division_id",
    "directorate_id",
    "grade_id",
    "education_id",
    "major_id",
    "year",
    "year_competency",
    "disc",
    "disc_word",
    "mbti",
]

# UI Theme Brand Colors
BRAND_COLORS = {
    "primary": "#4F46E5",     # Indigo
    "secondary": "#06B6D4",   # Cyan
    "accent": "#10B981",      # Emerald
    "danger": "#EF4444",      # Rose
    "warning": "#F59E0B",     # Amber
    "dark": "#1E293B",        # Slate Dark
    "light": "#F8FAFC",       # Slate Light
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0"
}
