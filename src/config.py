from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]

    DATA_DIR: Path = ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    MODELS_DIR: Path = ROOT / "models"
    REPORTS_DIR: Path = ROOT / "reports"
    SCREENSHOTS_DIR: Path = REPORTS_DIR / "screenshots"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"

    RAW_CSV: Path = RAW_DIR / "Restaurant reviews.csv"

    CLEAN_PARQUET: Path = PROCESSED_DIR / "reviews_clean.parquet"
    PROFILES_PARQUET: Path = PROCESSED_DIR / "restaurant_profiles.parquet"
    CORPUS_PARQUET: Path = PROCESSED_DIR / "restaurant_review_corpus.parquet"

    TFIDF_VECTORIZER: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
    TFIDF_MATRIX: Path = MODELS_DIR / "tfidf_matrix.joblib"
    RESTAURANT_INDEX: Path = MODELS_DIR / "restaurant_index.json"


PATHS = Paths()


@dataclass(frozen=True)
class AppConfig:
    RANDOM_STATE: int = 42
    TOP_N_DEFAULT: int = 10

    # Recommendation scoring weights
    W_SIM: float = 0.65
    W_RATING: float = 0.25
    W_POP: float = 0.10


CFG = AppConfig()