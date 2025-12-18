from __future__ import annotations

import pandas as pd

from src.config import PATHS
from src.ingestion import load_raw_csv
from src.preprocessing import preprocess_reviews
from src.feature_engineering import build_restaurant_profiles, build_restaurant_corpus
from src.recommender import train_tfidf, save_model
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    # Ensure folders exist
    ensure_dir(PATHS.PROCESSED_DIR)
    ensure_dir(PATHS.MODELS_DIR)

    # 1) Load raw
    df_raw = load_raw_csv(PATHS.RAW_CSV)

    # 2) Clean
    df_clean = preprocess_reviews(df_raw)
    df_clean.to_parquet(PATHS.CLEAN_PARQUET, index=False)
    logger.info("Saved: %s", PATHS.CLEAN_PARQUET)

    # 3) Profiles + corpus
    profiles = build_restaurant_profiles(df_clean)
    corpus = build_restaurant_corpus(df_clean)

    profiles.to_parquet(PATHS.PROFILES_PARQUET, index=False)
    corpus.to_parquet(PATHS.CORPUS_PARQUET, index=False)
    logger.info("Saved: %s", PATHS.PROFILES_PARQUET)
    logger.info("Saved: %s", PATHS.CORPUS_PARQUET)

    # 4) Train TF-IDF + save artifacts
    vectorizer, tfidf_matrix, index = train_tfidf(corpus)

    save_model(
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        index=index,
        vectorizer_path=PATHS.TFIDF_VECTORIZER,
        matrix_path=PATHS.TFIDF_MATRIX,
        index_path=PATHS.RESTAURANT_INDEX,
    )
    logger.info("Saved TF-IDF artifacts into: %s", PATHS.MODELS_DIR)

    logger.info("Pipeline complete ✅")


if __name__ == "__main__":
    main()