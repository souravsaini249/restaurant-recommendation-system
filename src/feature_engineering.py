from __future__ import annotations

import pandas as pd

from .preprocessing import SCHEMA
from .utils import get_logger


logger = get_logger(__name__)


def build_restaurant_profiles(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-restaurant profile features for ranking.
    """
    g = df_clean.groupby(SCHEMA.restaurant, dropna=True)

    profiles = g.agg(
        avg_rating=(SCHEMA.rating, "mean"),
        num_reviews=(SCHEMA.review, "count"),
        latest_review_date=(SCHEMA.time, "max"),
        sample_review=(SCHEMA.review, lambda x: x.iloc[0] if len(x) > 0 else ""),
    ).reset_index()

    # Handle missing rating gracefully
    profiles["avg_rating"] = profiles["avg_rating"].fillna(profiles["avg_rating"].median())
    profiles["num_reviews"] = profiles["num_reviews"].fillna(0).astype(int)

    logger.info("Built restaurant profiles: shape=%s", profiles.shape)
    return profiles


def build_restaurant_corpus(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all reviews per restaurant into a single text corpus.
    """
    corpus = (
        df_clean.groupby(SCHEMA.restaurant, dropna=True)[SCHEMA.review]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
        .rename(columns={SCHEMA.review: "corpus"})
    )

    corpus["corpus"] = corpus["corpus"].astype(str).str.strip()
    logger.info("Built restaurant corpus: shape=%s", corpus.shape)
    return corpus