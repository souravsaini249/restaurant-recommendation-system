from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .recommender import recommend_from_preferences, recommend_similar_restaurants
from .utils import get_logger


logger = get_logger(__name__)


def basic_coverage_metrics(profiles_df: pd.DataFrame, index: Dict[str, int]) -> dict:
    """
    Simple, defensible evaluation:
    - coverage: how many restaurants exist in both profiles and model index
    """
    total = len(profiles_df)
    in_index = profiles_df["Restaurant"].isin(index.keys()).sum()
    coverage = in_index / total if total else 0.0
    return {"restaurants_total": total, "restaurants_in_index": int(in_index), "coverage": float(coverage)}


def sample_qualitative_examples(
    profiles_df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    index: Dict[str, int],
) -> List[dict]:
    """
    Provide a few example outputs for report screenshots.
    """
    examples: List[dict] = []

    # Similar restaurants example (pick first restaurant in profiles)
    seed = profiles_df["Restaurant"].iloc[0]
    sim_recs = recommend_similar_restaurants(seed, profiles_df, tfidf_matrix, index, top_n=5)
    examples.append({"type": "similar_restaurants", "seed": seed, "recommendations": [r.__dict__ for r in sim_recs]})

    # Preference text examples
    for q in ["spicy chicken family dinner", "romantic ambience wine", "quick lunch budget friendly"]:
        recs = recommend_from_preferences(q, profiles_df, vectorizer, tfidf_matrix, index, top_n=5)
        examples.append({"type": "preference_query", "query": q, "recommendations": [r.__dict__ for r in recs]})

    return examples