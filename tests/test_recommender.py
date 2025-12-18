from __future__ import annotations

import pandas as pd

from src.recommender import train_tfidf, recommend_from_preferences


def test_recommend_from_preferences_runs():
    corpus = pd.DataFrame({
        "Restaurant": ["A", "B", "C"],
        "corpus": ["spicy chicken rice", "romantic wine ambience", "quick lunch cheap"],
    })
    vectorizer, tfidf_matrix, index = train_tfidf(corpus)

    profiles = pd.DataFrame({
        "Restaurant": ["A", "B", "C"],
        "avg_rating": [4.5, 4.0, 3.8],
        "num_reviews": [100, 50, 30],
        "latest_review_date": [None, None, None],
    })

    recs = recommend_from_preferences("spicy chicken", profiles, vectorizer, tfidf_matrix, index, top_n=2)
    assert len(recs) == 2
    assert recs[0].restaurant in {"A", "B", "C"}