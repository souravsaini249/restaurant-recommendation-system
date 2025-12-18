from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CFG
from .utils import get_logger, read_json, write_json

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecoResult:
    restaurant: str
    final_score: float
    similarity: float
    avg_rating: float
    num_reviews: int
    sample_review: str


def _minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi - lo < 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def train_tfidf(corpus_df: pd.DataFrame) -> Tuple[TfidfVectorizer, sparse.csr_matrix, Dict[str, int]]:
    """
    Train TF-IDF on restaurant corpus.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Must contain columns: ['Restaurant', 'corpus']

    Returns
    -------
    vectorizer : TfidfVectorizer
    tfidf_matrix : csr_matrix
    index : dict[str, int] mapping restaurant -> row id in matrix
    """
    if "Restaurant" not in corpus_df.columns or "corpus" not in corpus_df.columns:
        raise ValueError("corpus_df must contain columns: ['Restaurant', 'corpus']")

    restaurants = corpus_df["Restaurant"].astype(str).tolist()
    texts = corpus_df["corpus"].astype(str).tolist()

    # For very small corpora (e.g., unit tests with 3 docs), min_df=2 can prune everything.
    # Keep strong defaults for real datasets, but adapt safely for tiny inputs.
    n_docs = len(texts)
    effective_min_df = 2 if n_docs >= 20 else 1

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=effective_min_df,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    index = {r: i for i, r in enumerate(restaurants)}
    logger.info(
        "Trained TF-IDF: restaurants=%d | vocab=%d | min_df=%d",
        len(restaurants),
        len(vectorizer.vocabulary_),
        effective_min_df,
    )
    return vectorizer, tfidf_matrix, index


def save_model(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.csr_matrix,
    index: Dict[str, int],
    vectorizer_path,
    matrix_path,
    index_path,
) -> None:
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)
    write_json(index_path, index)
    logger.info("Saved model artifacts to models/ directory")


def load_model(vectorizer_path, matrix_path, index_path) -> Tuple[TfidfVectorizer, sparse.csr_matrix, Dict[str, int]]:
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)
    index = read_json(index_path)
    return vectorizer, tfidf_matrix, index


def recommend_similar_restaurants(
    seed_restaurant: str,
    profiles_df: pd.DataFrame,
    tfidf_matrix: sparse.csr_matrix,
    index: Dict[str, int],
    top_n: int = 10,
) -> List[RecoResult]:
    """
    Recommend restaurants similar to a given restaurant based on TF-IDF cosine similarity
    + hybrid ranking with rating and popularity.
    """
    if seed_restaurant not in index:
        raise ValueError(f"Unknown restaurant: {seed_restaurant}")

    seed_idx = index[seed_restaurant]
    sims = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).ravel()

    df = profiles_df.copy()
    df["similarity"] = df["Restaurant"].map(lambda r: float(sims[index[r]]) if r in index else 0.0)

    # remove itself
    df = df[df["Restaurant"] != seed_restaurant]

    # normalize rating and popularity
    df["rating_norm"] = _minmax(df["avg_rating"])
    df["pop_norm"] = _minmax(np.log1p(df["num_reviews"].clip(lower=0)))

    df["final_score"] = (
        CFG.W_SIM * df["similarity"]
        + CFG.W_RATING * df["rating_norm"]
        + CFG.W_POP * df["pop_norm"]
    )

    df = df.sort_values("final_score", ascending=False).head(top_n)

    results: List[RecoResult] = []
    for _, row in df.iterrows():
        results.append(
            RecoResult(
                restaurant=str(row["Restaurant"]),
                final_score=float(row["final_score"]),
                similarity=float(row["similarity"]),
                avg_rating=float(row["avg_rating"]),
                num_reviews=int(row["num_reviews"]),
                sample_review=str(row["sample_review"]),
            )
        )
    return results


def recommend_from_preferences(
    user_text: str,
    profiles_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.csr_matrix,
    index: Dict[str, int],
    top_n: int = 10,
) -> List[RecoResult]:
    """
    Recommend restaurants based on user's free-text preferences.
    """
    query = (user_text or "").strip()
    if len(query) < 3:
        raise ValueError("Please enter a longer preference text (at least 3 characters).")

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()

    df = profiles_df.copy()
    df["similarity"] = df["Restaurant"].map(lambda r: float(sims[index[r]]) if r in index else 0.0)

    df["rating_norm"] = _minmax(df["avg_rating"])
    df["pop_norm"] = _minmax(np.log1p(df["num_reviews"].clip(lower=0)))

    df["final_score"] = (
        CFG.W_SIM * df["similarity"]
        + CFG.W_RATING * df["rating_norm"]
        + CFG.W_POP * df["pop_norm"]
    )

    df = df.sort_values("final_score", ascending=False).head(top_n)

    results: List[RecoResult] = []
    for _, row in df.iterrows():
        results.append(
            RecoResult(
                restaurant=str(row["Restaurant"]),
                final_score=float(row["final_score"]),
                similarity=float(row["similarity"]),
                avg_rating=float(row["avg_rating"]),
                num_reviews=int(row["num_reviews"]),
                sample_review=str(row["sample_review"]),
            )
        )
    return results