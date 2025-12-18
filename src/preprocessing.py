from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class Schema:
    restaurant: str = "Restaurant"
    reviewer: str = "Reviewer"
    review: str = "Review"
    rating: str = "Rating"
    metadata: str = "Metadata"
    time: str = "Time"
    pictures: str = "Pictures"


SCHEMA = Schema()


_TEXT_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean_text(s: str) -> str:
    s = s.strip()
    s = _URL_RE.sub("", s)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = _TEXT_WS_RE.sub(" ", s)
    return s


def _parse_metadata(meta: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parses strings like: '1 Review , 2 Followers' into (reviews, followers).
    Returns (None, None) if parsing fails.
    """
    if meta is None or (isinstance(meta, float) and np.isnan(meta)):
        return None, None

    text = str(meta)
    reviews_match = re.search(r"(\d+)\s*Review", text, flags=re.IGNORECASE)
    followers_match = re.search(r"(\d+)\s*Follower", text, flags=re.IGNORECASE)

    reviews = int(reviews_match.group(1)) if reviews_match else None
    followers = int(followers_match.group(1)) if followers_match else None
    return reviews, followers


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the dataset:
    - Drop unnamed/garbage columns
    - Convert rating to numeric
    - Parse time to datetime
    - Clean review text
    - Parse metadata into numeric features
    """
    df = df.copy()

    # Drop unnamed / garbage columns (like '7514' or 'Unnamed: 0')
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") or c.strip() == "7514"]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Ensure expected columns exist
    required = [SCHEMA.restaurant, SCHEMA.reviewer, SCHEMA.review, SCHEMA.rating, SCHEMA.time]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Rating -> numeric
    df[SCHEMA.rating] = pd.to_numeric(df[SCHEMA.rating], errors="coerce")

    # Time -> datetime (coerce invalid)
    df[SCHEMA.time] = pd.to_datetime(df[SCHEMA.time], errors="coerce")

    # Text cleaning
    df[SCHEMA.review] = df[SCHEMA.review].astype(str).map(_clean_text)

    # Metadata parsing
    if SCHEMA.metadata in df.columns:
        parsed = df[SCHEMA.metadata].map(_parse_metadata)
        df["reviewer_total_reviews"] = parsed.map(lambda x: x[0])
        df["reviewer_followers"] = parsed.map(lambda x: x[1])
    else:
        df["reviewer_total_reviews"] = None
        df["reviewer_followers"] = None

    # Pictures -> numeric if exists
    if SCHEMA.pictures in df.columns:
        df[SCHEMA.pictures] = pd.to_numeric(df[SCHEMA.pictures], errors="coerce")

    # Basic sanity: drop rows with empty restaurant or review
    df[SCHEMA.restaurant] = df[SCHEMA.restaurant].astype(str).str.strip()
    df = df[df[SCHEMA.restaurant].str.len() > 0]
    df = df[df[SCHEMA.review].str.len() > 0]

    logger.info("Preprocessed reviews shape=%s", df.shape)
    return df