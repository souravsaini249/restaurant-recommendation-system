from __future__ import annotations

import pandas as pd

from src.preprocessing import preprocess_reviews


def test_preprocess_reviews_basic():
    df = pd.DataFrame({
        "Restaurant": ["A", "B"],
        "Reviewer": ["u1", "u2"],
        "Review": ["Great food!", "Nice place"],
        "Rating": ["5", "4"],
        "Metadata": ["1 Review , 2 Followers", "3 Reviews , 0 Followers"],
        "Time": ["5/25/2019 15:54", "6/01/2019 10:00"],
        "Pictures": ["1", "0"],
        "7514": [None, None],
    })

    out = preprocess_reviews(df)
    assert "7514" not in out.columns
    assert out["Rating"].dtype.kind in ("i", "f")
    assert "reviewer_total_reviews" in out.columns
    assert "reviewer_followers" in out.columns
    assert out.shape[0] == 2