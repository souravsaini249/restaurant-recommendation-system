from __future__ import annotations

import pandas as pd
import streamlit as st


def render_kpis(avg_rating: float, num_restaurants: int, total_reviews: int) -> None:
    """
    Render key performance indicators in a three-column layout.

    Args:
        avg_rating: Average restaurant rating
        num_restaurants: Total number of restaurants
        total_reviews: Total number of reviews
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📊 Average Rating",
            value=f"{avg_rating:.2f}",
            delta="⭐",
            help="Average rating across all restaurants"
        )

    with col2:
        st.metric(
            label="🏪 Total Restaurants",
            value=f"{num_restaurants:,}",
            help="Number of unique restaurants in dataset"
        )

    with col3:
        st.metric(
            label="💬 Total Reviews",
            value=f"{total_reviews:,}",
            help="Total number of customer reviews"
        )


def render_reco_table(df: pd.DataFrame) -> None:
    """
    Render recommendations table with formatted columns.

    Args:
        df: DataFrame containing recommendation results
    """
    if df.empty:
        st.warning("No recommendations to display.")
        return

    # Format columns for better display
    display_df = df.copy()
    display_df["restaurant"] = display_df["restaurant"].astype(str)
    display_df["final_score"] = display_df["final_score"].round(3)
    display_df["similarity"] = display_df["similarity"].round(3)
    display_df["avg_rating"] = display_df["avg_rating"].round(1)
    display_df["sample_review"] = display_df["sample_review"].astype(str)

    # Rename columns for clarity
    display_df = display_df.rename(columns={
        "restaurant": "Restaurant",
        "final_score": "Score",
        "similarity": "Similarity",
        "avg_rating": "Rating",
        "sample_review": "Reviews"
    })

    # Remove Score and Similarity columns as requested
    display_df = display_df[["Restaurant", "Rating", "Reviews"]]

    st.dataframe(
        display_df,
        width='stretch',
        column_config={
            "Restaurant": st.column_config.TextColumn("Restaurant", width="large"),
            "Rating": st.column_config.NumberColumn("Rating", format="%.1f", help="Average customer rating"),
            "Reviews": st.column_config.TextColumn("Reviews", help="Sample customer review")
        },
        hide_index=True
    )