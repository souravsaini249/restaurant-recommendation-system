from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PATHS
from src.components.plotting import plot_rating_distribution, plot_top_restaurants_by_reviews
from src.components.ui_helpers import render_kpis
import traceback


def _ensure_file(p):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """Load cleaned reviews and restaurant profiles with caching."""
    df_clean = pd.read_parquet(PATHS.CLEAN_PARQUET)
    profiles = pd.read_parquet(PATHS.PROFILES_PARQUET)
    return df_clean, profiles


def main():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Understand the dataset through visualizations and key metrics.")

    # Load data
    try:
        _ensure_file(PATHS.CLEAN_PARQUET)
        _ensure_file(PATHS.PROFILES_PARQUET)
        with st.spinner("Loading data..."):
            df_clean, profiles = load_data()
    except Exception as e:
        st.error("Failed to load dataset for EDA.")
        st.exception(e)
        st.text("Full traceback:")
        st.text(traceback.format_exc())
        return

    # Key metrics
    st.header("📈 Key Metrics")
    avg_rating = float(profiles["avg_rating"].mean())
    num_restaurants = int(profiles.shape[0])
    total_reviews = int(df_clean.shape[0])

    render_kpis(avg_rating, num_restaurants, total_reviews)

    # Dataset overview
    st.header("🔍 Dataset Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sample Reviews")
        # Add filter for rating
        min_rating = st.slider("Filter by minimum rating", 1.0, 5.0, 1.0, 0.5)
        filtered_df = df_clean[df_clean["Rating"] >= min_rating]
        st.dataframe(
            filtered_df.head(25)[["Restaurant", "Reviewer", "Review", "Rating", "Time"]],
            width='stretch'
        )
        st.caption(f"Showing {len(filtered_df.head(25))} of {len(filtered_df)} reviews (filtered by rating ≥ {min_rating})")

    with col2:
        st.subheader("Data Summary")
        st.metric("Total Reviews", f"{total_reviews:,}")
        st.metric("Unique Restaurants", f"{num_restaurants:,}")
        st.metric("Avg Reviews per Restaurant", f"{total_reviews / num_restaurants:.1f}")
        st.metric("Date Range", f"{df_clean['Time'].min().date()} to {df_clean['Time'].max().date()}")

    # Visualizations
    st.header("📊 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        fig = plot_rating_distribution(df_clean)
        st.pyplot(fig)
        st.caption("Distribution of customer ratings across all reviews.")

    with col2:
        st.subheader("Top Restaurants by Reviews")
        top_n = st.slider("Number of restaurants to show", 5, 20, 15)
        fig = plot_top_restaurants_by_reviews(profiles, top_n=top_n)
        st.pyplot(fig)
        st.caption(f"Most reviewed restaurants (top {top_n}).")


if __name__ == "__main__":
    main()