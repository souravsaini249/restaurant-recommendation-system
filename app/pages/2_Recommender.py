from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import CFG, PATHS
from src.recommender import load_model, recommend_from_preferences, recommend_similar_restaurants
from src.components.ui_helpers import render_reco_table
import traceback


def _ensure_file(p):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


@st.cache_data(show_spinner="Loading processed data...")
def load_processed():
    """Load restaurant profiles and corpus with caching."""
    profiles = pd.read_parquet(PATHS.PROFILES_PARQUET)
    corpus = pd.read_parquet(PATHS.CORPUS_PARQUET)
    return profiles, corpus


@st.cache_resource(show_spinner="Loading ML models...")
def load_artifacts():
    """Load TF-IDF models and index with caching."""
    return load_model(PATHS.TFIDF_VECTORIZER, PATHS.TFIDF_MATRIX, PATHS.RESTAURANT_INDEX)


def main():
    st.title("🎯 Restaurant Recommender")
    st.markdown("Find restaurants tailored to your tastes using AI-powered recommendations.")

    # Load data with explicit checks and clearer errors
    try:
        _ensure_file(PATHS.PROFILES_PARQUET)
        _ensure_file(PATHS.CORPUS_PARQUET)
        _ensure_file(PATHS.TFIDF_VECTORIZER)
        _ensure_file(PATHS.TFIDF_MATRIX)
        _ensure_file(PATHS.RESTAURANT_INDEX)

        profiles, corpus = load_processed()
        vectorizer, tfidf_matrix, index = load_artifacts()
    except Exception as e:
        st.error("Failed to load required data or model artifacts.")
        st.exception(e)
        st.text("Full traceback:")
        st.text(traceback.format_exc())
        return

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    top_n = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=25,
        value=CFG.TOP_N_DEFAULT,
        step=1,
        help="How many restaurants to recommend"
    )

    # Main content with tabs
    tab1, tab2 = st.tabs(["🔍 Similar Restaurants", "📝 From Preferences"])

    with tab1:
        st.header("Find Similar Restaurants")
        st.markdown("Choose a restaurant you like, and we'll find others with similar characteristics.")

        # Restaurant selection
        restaurant_list = sorted([r for r in profiles["Restaurant"].astype(str).unique() if r in index])
        if not restaurant_list:
            st.error("No restaurants available for recommendations.")
            return

        seed = st.selectbox(
            "Select a restaurant you enjoy:",
            restaurant_list,
            help="Choose from restaurants in our database"
        )

        # Show selected restaurant info
        if seed:
            rest_info = profiles[profiles["Restaurant"] == seed].iloc[0]
            st.info(f"**{seed}** - Rating: {rest_info['avg_rating']:.1f} ⭐, {rest_info['num_reviews']} reviews")

        # Recommendation button
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding similar restaurants..."):
                try:
                    recs = recommend_similar_restaurants(seed, profiles, tfidf_matrix, index, top_n=top_n)
                    if recs:
                        out = pd.DataFrame([r.__dict__ for r in recs])
                        st.success(f"✅ Found {len(recs)} restaurants similar to **{seed}**")
                        render_reco_table(out)
                    else:
                        st.warning("No recommendations found. Try a different restaurant.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("Recommend from Your Preferences")
        st.markdown("Describe what you're looking for in natural language, and we'll find matching restaurants.")

        # Text input
        user_text = st.text_area(
            "Describe your ideal restaurant experience:",
            placeholder="e.g., 'spicy chicken, family dinner, good ambience, affordable prices'",
            height=120,
            help="Be as specific as possible for better recommendations"
        )

        # Example suggestions
        with st.expander("💡 Example preferences"):
            st.markdown("""
            - "romantic dinner with wine and steak"
            - "quick lunch, healthy options, vegetarian"
            - "family-friendly, kids menu, casual atmosphere"
            - "spicy food, authentic cuisine, local favorites"
            """)

        # Recommendation button
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            if not user_text.strip():
                st.error("Please enter some preferences first.")
                return

            with st.spinner("Analyzing your preferences..."):
                try:
                    recs = recommend_from_preferences(user_text, profiles, vectorizer, tfidf_matrix, index, top_n=top_n)
                    if recs:
                        out = pd.DataFrame([r.__dict__ for r in recs])
                        st.success(f"✅ Found {len(recs)} restaurants matching your preferences")
                        render_reco_table(out)
                    else:
                        st.warning("No recommendations found. Try different keywords or be less specific.")
                except ValueError as e:
                    st.error(f"Input error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()