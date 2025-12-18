from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import PATHS
from src.recommender import load_model


@st.cache_data(show_spinner="Loading corpus data...")
def load_corpus():
    """Load restaurant review corpus with caching."""
    return pd.read_parquet(PATHS.CORPUS_PARQUET)


@st.cache_resource(show_spinner="Loading ML models...")
def load_artifacts():
    """Load TF-IDF models and index with caching."""
    return load_model(PATHS.TFIDF_VECTORIZER, PATHS.TFIDF_MATRIX, PATHS.RESTAURANT_INDEX)


def main():
    st.title("🔎 Recommendation Insights")
    st.markdown("Understand why restaurants are recommended by exploring their key characteristics.")

    # Load data
    corpus_df = load_corpus()
    vectorizer, tfidf_matrix, index = load_artifacts()

    # Restaurant selection
    restaurant_list = sorted([r for r in corpus_df["Restaurant"].astype(str).unique() if r in index])
    if not restaurant_list:
        st.error("No restaurants available for analysis.")
        return

    st.header("🔍 Restaurant Analysis")
    selected_restaurant = st.selectbox(
        "Choose a restaurant to analyze:",
        restaurant_list,
        help="Select a restaurant to see what makes it unique"
    )

    if selected_restaurant:
        # Get TF-IDF vector
        row = index[selected_restaurant]
        vec = tfidf_matrix[row].toarray().ravel()
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Settings
        col1, col2 = st.columns([1, 2])
        with col1:
            top_k = st.slider(
                "Number of top keywords to show",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                help="How many keywords to display"
            )

        # Calculate top keywords
        top_idx = np.argsort(vec)[::-1][:top_k]
        keywords = pd.DataFrame({
            "keyword": feature_names[top_idx],
            "tfidf_weight": vec[top_idx],
        }).sort_values("tfidf_weight", ascending=False)

        with col2:
            st.metric("Vocabulary Size", f"{len(feature_names):,}")
            st.metric("Selected Restaurant", selected_restaurant)

        # Display results
        st.header("📊 Top TF-IDF Keywords")
        st.markdown("These keywords represent the most important terms that define this restaurant based on customer reviews.")

        # Format for display
        display_df = keywords.copy()
        display_df["tfidf_weight"] = display_df["tfidf_weight"].round(4)

        st.dataframe(
            display_df,
            width='stretch',
            column_config={
                "keyword": st.column_config.TextColumn("Keyword", width="medium"),
                "tfidf_weight": st.column_config.NumberColumn("TF-IDF Weight", format="%.4f")
            }
        )

        # Additional insights
        with st.expander("ℹ️ How to interpret these results"):
            st.markdown("""
            - **TF-IDF Weight**: Higher values indicate terms that are important for this restaurant but not common across all restaurants.
            - **Keywords reflect**: Common themes in customer reviews like cuisine type, atmosphere, service quality, etc.
            - **Use for discovery**: Look for restaurants with similar high-weight keywords for comparable experiences.
            """)

        # Word cloud visualization (optional enhancement)
        if st.checkbox("Show word cloud visualization"):
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                # Create word cloud from top keywords
                word_freq = dict(zip(keywords["keyword"], keywords["tfidf_weight"] * 100))
                wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"Word Cloud for {selected_restaurant}", fontsize=16)
                st.pyplot(fig)
            except ImportError:
                st.info("Install 'wordcloud' package for word cloud visualization: `pip install wordcloud`")


if __name__ == "__main__":
    main()