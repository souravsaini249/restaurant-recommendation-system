from __future__ import annotations

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #f0f8ff, #e6f7ff);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .stDataFrame {
        font-size: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #2E86AB !important;
        color: white !important;
        border-color: #2E86AB !important;
    }
    .stButton button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #2E86AB, #A23B72) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #A23B72, #2E86AB) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
        transform: translateY(-2px) !important;
    }
    .stButton button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🍽️ Navigation")
    st.markdown("---")
    st.markdown("**Pages:**")
    st.markdown("- 📊 **EDA**: Explore the dataset")
    st.markdown("- 🎯 **Recommender**: Get recommendations")
    st.markdown("- 🔎 **Insights**: Understand recommendations")
    st.markdown("---")
    st.info("Built with Streamlit & Scikit-learn")

# Main content
st.markdown("""
<div class="hero-section">
    <h1>🍽️ Welcome to Restaurant Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>How it works:</strong><br>
    This app uses a hybrid recommendation approach combining:
    - **TF-IDF similarity** on customer review text
    - **Restaurant ratings** for quality
    - **Popularity** (review count) for reliability
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Get Started")
st.markdown("Use the sidebar to navigate to different sections:")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 EDA</h4>
        <p>Analyze the dataset with visualizations and statistics.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Recommender</h4>
        <p>Find restaurants similar to your favorites or based on your preferences.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🔎 Insights</h4>
        <p>Understand why restaurants are recommended using TF-IDF keywords.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Built with ❤️ using Python, Streamlit, and machine learning.*")