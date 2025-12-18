# 🍽️ Restaurant Recommendation System

An AI-powered restaurant recommendation system that combines customer reviews, ratings, and popularity to suggest personalized dining experiences. Built with Python, Streamlit, and machine learning.

This application is deployed to Streamlit which can be accessed via: https://restaurantrecommendations.streamlit.app/
The dataset has been collected from Kaggle which can be accessed via: https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews/data


## ✨ Features

- **Smart Recommendations**: Hybrid approach combining TF-IDF text similarity, customer ratings, and popularity scores
- **Exploratory Data Analysis**: Interactive visualizations and statistics of the restaurant dataset
- **Insights**: Understand recommendation logic with TF-IDF keyword analysis
- **Modern UI**: Professional Streamlit interface with colorful design and smooth animations
- **Fast Processing**: Optimized with caching and efficient algorithms
- **Responsive**: Works on desktop and mobile devices

##  Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd restaurant-recommendation-system
   ```

2. **Install dependencies**:

   pip install -r requirements.txt


3. **Build the data pipeline**:

   python -m src.pipeline_build


4. **Run the application**:

   streamlit run app/app.py


5. **Open your browser** to `http://localhost:8501`

## Project Structure

```
restaurant-recommendation-system/
├── app/                    # Streamlit application
│   ├── pages/             # Multi-page app sections
│   │   ├── 1_EDA.py       # Data exploration
│   │   ├── 2_Recommender.py # Recommendation engine
│   │   └── 3_Insights.py  # TF-IDF analysis
│   └── app.py             # Main app entry point
├── data/                  # Data directory
│   ├── raw/               # Raw restaurant reviews
│   └── processed/         # Cleaned and processed data
├── models/                # Trained ML models
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Core source code
│   ├── components/        # UI components
│   ├── config.py          # Configuration
│   ├── feature_engineering.py # Data processing
│   ├── preprocessing.py   # Text cleaning
│   ├── recommender.py     # Recommendation logic
│   └── utils.py           # Utilities
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## How It Works

The system uses a **hybrid recommendation approach**:

1. **TF-IDF Similarity**: Analyzes customer review text to find restaurants with similar content
2. **Rating Score**: Incorporates average customer ratings for quality assessment
3. **Popularity Boost**: Uses review count as a popularity indicator
4. **Weighted Combination**: Balances all factors for optimal recommendations

## Data

- **Source**: Restaurant reviews dataset with ratings and text feedback
- **Size**: 10,000 reviews across 100 restaurants
- **Features**: Restaurant names, customer reviews, ratings, timestamps

## Development

### Running Tests
```bash
pytest tests/
```

### Building Models
```bash
python -m src.pipeline_build
```

### Code Quality
- Follows PEP 8 style guidelines
- Type hints throughout
- Comprehensive docstrings
- Modular architecture

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


---

Author 

**Saurav Saini**

**Enjoy discovering your next favorite restaurant! **
