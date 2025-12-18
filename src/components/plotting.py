from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib style
plt.style.use('seaborn-v0_8')  # Use a modern style if available
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def plot_rating_distribution(df_clean: pd.DataFrame) -> plt.Figure:
    """
    Create a histogram of rating distribution.

    Args:
        df_clean: DataFrame containing review data with 'Rating' column

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots()

    # Plot histogram
    n, bins, patches = ax.hist(
        df_clean["Rating"].dropna(),
        bins=20,
        edgecolor='black',
        alpha=0.7,
        color='#2E86AB'
    )

    # Styling
    ax.set_title("Distribution of Customer Ratings", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.set_xlim(1, 5)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_rating = df_clean["Rating"].mean()
    ax.axvline(mean_rating, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rating:.2f}')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_top_restaurants_by_reviews(profiles_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Create a bar chart of top restaurants by review count.

    Args:
        profiles_df: DataFrame containing restaurant profiles
        top_n: Number of top restaurants to show

    Returns:
        matplotlib Figure object
    """
    # Get top restaurants
    top = profiles_df.sort_values("num_reviews", ascending=False).head(top_n)

    fig, ax = plt.subplots()

    # Create horizontal bar chart for better readability
    bars = ax.barh(
        top["Restaurant"].astype(str),
        top["num_reviews"],
        color='#A23B72',
        alpha=0.8,
        edgecolor='black'
    )

    # Styling
    ax.set_title(f"Top {top_n} Restaurants by Review Count", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_ylabel("Restaurant", fontsize=12)

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(top["num_reviews"]) * 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{int(width):,}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # Reverse y-axis so highest is on top
    ax.invert_yaxis()

    plt.tight_layout()
    return fig