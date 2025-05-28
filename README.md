# 🎬 Netflix Data Analytics Dashboard 📊

App link - https://27yhq9qsspw3cmaawj32at.streamlit.app/

## ✨ Overview

Welcome to the **Netflix Data Analytics Dashboard**! This Streamlit application is an advanced analytical suite designed as a comprehensive toolkit for exploring and deriving insights from a Netflix dataset. It's packed with **55 distinct tools** ranging from basic data overviews and core analytics to advanced AI-powered insights and gamified challenges.

This project serves as an excellent showcase for data analysis, visualization, and interactive dashboard development skills, perfect for a data analyst capstone project or for anyone interested in exploring the Netflix content landscape.

---

## 🚀 Features

This dashboard is brimming with features, categorized for ease of exploration:

### 📄 **Data Overview**
*   **Comprehensive Data Snapshot:** Get an instant understanding of your dataset's structure, including dimensions, missing values, data types, and statistical summaries.

### 📊 **Core Analytics (Tools 1-10)**
*   **Content Performance:** Analyze IMDb scores vs. viewership.
*   **Genre Trends:** Track genre popularity over time.
*   **Geographic Distribution:** Identify top content-producing countries.
*   **Duration Analysis:** Explore movie lengths and TV show seasons.
*   **Rating Distribution:** Understand audience ratings (PG, R, TV-MA).
*   **Release Timelines:** Visualize content releases by year and decade.
*   **ROI Analysis:** Examine budget vs. performance.
*   **Correlation Matrix:** Discover relationships between numeric features.
*   **Content Gap Analysis:** Find underrepresented genres in key markets.
*   **Predictive Analytics:** Simple models to predict viewership (illustrative).

### 🔬 **Advanced Analytics (Tools 11-30)**
*   **Recommendation Engine:** Simple genre-based suggestions.
*   **Data Export & Reporting:** Download data or summary reports.
*   **Director/Actor Performance:** Analyze talent based on titles and scores.
*   **Word Clouds:** Visualize frequent terms in titles.
*   **Content Evolution:** Track Movie vs. TV Show releases.
*   **Deep Dives:** Detailed stats for specific genres, content freshness, world maps, Movie vs. TV show comparisons, seasonality, keyword search, and multi-faceted correlations.

### 🧠 **AI-Powered Tools (Tools 31-35)**
*(Requires a Gemini API Key)*
*   **General Insights:** AI-generated strategic observations.
*   **AI Chat with Dataset:** Ask natural language questions about your data.
*   **Content Summaries:** AI-generated Netflix-style synopses.
*   **Title Suggestions:** Creative title ideas from AI.
*   **Sentiment Analysis:** Gauge sentiment from simulated review text.

### 🔍 **Deeper Analytical Perspectives (Tools 36-52)**
*   **Content Lifecycle:** Analyze acquisition lag and freshness.
*   **'Hidden Gems' Detector:** Find high-quality, low-visibility content.
*   **Genre Saturation Matrix:** Plot popularity vs. title count.
*   **N-gram Analysis:** Identify common phrases in titles.
*   **User Persona Recommendations:** Simulated content suggestions for different user types.
*   **Award Impact:** Analyze the effect of awards on performance.
*   **Language Diversity:** Explore content distribution and performance by language.
*   **Talent Genre Affinity:** See which genres directors/actors frequent.
*   **Technical Details vs. Performance:** Analyze aspect ratio, sound mix, etc.
*   **Franchise/Sequel Simulation:** Identify and analyze potential series.
*   **Budget Efficiency & Emerging Talent Spotlights.**
*   **Genre Synergy & Cross-Promotion.**
*   **Clustering for Content Discovery:** Uncover hidden content groups using K-Means.

### 🛠️ **Utilities (Tool 53)**
*   **Data Cleaning & Preparation Showcase:** Live demonstrations of handling missing values, feature engineering, and data type conversion.

### 🎮 **Gamified Analytics (Tools 54-55)**
*   **'Guess the Views' Challenge:** Test your intuition on content viewership.
*   **Netflix Trivia Challenge:** Quiz yourself on dataset facts with streak bonuses!

---

## ⚙️ How to Run

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-link>
    cd <repository-folder>
    ```

2.  **Install Dependencies:**
    Make sure you have Python installed. Then, install the required libraries:
    ```bash
    pip install streamlit pandas plotly google-generativeai seaborn matplotlib wordcloud scikit-learn
    ```
    *(It's highly recommended to use a virtual environment)*

3.  **Prepare Your Data:**
    *   Ensure you have a CSV file named `netflix_analysis.csv` in the same directory as `app.py`.
    *   The application is pre-configured to load this file. You can modify the loading logic in `app.py` if your file has a different name or path.
    *   The dataset should ideally contain columns like `title`, `type`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` (genres), `description`, `imdb_score`, `budget_millions`, `views_millions`, etc. Not all columns are mandatory for all tools, but more complete data will enable more features.

4.  **Set Up Gemini API Key (Optional but Recommended for AI Tools):**
    *   Obtain a Gemini API key from Google AI Studio.
    *   When you run the app, there will be a field in the sidebar to enter your API key. This key is used for Tools 31-35 and Tool 51.

5.  **Run the Streamlit App:**
    Open your terminal or command prompt, navigate to the directory containing `app.py`, and run:
    ```bash
    streamlit run app.py
    ```
    Your default web browser should open with the dashboard.

---

## 📦 Dependencies

*   `streamlit`
*   `pandas`
*   `plotly` / `plotly.express` / `plotly.graph_objects`
*   `google-generativeai`
*   `numpy`
*   `seaborn`
*   `matplotlib`
*   `wordcloud`
*   `scikit-learn` (for KMeans, StandardScaler, PCA, CountVectorizer, ML models in Tool 10)
*   `openpyxl` (implicitly required by pandas for Excel export, though not directly imported in `app.py`)

---

## 🔮 Potential Future Enhancements

*   **User Upload for Custom Datasets:** Allow users to upload their own Netflix-like CSV files.
*   **More Sophisticated ML Models:** Implement advanced recommendation algorithms (e.g., collaborative filtering, content-based with TF-IDF) or more robust predictive models.
*   **Interactive Data Cleaning Options:** Allow users to choose and apply cleaning steps directly.
*   **User Accounts & Persistent Favorites:** Save user preferences and favorites across sessions.
*   **Deeper NLP Analysis:** Sentiment analysis on descriptions, topic modeling.
*   **Time Series Forecasting:** For content addition or genre popularity.
*   **Enhanced UI/UX:** Further theme customizations and interactive elements.

---

## 🌟 Acknowledgements

*   This project is designed as a capstone and learning tool.
*   Inspiration from various data analytics dashboards and Netflix's own interface.
*   Libraries used: Streamlit, Pandas, Plotly, Scikit-learn, Google Generative AI, and others.

---

Enjoy exploring the world of Netflix data! 🍿✨
