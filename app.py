import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter

from itertools import combinations # Added for Tool 50
from sklearn.cluster import KMeans # Added for Tool 52
from sklearn.preprocessing import StandardScaler # Added for Tool 52
from sklearn.decomposition import PCA # Added for Tool 52
import random # Added for Tools 56, 57

st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide", page_icon="🎬")
st.title("🎬 Netflix Data Analytics Dashboard")


with st.sidebar:
    st.image("1.jpg", caption="", use_container_width=True)



st.image("2.jpg",use_container_width=True)

# Custom CSS for Dark Theme and Flashcards
st.markdown("""
<style>
body {
    color: #E0E0E0; /* Light grey text */
    background-color: #121212; /* Very dark grey background */
}

/* Main app container */
.stApp {
    background-color: #121212; /* Ensure app background matches body */
}

/* Metric Flashcards */
.metric-card {
    background: linear-gradient(145deg, #2a2d30, #222427); /* Darker, subtle gradient for cards */
    border-radius: 15px;
    padding: 25px 20px; /* Increased padding */
    margin-bottom: 20px; /* Increased margin */
    box-shadow: 7px 7px 15px #1b1c1e, -7px -7px 15px #2d2e32; /* Neumorphic style shadow */
    text-align: center;
    border: 1px solid #383838; /* Slightly more visible border */
}

.metric-card h4 { /* Title of the metric, e.g., "Total Titles" */
    font-size: 1.2em; /* Slightly larger title */
    color: #A0A0A0; /* Lighter grey for title */
    margin-bottom: 10px; /* More space below title */
    font-weight: 500;
    text-transform: uppercase; /* Uppercase for a more 'card' feel */
    letter-spacing: 0.5px;
}

.metric-card p { /* The actual metric value */
    font-size: 2.5em; /* Larger metric value */
    color: #4A90E2; /* A refined, less bright blue */
    font-weight: 700;
    margin: 0;
}

/* Expander Styling */
div[data-testid="stExpander"] {
    background-color: #1E1E1E; /* Dark background for expander content area */
    border: 1px solid #333;
    border-radius: 10px;
    margin-bottom: 15px;
}

div[data-testid="stExpander"] > div:first-child { /* Expander Header */
    background-color: #282A2D; /* Slightly lighter dark grey for header */
    border-radius: 9px 9px 0 0; /* Match outer radius */
    border-bottom: 1px solid #333;
    padding: 12px 18px !important; /* Adjusted padding */
}

div[data-testid="stExpander"] > div:first-child summary {
    color: #C0C0C0; /* Lighter text for expander title */
    font-weight: 600;
    font-size: 1.15em;
}
div[data-testid="stExpander"] > div:first-child summary:hover {
    color: #E0E0E0;
}

/* Sidebar styling */
div[data-testid="stSidebarUserContent"] {
    background-color: #1A1A1A; /* Slightly different dark for sidebar */
    padding: 15px;
}

.stButton > button {
    border: 1px solid #4A90E2;
    background-color: transparent;
    color: #4A90E2;
    padding: 8px 18px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #4A90E2;
    color: #121212; /* Dark text on hover */
}

/* Input widgets styling */
div[data-testid="stSelectbox"] label, 
div[data-testid="stTextInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stTextArea"] label {
    color: #B0B0B0; /* Lighter label text */
    font-weight: 500;
}

/* Markdown headers */
h1, h2, h3, h4, h5, h6 {
    color: #D0D0D0;
}

/* Plotly chart background - handled by template, but good to be aware */
.js-plotly-plot .plotly {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("**Advanced Analytics Suite for Data Analyst Capstone Project**")


# Sidebar
st.sidebar.header("📂 Data Source")
st.sidebar.info("Using pre-loaded dataset: `netflix_analysis.csv`")

# Attempt to load the pre-defined CSV file
try:
    df = pd.read_csv("netflix_analysis.csv")
    st.sidebar.success("`netflix_analysis.csv` loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: `netflix_analysis.csv` not found in the application directory.")
    st.error("CRITICAL ERROR: `netflix_analysis.csv` could not be loaded. Please ensure the file is in the same directory as the `app.py` script.")
    # Create an empty DataFrame to prevent downstream errors, or st.stop()
    df = pd.DataFrame() 
    # Optionally, you could use st.stop() here to halt execution if the file is critical
    # st.stop() 
except Exception as e:
    st.sidebar.error(f"An error occurred while loading `netflix_analysis.csv`: {e}")
    st.error(f"An error occurred: {e}")
    df = pd.DataFrame()
    # st.stop()

# Gemini API
gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Sidebar: User Utilities & Guide (Moved from main body)
with st.sidebar.expander("❤️ My Favorite Titles"):
    st.subheader("Your Saved Favorite Content")
    st.markdown("You can add titles to your favorites from various tools. Favorites are saved for your current session.")

    if 'favorite_titles' not in st.session_state:
        st.session_state.favorite_titles = []

    # Initialize/manage the list of sample titles for the selectbox
    if 'fav_sample_options_list' not in st.session_state:
        st.session_state.fav_sample_options_list = []

    if not df.empty and 'title' in df.columns:
        # Function to generate/refresh sample options
        def refresh_sample_options_for_favorites_sidebar(): # Renamed function for clarity
            unique_titles = df['title'].dropna().unique()
            if len(unique_titles) > 0:
                sample_size = min(5, len(unique_titles))
                st.session_state.fav_sample_options_list = random.sample(list(unique_titles), sample_size)
            else:
                st.session_state.fav_sample_options_list = []

        # Populate on first load if empty and data is available
        if not st.session_state.fav_sample_options_list and len(df['title'].dropna().unique()) > 0 :
            refresh_sample_options_for_favorites_sidebar()

        # Button to explicitly refresh the sample titles
        if st.button("Refresh Sample Titles for Selection", key="refresh_fav_samples_button_sidebar"):
            refresh_sample_options_for_favorites_sidebar()
            st.rerun() 

        if st.session_state.fav_sample_options_list:
            selected_title_for_adding_sidebar = st.selectbox(
                "Select a sample title to add (demo):",
                options=st.session_state.fav_sample_options_list,
                key="fav_title_selector_sidebar" 
            )

            if st.button("Add to Favorites", key="fav_add_button_sidebar"):
                if selected_title_for_adding_sidebar:
                    if selected_title_for_adding_sidebar not in st.session_state.favorite_titles:
                        st.session_state.favorite_titles.append(selected_title_for_adding_sidebar)
                        st.success(f"Added '{selected_title_for_adding_sidebar}'!")
                    else:
                        st.info(f"'{selected_title_for_adding_sidebar}' is already in favorites.")
                else:
                    st.warning("No title selected.")
        else:
            st.write("No sample titles available.")

    if not st.session_state.favorite_titles:
        st.info("No favorites added yet.")
    else:
        st.write("Your favorites:")
        for i, title_in_fav_list in enumerate(list(st.session_state.favorite_titles)): 
            col1_fav, col2_fav = st.columns([0.8, 0.2])
            col1_fav.write(f"- {title_in_fav_list}")
            if col2_fav.button(f"X##{i}", key=f"remove_fav_{i}_sidebar", help="Remove"): # Smaller remove button
                del st.session_state.favorite_titles[i] 
                st.rerun()

        if st.button("Clear All Favorites", key="clear_fav_sidebar"):
            st.session_state.favorite_titles = []
            st.rerun()

with st.sidebar.expander("📚 In-App Guide"):
    st.subheader("Understanding Your Dashboard")
    st.markdown("""
    This guide provides a brief overview of key analytical tool categories. 
    Explore each tool's expander in the main dashboard for specific functionalities, visualizations, and interactive elements.
    """)
    st.markdown("---")
    st.markdown("#### **📄 Data Overview**")
    st.markdown("- **Comprehensive Data Overview:** Provides an initial snapshot of your dataset. See the first few rows, check data dimensions (rows and columns), identify missing values per column, review column data types, and get a basic statistical summary for all columns.")
    st.markdown("- **Key Columns Used:** All columns in the dataset.")
    st.markdown("- **How to Use:** Simply expand the section to view the data summary tables.")
    st.markdown("- **Example Insight:** Quickly understand the size, structure, and initial quality of your dataset.")
    st.markdown("---")
    st.markdown("#### **📊 Core Analytics (Tools 1-10)**")
    st.markdown("- **Tool 1: Content Performance Analytics:**")
    st.markdown("  - **Purpose:** Analyze the relationship between content quality (IMDb Score) and audience reach (Views).")
    st.markdown("  - **Key Columns Used:** `imdb_score`, `views_millions`, `type`, `budget_millions` (for bubble size).")
    st.markdown("  - **How to Use:** View the scatter plot to see how different content types perform. Top viewed titles are listed below.")
    st.markdown("  - **Example Insight:** Identify if higher IMDb scores consistently lead to more views, and whether this pattern differs for movies vs. TV shows.")
    st.markdown("- **Tool 2: Genre Trend Analysis:**")
    st.markdown("  - **Purpose:** Visualize how the popularity of different genres has changed over the years.")
    st.markdown("  - **Key Columns Used:** `release_year`, `listed_in`.")
    st.markdown("  - **How to Use:** Use sliders to select the number of top genres and apply smoothing to the trend lines. View the line chart and the overall top genres list.")
    st.markdown("  - **Example Insight:** Discover which genres are currently trending up or down, or which have dominated in the past.")
    st.markdown("- **Tool 3: Geographic Content Distribution:**")
    st.markdown("  - **Purpose:** Understand which countries are the primary sources of content in the dataset.")
    st.markdown("  - **Key Columns Used:** `country`.")
    st.markdown("  - **How to Use:** View the horizontal bar chart showing the top N countries by title count. See their percentage share of total content.")
    st.markdown("  - **Example Insight:** Identify the major content-producing regions and their relative contribution.")
    st.markdown("- **Tool 4: Content Duration Analysis:**")
    st.markdown("  - **Purpose:** Analyze the typical length of movies (in minutes) and TV shows (in seasons).")
    st.markdown("  - **Key Columns Used:** `duration`, `type`.")
    st.markdown("  - **How to Use:** View the separate histograms for movie durations and TV show seasons.")
    st.markdown("  - **Example Insight:** Understand the most common movie lengths or how many seasons TV shows typically run in this dataset.")
    st.markdown("- **Tool 5: Rating Distribution Analysis:**")
    st.markdown("  - **Purpose:** See the distribution of content ratings (e.g., PG, R, TV-MA) and the proportion of mature vs. family-friendly content.")
    st.markdown("  - **Key Columns Used:** `rating`.")
    st.markdown("  - **How to Use:** View the pie chart showing the percentage breakdown of each rating. See calculated percentages for mature and family content.")
    st.markdown("  - **Example Insight:** Understand the target audience mix of the content in the dataset.")
    st.markdown("- **Tool 6: Release Year Timeline:**")
    st.markdown("  - **Purpose:** Visualize the volume of content released each year and summarize releases by decade.")
    st.markdown("  - **Key Columns Used:** `release_year`.")
    st.markdown("  - **How to Use:** View the area chart showing title counts per year. See a dictionary summary of titles per decade.")
    st.markdown("  - **Example Insight:** Observe periods of high or low content production and identify historical trends.")
    st.markdown("- **Tool 7: Budget vs Performance ROI:**")
    st.markdown("  - **Purpose:** Analyze the relationship between content budget and its performance (Return on Investment, calculated as Views/Budget).")
    st.markdown("  - **Key Columns Used:** `budget_millions`, `views_millions`, `type`, `imdb_score` (for bubble size).")
    st.markdown("  - **How to Use:** View the scatter plot. Use checkboxes for log scales and a slider for marker opacity. See a table of top ROI titles.")
    st.markdown("  - **Example Insight:** Identify content that achieved high viewership relative to its cost, potentially indicating efficient spending or organic popularity.")
    st.markdown("- **Tool 8: Content Correlation Matrix:**")
    st.markdown("  - **Purpose:** Quickly identify linear relationships between all numeric features in the dataset.")
    st.markdown("  - **Key Columns Used:** All numeric columns.")
    st.markdown("  - **How to Use:** View the heatmap. Use the slider to highlight correlations above a certain threshold.")
    st.markdown("  - **Example Insight:** See if features like IMDb score, budget, or views tend to increase or decrease together.")
    st.markdown("- **Tool 9: Content Gap Analysis:**")
    st.markdown("  - **Purpose:** Identify genres that are popular in certain countries but potentially underrepresented in others within the dataset.")
    st.markdown("  - **Key Columns Used:** `country`, `listed_in`.")
    st.markdown("  - **How to Use:** The tool automatically lists potential genre gaps for the top countries based on a pivot table analysis.")
    st.markdown("  - **Example Insight:** Discover potential opportunities for content acquisition or production in specific countries based on genre popularity elsewhere.")
    st.markdown("- **Tool 10: Predictive Analytics Dashboard:**")
    st.markdown("  - **Purpose:** Demonstrate simple machine learning models to predict 'Views (Millions)' based on 'IMDb Score' and 'Budget'.")
    st.markdown("  - **Key Columns Used:** `imdb_score`, `budget_millions`, `views_millions`.")
    st.markdown("  - **How to Use:** Select a model type (Linear, Polynomial, SVR). View the model's performance metrics (R-squared, MAE, RMSE) on a test set and a scatter plot of actual vs. predicted views. Enter custom values to get a prediction. *Disclaimer: These are simplified models for illustration.*")
    st.markdown("  - **Example Insight:** Understand how well simple models can predict viewership based on these features and see the impact of different model types.")
    st.markdown("---")
    st.markdown("#### **🔬 Advanced Analytics (Tools 11-30)**")
    st.markdown("- **Tool 11: Content Recommendation Engine:**")
    st.markdown("  - **Purpose:** Provide simple content recommendations based on a selected genre.")
    st.markdown("  - **Key Columns Used:** `listed_in`, `imdb_score`, `title`, `country`, `release_year`.")
    st.markdown("  - **How to Use:** Select a genre from the dropdown. The tool lists top-rated titles within that genre.")
    st.markdown("  - **Example Insight:** Find highly-rated content within specific interest areas.")
    st.markdown("- **Tool 12: Data Export & Reporting:**")
    st.markdown("  - **Purpose:** Allows you to download the dataset or a summary report in different formats.")
    st.markdown("  - **Key Columns Used:** All columns (for full data), various key columns for the summary report.")
    st.markdown("  - **How to Use:** Select the desired format (CSV, JSON, Excel Summary) and click 'Generate Export' to get a download button.")
    st.markdown("  - **Example Insight:** Easily obtain the data or a quick summary for external use or further analysis.")
    st.markdown("- **Tool 13: Director Performance Analysis:**")
    st.markdown("  - **Purpose:** Analyze directors based on the number of titles they have and the average IMDb score of their work.")
    st.markdown("  - **Key Columns Used:** `director`, `title`, `imdb_score`.")
    st.markdown("  - **How to Use:** View the bar charts showing top directors by title count and average IMDb score (for directors with multiple titles).")
    st.markdown("  - **Example Insight:** Identify prolific directors or those whose work is consistently highly rated.")
    st.markdown("- **Tool 14: Title Word Cloud:**")
    st.markdown("  - **Purpose:** Visualize the most frequent words used in content titles.")
    st.markdown("  - **Key Columns Used:** `title`.")
    st.markdown("  - **How to Use:** Expand the section to see the generated word cloud.")
    st.markdown("  - **Example Insight:** Discover common themes or keywords present across the content titles.")
    st.markdown("- **Tool 15: Content Type Evolution Over Time:**")
    st.markdown("  - **Purpose:** Track the trend of Movie vs. TV Show releases over the years.")
    st.markdown("  - **Key Columns Used:** `release_year`, `type`.")
    st.markdown("  - **How to Use:** View the line chart showing the count of Movies and TV Shows released each year.")
    st.markdown("  - **Example Insight:** See if the platform has shifted its focus between movies and TV shows over time.")
    st.markdown("- **Tool 16: Actor/Cast Performance Analysis:**")
    st.markdown("  - **Purpose:** Analyze actors based on the number of titles they've appeared in and the average IMDb score of those titles.")
    st.markdown("  - **Key Columns Used:** `cast`, `title`, `imdb_score`.")
    st.markdown("  - **How to Use:** View the bar charts showing top actors by title count and average IMDb score.")
    st.markdown("  - **Example Insight:** Identify the most frequently appearing actors or those associated with highly-rated content.")
    st.markdown("- **Tool 17: Genre Deep Dive:**")
    st.markdown("  - **Purpose:** Get detailed statistics and top titles for a specific genre.")
    st.markdown("  - **Key Columns Used:** `listed_in`, `release_year`, `imdb_score`, `title`, `type`.")
    st.markdown("  - **How to Use:** Select a genre from the dropdown. View metrics like total titles, average IMDb score, release trend, and a list of top titles within that genre.")
    st.markdown("  - **Example Insight:** Gain a focused understanding of the characteristics and performance of content within a particular genre.")
    st.markdown("- **Tool 18: Content Freshness Analysis:**")
    st.markdown("  - **Purpose:** Analyze the age of content (years since release) and its relationship with performance metrics.")
    st.markdown("  - **Key Columns Used:** `release_year`, `imdb_score`, `views_millions`.")
    st.markdown("  - **How to Use:** View histograms of content age and scatter plots showing age vs. IMDb score and age vs. views, including trendlines.")
    st.markdown("  - **Example Insight:** See if newer content tends to have higher ratings or views, or if older 'classic' content performs differently.")
    st.markdown("- **Tool 19: Interactive World Map of Content Production:**")
    st.markdown("  - **Purpose:** Visualize the geographic distribution of content production on a world map.")
    st.markdown("  - **Key Columns Used:** `country`.")
    st.markdown("  - **How to Use:** View the choropleth map where countries are colored based on the number of titles they produced (using the first listed country). Hover over countries for details.")
    st.markdown("  - **Example Insight:** See which countries are major content hubs and explore production volumes globally.")
    st.markdown("- **Tool 20: Movie vs. TV Show Deep Comparison:**")
    st.markdown("  - **Purpose:** Compare key metrics and rating distributions specifically between Movies and TV Shows.")
    st.markdown("  - **Key Columns Used:** `type`, `imdb_score`, `duration`, `rating`.")
    st.markdown("  - **How to Use:** View side-by-side metrics (counts, average scores, average length/seasons) and rating distribution pie charts for each content type.")
    st.markdown("  - **Example Insight:** Understand fundamental differences in characteristics and audience ratings between movies and TV shows in the dataset.")
    st.markdown("- **Tool 21: Release Month/Seasonality Analysis:**")
    st.markdown("  - **Purpose:** Analyze if there are seasonal patterns in when content is added to the platform.")
    st.markdown("  - **Key Columns Used:** `date_added`.")
    st.markdown("  - **How to Use:** View the bar chart showing the number of titles added for each month of the year.")
    st.markdown("  - **Example Insight:** Identify if certain months are more active for content releases/additions.")
    st.markdown("- **Tool 22: Keyword Search in Titles:**")
    st.markdown("  - **Purpose:** Find titles that contain a specific word or phrase.")
    st.markdown("  - **Key Columns Used:** `title`, `type`, `release_year`, `imdb_score`.")
    st.markdown("  - **How to Use:** Enter a keyword in the text input box. The tool lists matching titles and some key details.")
    st.markdown("  - **Example Insight:** Quickly find content related to a specific theme or topic.")
    st.markdown("- **Tool 23: Content Rating vs. IMDb Score Analysis:**")
    st.markdown("  - **Purpose:** Analyze how content ratings (e.g., PG, R) relate to average IMDb scores.")
    st.markdown("  - **Key Columns Used:** `rating`, `imdb_score`.")
    st.markdown("  - **How to Use:** View the bar chart showing the average IMDb score for each rating category and a box plot showing the distribution of scores within each rating.")
    st.markdown("  - **Example Insight:** See if certain rating categories tend to have higher or lower average critical reception.")
    st.markdown("- **Tool 24: Top Director-Actor Collaborations:**")
    st.markdown("  - **Purpose:** Identify the most frequent pairings of directors and actors in the dataset.")
    st.markdown("  - **Key Columns Used:** `director`, `cast`, `title`.")
    st.markdown("  - **How to Use:** View the horizontal bar chart showing the top director-actor pairs by the number of titles they've worked on together.")
    st.markdown("  - **Example Insight:** Discover notable creative partnerships within the content.")
    st.markdown("- **Tool 25: IMDb Score Trends Over Release Years:**")
    st.markdown("  - **Purpose:** Track the average IMDb score of content released each year.")
    st.markdown("  - **Key Columns Used:** `release_year`, `imdb_score`.")
    st.markdown("  - **How to Use:** View the line chart showing the yearly average IMDb score, including a rolling average trendline.")
    st.markdown("  - **Example Insight:** See if the average critical quality of content has changed over time based on release year.")
    st.markdown("- **Tool 26: Multi-Country Content Profile Comparison:**")
    st.markdown("  - **Purpose:** Compare the content profiles (type distribution, average IMDb score) of selected countries.")
    st.markdown("  - **Key Columns Used:** `country`, `type`, `listed_in`, `imdb_score`.")
    st.markdown("  - **How to Use:** Select up to 3 countries from the multiselect dropdown. View bar charts comparing their Movie/TV show mix and average IMDb scores.")
    st.markdown("  - **Example Insight:** Understand how the content focus and critical reception differ between key production countries.")
    st.markdown("- **Tool 27: Content Length vs. IMDb Score:**")
    st.markdown("  - **Purpose:** Analyze the relationship between content length (movie duration or TV show seasons) and IMDb score.")
    st.markdown("  - **Key Columns Used:** `duration`, `imdb_score`, `type`.")
    st.markdown("  - **How to Use:** View separate scatter plots for Movies (duration vs. score) and TV Shows (seasons vs. score), including trendlines.")
    st.markdown("  - **Example Insight:** See if there's a correlation between how long content is and how it's rated.")
    st.markdown("- **Tool 28: Genre Co-occurrence Analysis:**")
    st.markdown("  - **Purpose:** Identify which genres are most frequently listed together for the same titles.")
    st.markdown("  - **Key Columns Used:** `listed_in`.")
    st.markdown("  - **How to Use:** View the horizontal bar chart showing the top genre pairs by their frequency of co-occurrence.")
    st.markdown("  - **Example Insight:** Understand common genre combinations and potential audience overlaps.")
    st.markdown("- **Tool 29: Cast Size vs. Performance:**")
    st.markdown("  - **Purpose:** Explore if the number of cast members listed for a title relates to its performance (IMDb score or views).")
    st.markdown("  - **Key Columns Used:** `cast`, `imdb_score`, `views_millions`.")
    st.markdown("  - **How to Use:** View scatter plots showing cast size vs. IMDb score and cast size vs. views, including trendlines.")
    st.markdown("  - **Example Insight:** See if titles with larger casts tend to perform better or worse.")
    st.markdown("- **Tool 30: Content Addition Trend (Yearly):**")
    st.markdown("  - **Purpose:** Visualize the volume of content added to the Netflix platform each year, broken down by content type.")
    st.markdown("  - **Key Columns Used:** `date_added`, `type`.")
    st.markdown("  - **How to Use:** View the stacked bar chart showing yearly additions, segmented by Movies and TV Shows.")
    st.markdown("  - **Example Insight:** Understand the platform's content acquisition strategy over time.")
    st.markdown("---")
    st.markdown("#### **🧠 AI-Powered Tools (Tools 31-35)**")
    st.markdown("*(Requires a Gemini API Key to be entered in the sidebar)*")
    st.markdown("- **Tool 31: AI-Powered General Insights:**")
    st.markdown("  - **Purpose:** Get high-level strategic insights (e.g., content strategy, market gaps) generated by AI based on a summary of your dataset.")
    st.markdown("  - **Key Columns Used:** Uses a summary derived from various key columns.")
    st.markdown("  - **How to Use:** Select an analysis type (e.g., Content Strategy, Market Gaps) and click 'Generate AI Insights'. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Obtain AI-driven perspectives on potential strategic directions or areas for growth.")
    st.markdown("- **Tool 32: AI Chat with Dataset:**")
    st.markdown("  - **Purpose:** Ask natural language questions about your dataset and get answers generated by AI.")
    st.markdown("  - **Key Columns Used:** Uses a summary derived from various key columns.")
    st.markdown("  - **How to Use:** Enter your question in the text area and click 'Ask AI'. The AI responds based on the provided dataset summary. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Quickly get answers to specific questions about the data without needing to write code or use other tools.")
    st.markdown("- **Tool 33: AI-Generated Content Summaries:**")
    st.markdown("  - **Purpose:** Generate short, engaging summaries for specific titles in your dataset, mimicking a Netflix preview style.")
    st.markdown("  - **Key Columns Used:** `title`, `type`, `listed_in`.")
    st.markdown("  - **How to Use:** Select a title from the dropdown and click 'Generate Summary'. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Create compelling descriptions for content for marketing or presentation purposes.")
    st.markdown("- **Tool 34: AI-Generated Title Suggestions:**")
    st.markdown("  - **Purpose:** Get creative title ideas for new content based on themes or keywords you provide.")
    st.markdown("  - **Key Columns Used:** None directly from the dataset, uses user input.")
    st.markdown("  - **How to Use:** Enter a theme or keywords and select a content type. Click 'Suggest Titles'. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Brainstorm potential titles for new projects based on desired concepts.")
    st.markdown("- **Tool 35: AI-Driven Sentiment Analysis (Simulated Review):**")
    st.markdown("  - **Purpose:** Analyze the sentiment (Positive, Negative, Neutral) of a piece of text, simulating review analysis.")
    st.markdown("  - **Key Columns Used:** None directly from the dataset, uses user input.")
    st.markdown("  - **How to Use:** Paste a review text into the text area and click 'Analyze Sentiment'. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Understand how AI can be used to gauge audience sentiment from textual feedback.")
    st.markdown("---")
    st.markdown("#### **🔍 Deeper Perspectives (Tools 36-52)**")
    st.markdown("- **Tool 36: Content Lifecycle, Acquisition & Freshness Analysis:**")
    st.markdown("  - **Purpose:** Analyze the age of content when it was added to Netflix (acquisition lag), how long it stays on the platform, and how freshness relates to performance and content type/genre.")
    st.markdown("  - **Key Columns Used:** `release_year`, `date_added`, `imdb_score`, `type`, `listed_in`.")
    st.markdown("  - **How to Use:** View histograms and scatter plots showing age at addition, IMDb score vs. age at addition, performance vs. years on platform, and freshness comparisons by type and genre.")
    st.markdown("  - **Example Insight:** Understand Netflix's content acquisition strategy regarding content age and how the age of content might relate to its performance or how long it remains popular.")
    st.markdown("- **Tool 37: 'Hidden Gems' Detector:**")
    st.markdown("  - **Purpose:** Identify content that has high critical acclaim (IMDb score) but relatively low visibility (views or budget as a proxy).")
    st.markdown("  - **Key Columns Used:** `imdb_score`, `views_millions` or `budget_millions`, `title`, `type`, `release_year`.")
    st.markdown("  - **How to Use:** Use sliders to define the criteria for a 'hidden gem' (minimum IMDb score, maximum views/budget). View the scatter plot with gems highlighted and a table of identified titles.")
    st.markdown("  - **Example Insight:** Discover potentially underappreciated content that might be worth promoting.")
    st.markdown("- **Tool 38: Genre Popularity vs. Saturation Matrix:**")
    st.markdown("  - **Purpose:** Visualize genres based on their average IMDb score (popularity) and the number of titles (saturation).")
    st.markdown("  - **Key Columns Used:** `listed_in`, `imdb_score`, `budget_millions` (for bubble size).")
    st.markdown("  - **How to Use:** View the scatter plot where each point represents a genre. The X-axis is saturation (title count), Y-axis is popularity (average IMDb), and bubble size can represent average budget.")
    st.markdown("  - **Example Insight:** Identify genres that are highly popular but not saturated (potential growth areas) or highly saturated but less popular (potential areas for refinement).")
    st.markdown("- **Tool 39: N-gram Analysis on Titles:**")
    st.markdown("  - **Purpose:** Identify the most common sequences of words (bi-grams and tri-grams) appearing in content titles.")
    st.markdown("  - **Key Columns Used:** `title`.")
    st.markdown("  - **How to Use:** View the horizontal bar charts showing the top 10 most frequent 2-word and 3-word phrases in titles.")
    st.markdown("  - **Example Insight:** Understand common linguistic patterns or themes emphasized in content titles.")
    st.markdown("- **Tool 40: User Persona-Based Recommendations (Simulated):**")
    st.markdown("  - **Purpose:** Provide content recommendations tailored to predefined user profiles (personas) based on their simulated preferences.")
    st.markdown("  - **Key Columns Used:** `type`, `listed_in`, `imdb_score`, `rating`, `title`.")
    st.markdown("  - **How to Use:** Select a user persona from the dropdown. The tool filters and lists titles that match the persona's criteria (genre, rating, minimum IMDb score).")
    st.markdown("  - **Example Insight:** See how different audience segments might be served by the existing content library.")
    st.markdown("- **Tool 41: Award Impact Analysis:**")
    st.markdown("  - **Purpose:** Analyze if receiving awards or nominations (specifically 'Best Picture' nomination if available) correlates with higher IMDb scores or views.")
    st.markdown("  - **Key Columns Used:** `title`, `imdb_score`, `awards_won`, `nomination_for_best_picture`, `views_millions`.")
    st.markdown("  - **How to Use:** View bar charts comparing metrics for nominated vs. non-nominated content. View a scatter plot of awards won vs. IMDb score and a table of top award-winning titles.")
    st.markdown("  - **Example Insight:** Understand the potential impact of critical recognition on content performance.")
    st.markdown("- **Tool 42: Content Language Diversity & Performance:**")
    st.markdown("  - **Purpose:** Analyze the distribution of content by language and compare performance metrics (IMDb score, views) across different languages.")
    st.markdown("  - **Key Columns Used:** `language`, `imdb_score`, `views_millions`, `type`.")
    st.markdown("  - **How to Use:** Use a slider to select the number of top languages. View pie charts and bar charts showing content counts, average IMDb scores, average views, and content type distribution for these languages.")
    st.markdown("  - **Example Insight:** See which languages are most represented and how content in different languages performs critically and in terms of viewership.")
    st.markdown("- **Tool 43: Director & Actor Genre Affinity:**")
    st.markdown("  - **Purpose:** Visualize the genres most associated with a specific director or actor.")
    st.markdown("  - **Key Columns Used:** `director`, `cast`, `listed_in`.")
    st.markdown("  - **How to Use:** Select a director or actor from the dropdown. View a bar chart showing the count of titles they've been involved with in each genre.")
    st.markdown("  - **Example Insight:** Understand the typical genre focus of key creative talents.")
    st.markdown("- **Tool 44: Content Attributes & Technical Details vs. Performance:**")
    st.markdown("  - **Purpose:** Analyze if characteristics like title length, presence of numbers in titles, or technical aspects (Aspect Ratio, Sound Mix - if available) correlate with IMDb score.")
    st.markdown("  - **Key Columns Used:** `title`, `imdb_score`, `aspect_ratio`, `sound_mix`.")
    st.markdown("  - **How to Use:** View scatter plots and bar charts comparing IMDb scores based on title word count, presence of numbers, aspect ratio, and sound mix.")
    st.markdown("  - **Example Insight:** See if seemingly minor content attributes or technical choices have any observable relationship with critical reception.")
    st.markdown("- **Tool 45: Simulated Franchise/Sequel Analysis:**")
    st.markdown("  - **Purpose:** Use simple title pattern matching to identify potential sequels or franchise entries and compare their average IMDb score to likely standalone titles.")
    st.markdown("  - **Key Columns Used:** `title`, `imdb_score`.")
    st.markdown("  - **How to Use:** View the bar chart comparing the average IMDb score of titles identified as potential franchises vs. standalone titles. See examples of identified titles. *Note: This is a heuristic simulation.*")
    st.markdown("  - **Example Insight:** Get a preliminary idea if franchise entries in this dataset tend to perform better or worse critically than standalone content.")
    st.markdown("- **Tool 46: Genre Performance - Movies vs. TV Shows:**")
    st.markdown("  - **Purpose:** Compare the average IMDb score and title count for Movies versus TV Shows within a specific genre.")
    st.markdown("  - **Key Columns Used:** `listed_in`, `type`, `imdb_score`, `title`.")
    st.markdown("  - **How to Use:** Select a genre from the dropdown. View side-by-side bar charts comparing the average IMDb score and the number of titles for Movies and TV Shows in that genre.")
    st.markdown("  - **Example Insight:** Understand if a particular genre is more successful or prevalent as a movie or a TV show format.")
    st.markdown("- **Tool 47: Decade-wise Genre Evolution & Dominance:**")
    st.markdown("  - **Purpose:** Visualize how the popularity (number of titles) of top genres has evolved across different decades.")
    st.markdown("  - **Key Columns Used:** `release_year`, `listed_in`.")
    st.markdown("  - **How to Use:** View the stacked area chart showing the count of titles for top genres within each decade.")
    st.markdown("  - **Example Insight:** See which genres dominated in different historical periods and how their presence has changed.")
    st.markdown("- **Tool 48: Budget Efficiency Tiers & ROI Analysis:**")
    st.markdown("  - **Purpose:** Categorize content into budget tiers (Low, Medium, High) and analyze the distribution of ROI and average IMDb score within each tier.")
    st.markdown("  - **Key Columns Used:** `budget_millions`, `views_millions`, `imdb_score`.")
    st.markdown("  - **How to Use:** View box plots showing the distribution of ROI for each budget tier and a bar chart showing the average IMDb score per tier.")
    st.markdown("  - **Example Insight:** Understand if higher budgets correlate with better critical reception or if lower-budget content can achieve high ROI.")
    st.markdown("- **Tool 49: Emerging Talent Spotlight (Directors/Actors):**")
    st.markdown("  - **Purpose:** Identify directors and actors who have a relatively small number of titles but high average IMDb scores, potentially indicating emerging talent.")
    st.markdown("  - **Key Columns Used:** `director`, `cast`, `title`, `imdb_score`, `budget_millions`, `views_millions`.")
    st.markdown("  - **How to Use:** Use sliders to define the criteria for 'emerging' talent (maximum number of titles, minimum average IMDb score). View tables listing identified directors and actors.")
    st.markdown("  - **Example Insight:** Discover individuals who are consistently involved in highly-rated content early in their careers.")
    st.markdown("- **Tool 50: Genre Synergy & Cross-Promotion Opportunities:**")
    st.markdown("  - **Purpose:** Analyze pairs of genres that frequently co-occur in titles and identify top-performing pairs by average IMDb score, suggesting potential cross-promotion opportunities.")
    st.markdown("  - **Key Columns Used:** `listed_in`, `imdb_score`, `views_millions`.")
    st.markdown("  - **How to Use:** Use a slider to set a minimum occurrence threshold for genre pairs. View a horizontal bar chart showing the top-performing genre pairs by average IMDb score.")
    st.markdown("  - **Example Insight:** Identify genre combinations that are popular and critically successful, suggesting potential for bundling or recommending content across these genres.")
    st.markdown("- **Tool 51: AI-Driven User Sentiment Analysis (Review Text):**")
    st.markdown("  - **Purpose:** (Similar to Tool 35) Analyze the sentiment of a user-provided text using AI, simulating how review sentiment could be analyzed at scale.")
    st.markdown("  - **Key Columns Used:** None directly from the dataset, uses user input.")
    st.markdown("  - **How to Use:** Paste a review text into the text area and click 'Analyze Sentiment with AI'. Requires Gemini API key.")
    st.markdown("  - **Example Insight:** Understand the potential application of AI for analyzing large volumes of user feedback.")
    st.markdown("- **Tool 52: Clustering for Content Discovery:**")
    st.markdown("  - **Purpose:** Use K-Means clustering to group content based on selected numeric features, potentially revealing hidden segments or types of content.")
    st.markdown("  - **Key Columns Used:** `imdb_score`, `release_year`, `budget_millions`, `views_millions`, `duration`, `type`, `title`, `listed_in`.")
    st.markdown("  - **How to Use:** Select numeric features for clustering and choose the number of clusters (k). Click 'Run Clustering'. View a scatter plot of the clusters (using PCA if more than 2 features) and tables summarizing the characteristics and top titles within each cluster.")
    st.markdown("  - **Example Insight:** Discover distinct groups of content based on their attributes, such as 'high-rated, recent movies' or 'older, low-budget TV shows'.")
    st.markdown("---")
    st.markdown("#### **🛠️ Utilities (Tool 53)**")
    st.markdown("- **Tool 53: Data Cleaning & Preparation Showcase:**")
    st.markdown("  - **Purpose:** Illustrate fundamental data cleaning and preparation steps essential for analysis.")
    st.markdown("  - **Key Columns Used:** Demonstrates concepts using various columns like `director`, `cast`, `country`, `imdb_score`, `duration`, `release_year`, `listed_in`.")
    st.markdown("  - **How to Use:** Expand the section to see demonstrations of handling missing values (with simulation if your data is clean), examples of feature engineering (like extracting numeric duration or creating decades), and data type conversion.")
    st.markdown("  - **Example Insight:** Understand the importance of cleaning data before analysis and see common techniques in action.")
    st.markdown("---")
    st.markdown("#### **🎮 Gamified Analytics (Tools 54-55)**")
    st.markdown("- **Tool 54: 'Guess the Views' Challenge:**")
    st.markdown("  - **Purpose:** Test your intuition about how many views (simulated) a title might have based on its attributes.")
    st.markdown("  - **Key Columns Used:** `views_millions`, `title`, `type`, `listed_in`, `release_year`, `imdb_score`.")
    st.markdown("  - **How to Use:** Click 'Get New Challenge Title' to see a random title and its details (excluding views). Enter your guess for views and click 'Submit Guess & Reveal'. Get feedback on your accuracy and earn points.")
    st.markdown("  - **Example Insight:** Develop a feel for how different content attributes might relate to viewership.")
    st.markdown("- **Tool 55: Netflix Trivia Challenge:**")
    st.markdown("  - **Purpose:** Test your knowledge about the dataset itself through multiple-choice trivia questions.")
    st.markdown("  - **Key Columns Used:** Various columns used to generate questions (e.g., `imdb_score`, `title`, `release_year`, `listed_in`, `country`).")
    st.markdown("  - **How to Use:** Click 'Start/Restart Trivia Challenge'. Answer the multiple-choice questions presented. See if your answer is correct and track your score.")
    st.markdown("  - **Example Insight:** Reinforce your understanding of key facts and patterns within the dataset.")

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)
total_titles = len(df)
movies_count = len(df[df['type'] == 'Movie']) if 'type' in df.columns else 0
tv_shows_count = len(df[df['type'] == 'TV Show']) if 'type' in df.columns else 0
countries_count = df['country'].nunique() if 'country' in df.columns else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Titles</h4>
        <p>{total_titles}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Movies</h4>
        <p>{movies_count}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>TV Shows</h4>
        <p>{tv_shows_count}</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Countries</h4>
        <p>{countries_count}</p>
    </div>
    """, unsafe_allow_html=True)

# Comprehensive Data Overview
with st.expander("📄 Comprehensive Data Overview", expanded=True):
    st.subheader("Initial Dataset Insights")

    if not df.empty:
        st.markdown("#### First 5 Rows:")
        st.dataframe(df.head())

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("#### Data Dimensions:")
            st.write(f"Rows: {df.shape[0]:,}")
            st.write(f"Columns: {df.shape[1]}")

        with col_info2:
            st.markdown("#### Missing Values (per column):")
            missing_counts = df.isnull().sum()
            missing_df = missing_counts[missing_counts > 0].rename("Missing Count").to_frame()
            if not missing_df.empty:
                st.dataframe(missing_df.T)
            else:
                st.success("No missing values found in the dataset.")

        st.markdown("#### Column Data Types:")
        st.dataframe(df.dtypes.rename("Data Type").to_frame().T)

        st.markdown("#### Basic Statistical Summary:")
        st.dataframe(df.describe(include='all'))

    else:
        st.warning("No data loaded to display overview.")


# Tool 1: Content Performance Analytics
with st.expander("📊 Tool 1: Content Performance Analytics"): # Renumbered (was 1)
    if 'imdb_score' in df.columns and 'views_millions' in df.columns:
        fig = px.scatter(df, x='imdb_score', y='views_millions', color='type', size='budget_millions',
                        title="Content Performance: Rating vs Viewership",
                        labels={'imdb_score': 'IMDB Score', 'views_millions': 'Views (Millions)'},
                        template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        top_performers = df.nlargest(5, 'views_millions')[['title', 'views_millions', 'imdb_score']]
        st.subheader("Top 5 Most Viewed")
        st.dataframe(top_performers)

# Tool 2: Genre Trend Analysis
with st.expander("📈 Tool 2: Genre Trend Analysis"): # Renumbered (was 2)
    if 'release_year' in df.columns and 'listed_in' in df.columns:
        # Data Preparation
        genre_data = []
        for _, row in df.iterrows():
            # Ensure 'listed_in' is a string and handle potential NaNs
            if pd.notna(row['listed_in']) and isinstance(row['listed_in'], str):
                genres = [g.strip() for g in row['listed_in'].split(',')]
                for genre in genres:
                    if genre: # Ensure genre is not an empty string
                        genre_data.append({'release_year': row['release_year'], 'genre': genre})
        
        if not genre_data:
            st.info("No valid genre data found to analyze.")
        else:
            genre_df = pd.DataFrame(genre_data)
            
            # Ensure release_year is numeric for proper sorting and grouping
            genre_df['release_year'] = pd.to_numeric(genre_df['release_year'], errors='coerce')
            genre_df.dropna(subset=['release_year'], inplace=True)
            genre_df['release_year'] = genre_df['release_year'].astype(int)

            genre_trends = genre_df.groupby(['release_year', 'genre']).size().reset_index(name='count')
            
            # Determine top N genres
            num_top_genres = st.slider("Number of Top Genres to Display:", min_value=3, max_value=15, value=6, key="genre_slider_tool2")
            top_genres_list = genre_df['genre'].value_counts().head(num_top_genres).index.tolist()
            
            genre_trends_top = genre_trends[genre_trends['genre'].isin(top_genres_list)].copy() # Use .copy()
            
            # Smoothing
            smoothing_window = st.slider("Smoothing Window (years):", min_value=1, max_value=7, value=3, step=2, key="smoothing_slider_tool2", help="Set to 1 for no smoothing (raw data).")

            if not genre_trends_top.empty:
                # Sort before applying rolling window
                genre_trends_top = genre_trends_top.sort_values(by=['genre', 'release_year'])
                
                if smoothing_window > 1:
                    # Apply rolling average per genre
                    genre_trends_top['display_count'] = genre_trends_top.groupby('genre')['count'].transform(
                        lambda x: x.rolling(window=smoothing_window, center=True, min_periods=1).mean()
                    )
                    y_axis_label = f'Smoothed Count of Titles ({smoothing_window}-year avg)'
                    plot_title = f"Top {num_top_genres} Genre Popularity Trends (Smoothed)"
                else:
                    genre_trends_top['display_count'] = genre_trends_top['count']
                    y_axis_label = 'Count of Titles'
                    plot_title = f"Top {num_top_genres} Genre Popularity Trends"

                fig = px.line(genre_trends_top,
                             x='release_year', y='display_count', color='genre',
                             title=plot_title,
                             labels={'release_year': 'Release Year', 'display_count': y_axis_label},
                             template="plotly_dark")
                
                if smoothing_window > 1 and st.checkbox("Show actual data points", value=False, key="show_actual_genre_points_tool2"):
                    for genre_val in top_genres_list: # Iterate using genre_val to avoid conflict
                        actual_data = genre_trends_top[genre_trends_top['genre'] == genre_val]
                        if not actual_data.empty:
                             fig.add_scatter(x=actual_data['release_year'], y=actual_data['count'], mode='markers', name=f'{genre_val} (Actual)',
                                             marker=dict(size=5, opacity=0.6))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"Top {num_top_genres} Most Popular Genres (Overall)")
                st.dataframe(genre_df['genre'].value_counts().head(num_top_genres).rename("Total Titles"))
            else:
                st.info("Not enough data for the selected top genres to display trends.")
    else:
        st.info("Release year and/or listed_in (genre) information not available for this analysis.")

# Tool 3: Geographic Content Distribution
with st.expander("🌍 Tool 3: Geographic Content Distribution"): # Renumbered (was 3)
    if 'country' in df.columns:
        # Handle potential multiple countries per title by taking the first one for this aggregation
        # Create a temporary series for this calculation to avoid modifying df
        first_country_series = df['country'].astype(str).apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else None).dropna()
        
        if not first_country_series.empty:
            top_n_countries = 10
            country_counts_top_n = first_country_series.value_counts().head(top_n_countries)
            
            fig = px.bar(country_counts_top_n, x=country_counts_top_n.values, y=country_counts_top_n.index, orientation='h',
                    title=f"Top {top_n_countries} Countries by Content Production (Primary Country)", 
                    labels={'x': 'Number of Titles', 'y': 'Country'},
                    template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(f"Share of Total Content for Top {top_n_countries} Countries")
            total_titles_all_countries = len(first_country_series)
            market_share_overall = (country_counts_top_n / total_titles_all_countries * 100).round(2)
            st.dataframe(market_share_overall.rename("Share of Total Content (%)"))
        else:
            st.info("No valid country data to display.")
    else:
        st.info("Country information not available for this analysis.")

# Tool 4: Content Duration Analysis
with st.expander("⏱️ Tool 4: Content Duration Analysis"): # Renumbered (was 4)
    if 'duration' in df.columns:
        # Extract numeric duration for movies
        movie_durations = []
        tv_seasons = []
        
        for _, row in df.iterrows():
            duration = str(row['duration'])
            if 'min' in duration:
                movie_durations.append(int(re.findall(r'\d+', duration)[0]))
            elif 'Season' in duration:
                tv_seasons.append(int(re.findall(r'\d+', duration)[0]))
        
        col1, col2 = st.columns(2)
        with col1:
            if movie_durations:
                fig = px.histogram(x=movie_durations, title="Movie Duration Distribution",
                                 labels={'x': 'Duration (minutes)', 'count': 'Frequency'},
                                 template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if tv_seasons:
                fig = px.histogram(x=tv_seasons, title="TV Show Seasons Distribution",
                                 labels={'x': 'Number of Seasons', 'count': 'Frequency'},
                                 template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# Tool 5: Rating Distribution Analysis
with st.expander("🏆 Tool 5: Rating Distribution Analysis"): # Renumbered (was 5)
    if 'rating' in df.columns:
        rating_counts = df['rating'].value_counts()
        fig = px.pie(values=rating_counts.values, names=rating_counts.index,
                    title="Content Rating Distribution",
                    template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age demographic insights
        mature_content = len(df[df['rating'].isin(['R', 'TV-MA'])])
        family_content = len(df[df['rating'].isin(['G', 'PG', 'TV-G', 'TV-Y'])])
        st.write(f"Mature Content: {mature_content} ({mature_content/len(df)*100:.1f}%)")
        st.write(f"Family-Friendly: {family_content} ({family_content/len(df)*100:.1f}%)")

# Tool 6: Release Year Timeline
with st.expander("📅 Tool 6: Release Year Timeline"): # Renumbered (was 6)
    if 'release_year' in df.columns:
        yearly_releases = df['release_year'].value_counts().sort_index()
        fig = px.area(x=yearly_releases.index, y=yearly_releases.values,
                     title="Content Release Timeline", labels={'x': 'Release Year', 'y': 'Number of Titles'},
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Decade analysis
        df['decade'] = (df['release_year'] // 10) * 10
        decade_counts = df['decade'].value_counts().sort_index()
        st.write("Content by Decade:", decade_counts.to_dict())

# Tool 7: Budget vs Performance ROI
with st.expander("💰 Tool 7: Budget vs Performance ROI"): # Renumbered (was 7)
    if 'budget_millions' in df.columns and 'views_millions' in df.columns:
        df_roi = df.copy()
        # Ensure relevant columns are numeric and handle NaNs by dropping rows for this specific analysis
        df_roi['budget_millions'] = pd.to_numeric(df_roi['budget_millions'], errors='coerce')
        df_roi['views_millions'] = pd.to_numeric(df_roi['views_millions'], errors='coerce')
        df_roi.dropna(subset=['budget_millions', 'views_millions'], inplace=True)

        # Calculate ROI, handling potential zero budget
        # Replace with np.nan if budget is 0 or very small to avoid infinite ROI
        # A small epsilon can be used if near-zero budgets are possible and problematic
        df_roi['roi'] = np.where(df_roi['budget_millions'] > 0.01, df_roi['views_millions'] / df_roi['budget_millions'], np.nan)
        
        # Filters for the plot
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        log_x = col_filter1.checkbox("Log Scale for Budget (X-axis)", key="roi_log_x")
        log_y = col_filter2.checkbox("Log Scale for ROI (Y-axis)", key="roi_log_y")
        marker_opacity = col_filter3.slider("Marker Opacity:", 0.1, 1.0, 0.7, key="roi_opacity")

        # Filter out NaN ROI values for plotting
        df_roi_plot = df_roi.dropna(subset=['roi'])

        if not df_roi_plot.empty:
            fig = px.scatter(df_roi_plot, x='budget_millions', y='roi', color='type', 
                            size='imdb_score' if 'imdb_score' in df_roi_plot.columns else None,
                            title="Budget vs ROI Analysis",
                            labels={'budget_millions': 'Budget (Millions)', 'roi': 'ROI (Views/Budget Ratio)'},
                            template="plotly_dark",
                            log_x=log_x, 
                            log_y=log_y,
                            opacity=marker_opacity)
            
            fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))) # Add border to markers

            st.plotly_chart(fig, use_container_width=True)
            
            # Display top ROI content based on the potentially filtered df_roi_plot
            if not df_roi_plot.empty:
                display_cols_roi = ['title']
                if 'budget_millions' in df_roi_plot.columns: display_cols_roi.append('budget_millions')
                if 'views_millions' in df_roi_plot.columns: display_cols_roi.append('views_millions')
                if 'roi' in df_roi_plot.columns: display_cols_roi.append('roi')
                if 'type' in df_roi_plot.columns: display_cols_roi.append('type')
                
                high_roi = df_roi_plot.nlargest(5, 'roi')[display_cols_roi]
                st.subheader("Top 5 Content by ROI (Return on Investment)")
                st.dataframe(high_roi.style.format({"roi": "{:.2f}", "budget_millions": "{:.1f}", "views_millions": "{:.1f}"}))
        else:
            st.info("Not enough valid data to calculate or display ROI.")
    else:
        st.info("Budget and/or viewership information not available for ROI analysis.")

# Tool 8: Content Correlation Matrix
# Tool 8: Content Correlation Matrix
with st.expander("🔗 Tool 8: Content Correlation Matrix"): # Renumbered (was 8)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r', text_auto=True,
                       template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Key Correlation Insights")
        corr_threshold = st.slider("Select Correlation Threshold:", 0.1, 1.0, 0.5, 0.05, key="corr_thresh_tool8")
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) >= corr_threshold:
                    st.markdown(f"- **{col1} & {col2}**: `{corr_val:.3f}` ({'Positive' if corr_val > 0 else 'Negative'} Correlation)")
    else:
        st.info("Not enough numeric columns (at least 2) available in the dataset to generate a correlation matrix.")

# Tool 9: Content Gap Analysis
with st.expander("📊 Tool 9: Content Gap Analysis"): # Renumbered (was 9)
    if 'country' in df.columns and 'listed_in' in df.columns:
        # Genre distribution by country
        country_genre_data = []
        for _, row in df.iterrows():
            genres = [g.strip() for g in str(row['listed_in']).split(',')]
            for genre in genres:
                country_genre_data.append({'country': row['country'], 'genre': genre})
        
        cg_df = pd.DataFrame(country_genre_data)
        pivot_table = cg_df.groupby(['country', 'genre']).size().unstack(fill_value=0)
        
        # Identify underrepresented genres per country
        st.subheader("Genre Gaps by Country")
        for country in pivot_table.index[:5]:
            country_genres = pivot_table.loc[country]
            missing_genres = country_genres[country_genres == 0].index.tolist()[:3]
            if missing_genres:
                st.write(f"**{country}**: Missing {', '.join(missing_genres)}")

# Tool 10: Predictive Analytics Dashboard
with st.expander("🔮 Tool 10: Predictive Analytics Dashboard"): # Renumbered (was 10)
    if 'imdb_score' in df.columns and 'views_millions' in df.columns:
        st.info(f"""
        **Disclaimer:** The models presented here (Simple Linear Regression, Polynomial Regression, SVR)
        are for illustrative purposes. They use 'IMDb Score' and 'Budget (Millions)'
        to predict 'Views (Millions)'. Real-world viewership is influenced by many more
        complex factors, and robust model development requires careful feature engineering,
        hyperparameter tuning, and cross-validation. The R-squared values here are on the test set.
        Lower MAE/RMSE and higher R-squared (closer to 1) generally indicate better performance.
        """)

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        df_model = df[['imdb_score', 'budget_millions', 'views_millions']].copy()
        # Ensure features are numeric, converting errors to NaN
        df_model['imdb_score'] = pd.to_numeric(df_model['imdb_score'], errors='coerce')
        df_model['budget_millions'] = pd.to_numeric(df_model['budget_millions'], errors='coerce')
        # Ensure target variable is also numeric
        df_model['views_millions'] = pd.to_numeric(df_model['views_millions'], errors='coerce')
        df_model.dropna(inplace=True)

        st.markdown(f"**Data for Modeling:** After preprocessing and removing rows with missing values in 'IMDb Score', 'Budget (Millions)', or 'Views (Millions)', there are **{len(df_model)}** rows available for training and testing the predictive models.")

        min_data_points_for_modeling = 30 # Increased threshold for more stable modeling

        if len(df_model) >= min_data_points_for_modeling:
            X = df_model[['imdb_score', 'budget_millions']]
            y = df_model['views_millions']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write(f"Training set size: {len(X_train)} samples. Test set size: {len(X_test)} samples.")

            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline

            model_type = st.selectbox(
                "Select Model Type:",
                ["Simple Linear Regression", "Polynomial Regression", "Support Vector Regressor (SVR)"],
                key="pred_model_type_tool10" # Added specific key
            )

            model = None
            pipeline = None

            if model_type == "Simple Linear Regression":
                pipeline = Pipeline([
                    ('scaler', StandardScaler()), # Good practice even for simple LR
                    ('linear_regression', LinearRegression())
                ])
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps['linear_regression']

            elif model_type == "Polynomial Regression":
                poly_degree = st.slider("Select Polynomial Degree:", 2, 5, 2, key="poly_degree_selector_tool10") # Changed key
                pipeline = Pipeline([
                    ('poly_features', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('linear_regression', LinearRegression())
                ])
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps['linear_regression']

            elif model_type == "Support Vector Regressor (SVR)":
                kernel = st.selectbox("Select SVR Kernel:", ['linear', 'rbf', 'poly'], key="svr_kernel_selector_tool10") # Changed key
                c_param = st.number_input("SVR C (Regularization parameter):", 0.1, 100.0, 1.0, 0.1, key="svr_c_param_tool10") # Changed key
                gamma_param = "scale"
                if kernel in ['rbf', 'poly']:
                    gamma_param_option = st.selectbox("SVR Gamma:", ['scale', 'auto', 'custom_value'], key="svr_gamma_option_selector_tool10") # Changed key
                    if gamma_param_option == 'custom_value':
                        gamma_param = st.number_input("Custom Gamma value:", 0.001, 10.0, 0.1, 0.001, format="%.3f", key="svr_gamma_value_tool10") # Changed key
                    else:
                        gamma_param = gamma_param_option
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR(kernel=kernel, C=c_param, gamma=gamma_param))
                ])
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps['svr']

            if pipeline: # If a model was selected and pipeline created
                predictions = pipeline.predict(X_test)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
                fig_pred.add_trace(go.Scatter(x=[min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())], 
                                        y=[min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())], 
                                        mode='lines', name='Perfect Prediction Line', line=dict(dash='dash')))
                fig_pred.update_layout(title=f"{model_type} - Model Performance (on Test Set)",
                                 xaxis_title="Actual Views (Millions)", yaxis_title="Predicted Views (Millions)",
                                 template="plotly_dark")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.subheader("Model Evaluation Metrics (on Test Set)")
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
                if r2 < 0:
                    st.warning(f"Note: A negative R-squared ({r2:.3f}) indicates the model performs worse than simply predicting the average views. This suggests the chosen features ('IMDb Score', 'Budget') may not have a strong linear or simple non-linear relationship with 'Views (Millions)' in this dataset, or there isn't enough data.")

                st.write(f"- R-squared: {r2:.3f}")
                st.write(f"- Mean Absolute Error (MAE): {mae:.2f} million views")
                st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f} million views")
                
                st.subheader("Model Insights")
                st.caption("Note: Coefficients are based on standardized (scaled) features due to the `StandardScaler` in the pipeline. An intercept occurs when scaled features are zero (i.e., original features are at their mean).")

                if model_type == "Simple Linear Regression" and hasattr(model, 'coef_'):
                    st.write(f"- IMDb Score Coefficient: {model.coef_[0]:.2f}")
                    st.write(f"- Budget (Millions) Coefficient: {model.coef_[1]:.2f}")
                    st.write(f"- Intercept: {model.intercept_:.2f} million views")
                elif model_type == "Polynomial Regression" and hasattr(model, 'coef_'):
                    poly_feature_names = pipeline.named_steps['poly_features'].get_feature_names_out(X_train.columns)
                    st.write("Polynomial Feature Coefficients:")
                    coeffs_df = pd.DataFrame({'Feature': poly_feature_names, 'Coefficient': model.coef_})
                    st.dataframe(coeffs_df)
                    st.write(f"- Intercept: {model.intercept_:.2f} million views")
                elif model_type == "Support Vector Regressor (SVR)":
                    if model.kernel == 'linear' and hasattr(model, 'coef_'):
                        st.write(f"- IMDb Score Coefficient: {model.coef_[0][0]:.2f}")
                        st.write(f"- Budget (Millions) Coefficient: {model.coef_[0][1]:.2f}")
                    else:
                        st.write("Coefficients are not directly interpretable for non-linear SVR kernels in the same way as linear models.")
                    st.write(f"- Intercept: {model.intercept_[0]:.2f} million views (Note: SVR intercept interpretation can differ)")
                    st.write(f"- Number of Support Vectors: {model.support_vectors_.shape[0]}")

            st.subheader("Try a Prediction")
            pred_imdb = st.number_input("Enter IMDb Score (e.g., 7.5):", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key="pred_imdb_input_tool10") # Added key
            pred_budget = st.number_input("Enter Budget (Millions, e.g., 50):", min_value=0.0, value=50.0, step=1.0, key="pred_budget_input_tool10") # Added key
            
            if st.button("Predict Views", key="predict_views_button_tool10"):
                if pipeline:
                    input_data = pd.DataFrame([[pred_imdb, pred_budget]], columns=['imdb_score', 'budget_millions'])
                    # The pipeline will handle scaling and polynomial features if applicable
                    predicted_views = pipeline.predict(input_data)
                    st.success(f"Predicted Views ({model_type}): {predicted_views[0]:.2f} million")
                else:
                    st.warning("Please select and train a model first.")
        else:
            st.warning(f"""
            Not enough data (only {len(df_model)} rows) after cleaning to reliably train and evaluate the predictive model. 
            A minimum of {min_data_points_for_modeling} data points with valid 'IMDb Score', 'Budget (Millions)', and 'Views (Millions)' is recommended.
            Please check your `netflix_analysis.csv` file for these columns and ensure they contain sufficient numeric data.
            """)
    else:
        st.info("IMDb score, budget, and/or viewership information not available for this predictive analysis.")

# Advanced Analytics Tools
st.header("🔬 Advanced Analytics")

# Tool 14: AI-Powered Insights - MOVED TO AI SECTION
# Tool 21: AI Chat with Dataset - MOVED TO AI SECTION

# Tool 11: Content Recommendation Engine
with st.expander("🎯 Tool 11: Content Recommendation Engine"): # Renumbered (was 12)
    if 'listed_in' in df.columns:
        user_genre = st.selectbox("Select preferred genre:",
                                 ['Drama', 'Comedy', 'Action', 'Horror', 'Sci-Fi', 'Crime'], key="tool11_genre_select") # Key updated
        
        # Simple content-based filtering
        genre_matches = df[df['listed_in'].str.contains(user_genre, na=False)]
        
        if not genre_matches.empty:
            if 'imdb_score' in df.columns:
                recommendations = genre_matches.nlargest(5, 'imdb_score')
            else:
                recommendations = genre_matches.head(5)
            
            st.subheader(f"Top {user_genre} Recommendations")
            st.dataframe(recommendations[['title', 'country', 'release_year']])
    else:
        st.info("'listed_in' column not available for recommendations.")

# Tool 12: Data Export & Reporting
with st.expander("📤 Tool 12: Data Export & Reporting"): # Renumbered (was 15)
    export_format = st.selectbox("Export format:", ["CSV", "JSON", "Excel Summary"])
    
    if st.button("Generate Export"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "netflix_analysis.csv", "text/csv")
        elif export_format == "JSON":
            json_data = df.to_json(orient='records', indent=4)
            st.download_button("Download JSON", json_data, "netflix_analysis.json", "application/json")
        elif export_format == "Excel Summary":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Full Data', index=False)
                
                if not df.empty:
                    df.describe(include='all').to_excel(writer, sheet_name='Summary Statistics')

                    # Key Metrics Sheet
                    metrics_data = {
                        "Metric": ["Total Titles", "Movies", "TV Shows", "Unique Countries"],
                        "Value": [
                            len(df),
                            len(df[df['type'] == 'Movie']) if 'type' in df.columns else 'N/A',
                            len(df[df['type'] == 'TV Show']) if 'type' in df.columns else 'N/A',
                            df['country'].nunique() if 'country' in df.columns else 'N/A'
                        ]
                    }
                    pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Key Metrics', index=False)

                    if 'views_millions' in df.columns and 'title' in df.columns:
                        df.nlargest(10, 'views_millions')[['title', 'views_millions']].to_excel(writer, sheet_name='Top Content (Views)', index=False)
                    if 'imdb_score' in df.columns and 'title' in df.columns:
                        df.nlargest(10, 'imdb_score')[['title', 'imdb_score']].to_excel(writer, sheet_name='Top Content (IMDb)', index=False)
                    if 'listed_in' in df.columns:
                        # Explode genres for accurate counting
                        genres_exploded = df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')
                        genres_exploded['genre'].value_counts().to_excel(writer, sheet_name='Genre Counts', header=['Count'])
                    if 'country' in df.columns:
                        # Use primary country
                        df.assign(primary_country=df['country'].astype(str).apply(lambda x: x.split(',')[0].strip()))['primary_country'].value_counts().to_excel(writer, sheet_name='Country Counts', header=['Count'])

            excel_data = output.getvalue()
            st.download_button(label="Download Excel Summary",
                               data=excel_data,
                               file_name="netflix_summary_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# Tool 13: Director Performance Analysis
with st.expander("🎬 Tool 13: Director Performance Analysis"): # Renumbered (was 16)
    if 'director' in df.columns and 'title' in df.columns:
        st.subheader("Director Analysis")
        # Filter out rows where director is NaN or 'Unknown' if necessary, though sample data is clean
        # For this example, we'll assume directors are mostly single individuals or known groups
        # In a real dataset, director column might need more cleaning (e.g., splitting multiple directors)        
        # Handle cases where director might be NaN or not a string
        valid_directors_df = df[df['director'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
        director_counts = valid_directors_df['director'].value_counts().head(10)

        fig_director_titles = px.bar(director_counts, x=director_counts.index, y=director_counts.values,
                                     labels={'x': 'Director', 'y': 'Number of Titles'},
                                     title="Top 10 Directors by Number of Titles",
                                     template="plotly_dark")
        st.plotly_chart(fig_director_titles, use_container_width=True)

        if 'imdb_score' in df.columns:
            # Calculate average IMDb score per director
            # For simplicity, considering only directors with at least 2 titles for score analysis
            director_title_counts = df['director'].value_counts()
            directors_for_score_analysis = director_title_counts[director_title_counts >= 2].index
            
            if not directors_for_score_analysis.empty:
                avg_score_by_director = valid_directors_df[valid_directors_df['director'].isin(directors_for_score_analysis)].groupby('director')['imdb_score'].mean().sort_values(ascending=False).head(10)
                fig_director_score = px.bar(avg_score_by_director, x=avg_score_by_director.index, y=avg_score_by_director.values,
                                             labels={'x': 'Director', 'y': 'Average IMDb Score'},
                                             title="Top Directors by Average IMDb Score (min. 2 titles)",
                                             template="plotly_dark")
                st.plotly_chart(fig_director_score, use_container_width=True)
            else:
                st.write("Not enough data for director IMDb score analysis (requires directors with >= 2 titles).")
    else:
        st.info("Director and/or title information not available for this analysis.")

# Tool 14: Title Word Cloud
with st.expander("☁️ Tool 14: Title Word Cloud"): # Renumbered (was 17)
    if 'title' in df.columns:
        st.subheader("Word Cloud from Content Titles")
        text = " ".join(title for title in df['title'].astype(str))
        if text.strip():
            # Use a colormap that works well on dark backgrounds
            wordcloud = WordCloud(width=800, height=400, background_color='#121212', colormap="viridis", color_func=lambda *args, **kwargs: "lightblue").generate(text)
            fig, ax = plt.subplots(figsize=(10,5))
            fig.patch.set_facecolor('#121212') # Match app background
            ax.set_facecolor('#121212')
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No titles available to generate a word cloud.")
    else:
        st.info("Title information not available for word cloud generation.")

# Tool 15: Content Type Evolution Over Time
with st.expander("🔄 Tool 15: Content Type Evolution Over Time"): # Renumbered (was 18)
    if 'release_year' in df.columns and 'type' in df.columns:
        content_type_evolution = df.groupby(['release_year', 'type']).size().reset_index(name='count')
        fig = px.line(content_type_evolution, x='release_year', y='count', color='type',
                     title="Content Type Releases Over Time", 
                     labels={'release_year': 'Release Year', 'count': 'Number of Titles'},
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Release year and/or type information not available for this analysis.")

# Tool 16: Actor/Cast Performance Analysis
with st.expander("🎭 Tool 16: Actor/Cast Performance Analysis"): # Renumbered (was 19)
    if 'cast' in df.columns and 'title' in df.columns:
        st.subheader("Actor/Cast Analysis")
        
        # Explode cast members for individual analysis
        # Ensure 'cast' is string and handle potential NaNs before splitting
        df_cast = df.dropna(subset=['cast'])
        if not df_cast.empty:
            actor_list = df_cast.assign(actor=df_cast['cast'].str.split(', ')).explode('actor')
            actor_list['actor'] = actor_list['actor'].str.strip() # Clean up actor names

            # Top actors by number of titles
            actor_counts = actor_list['actor'].value_counts().head(10)
            if not actor_counts.empty:
                fig_actor_titles = px.bar(actor_counts, x=actor_counts.index, y=actor_counts.values,
                                          labels={'x': 'Actor', 'y': 'Number of Titles'},
                                          title="Top 10 Actors by Number of Titles Appeared In",
                                          template="plotly_dark")
                st.plotly_chart(fig_actor_titles, use_container_width=True)
            else:
                st.write("Not enough data for actor title count analysis.")

            # Top actors by average IMDb score
            if 'imdb_score' in df.columns:
                # Consider actors with at least 2 appearances for score analysis
                actor_title_counts = actor_list['actor'].value_counts()
                actors_for_score_analysis = actor_title_counts[actor_title_counts >= 1].index # Lowered to 1 for sample data
                
                if not actors_for_score_analysis.empty:
                    avg_score_by_actor = actor_list[actor_list['actor'].isin(actors_for_score_analysis)].groupby('actor')['imdb_score'].mean().sort_values(ascending=False).head(10)
                    if not avg_score_by_actor.empty:
                        fig_actor_score = px.bar(avg_score_by_actor, x=avg_score_by_actor.index, y=avg_score_by_actor.values,
                                                 labels={'x': 'Actor', 'y': 'Average IMDb Score of Titles'},
                                                 title="Top Actors by Average IMDb Score (min. 1 title)",
                                                 template="plotly_dark")
                        st.plotly_chart(fig_actor_score, use_container_width=True)
                    else:
                        st.write("Could not calculate average IMDb scores for actors.")
                else:
                    st.write("Not enough data for actor IMDb score analysis (requires actors with >= 1 title).")
        else:
            st.write("No cast information available to analyze.")
    else:
        st.info("Cast, title, and/or IMDb score information not available for this analysis.")

# Tool 17: Genre Deep Dive
with st.expander("🔎 Tool 17: Genre Deep Dive"): # Renumbered (was 20)
    if 'listed_in' in df.columns and 'release_year' in df.columns and 'imdb_score' in df.columns and 'title' in df.columns:
        all_genres = sorted(list(set(g.strip() for sublist in df['listed_in'].str.split(',') for g in sublist if g.strip())))
        selected_genre = st.selectbox("Select a Genre for Deep Dive:", all_genres)

        if selected_genre:
            genre_df = df[df['listed_in'].str.contains(selected_genre, case=False, na=False)]
            st.subheader(f"Deep Dive: {selected_genre}")

            if not genre_df.empty:
                st.metric(f"Titles in {selected_genre}", len(genre_df))
                st.metric(f"Average IMDb Score for {selected_genre}", f"{genre_df['imdb_score'].mean():.2f}" if not genre_df['imdb_score'].empty else "N/A")

                # Release trend for the selected genre
                genre_release_trend = genre_df.groupby('release_year').size().reset_index(name='count')
                fig_genre_trend = px.line(genre_release_trend, x='release_year', y='count', 
                                          title=f"Release Trend for {selected_genre}",
                                          template="plotly_dark")
                st.plotly_chart(fig_genre_trend, use_container_width=True)

                st.subheader(f"Top 5 Titles in {selected_genre} (by IMDb Score)")
                st.dataframe(genre_df.nlargest(5, 'imdb_score')[['title', 'release_year', 'imdb_score', 'type']])
            else:
                st.write(f"No titles found for the genre: {selected_genre}")
    else:
        st.info("Required columns (listed_in, release_year, imdb_score, title) not available for Genre Deep Dive.")

# Tool 18: Content Freshness Analysis
with st.expander("⏳ Tool 18: Content Freshness Analysis"): # Renumbered (was 22)
    if 'release_year' in df.columns:
        st.subheader("Content Age Analysis")
        current_year = datetime.now().year
        df_copy = df.copy() # Work on a copy to avoid modifying the original df
        df_copy['content_age'] = current_year - df_copy['release_year']

        fig_age_dist = px.histogram(df_copy, x='content_age', nbins=20,
                                    title="Distribution of Content Age (Years)",
                                    labels={'content_age': 'Content Age (Years)', 'count': 'Frequency'},
                                    template="plotly_dark")
        st.plotly_chart(fig_age_dist, use_container_width=True)

        if 'imdb_score' in df_copy.columns:
            fig_age_score = px.scatter(df_copy, x='content_age', y='imdb_score', trendline="ols",
                                       title="Content Age vs. IMDb Score",
                                       labels={'content_age': 'Content Age (Years)', 'imdb_score': 'IMDb Score'},
                                       template="plotly_dark")
            st.plotly_chart(fig_age_score, use_container_width=True)

        if 'views_millions' in df_copy.columns:
            fig_age_views = px.scatter(df_copy, x='content_age', y='views_millions', trendline="ols",
                                       title="Content Age vs. Views (Millions)",
                                       labels={'content_age': 'Content Age (Years)', 'views_millions': 'Views (Millions)'},
                                       template="plotly_dark")
            st.plotly_chart(fig_age_views, use_container_width=True)
    else:
        st.info("Release year information not available for content freshness analysis.")

# Tool 19: Interactive World Map of Content Production
with st.expander("🗺️ Tool 19: Interactive World Map of Content Production"): # Renumbered (was 23)
    if 'country' in df.columns:
        st.subheader("Global Content Production Map")
        # Handle multiple countries by taking the first one listed for simplicity in mapping
        # A more advanced approach might involve exploding rows or using a primary production country
        df_map = df.copy()
        df_map['primary_country'] = df_map['country'].astype(str).apply(lambda x: x.split(',')[0].strip())
        
        country_counts_map = df_map['primary_country'].value_counts().reset_index()
        country_counts_map.columns = ['country', 'title_count']

        if not country_counts_map.empty:
            fig_map = px.choropleth(country_counts_map, 
                                    locations="country", 
                                    locationmode='country names', 
                                    color="title_count",
                                    hover_name="country", 
                                    color_continuous_scale=px.colors.sequential.Plasma,
                                    title="Number of Titles Produced by Country",
                                    template="plotly_dark")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.write("No country data available to display on the map.")
    else:
        st.info("Country information not available for map visualization.")

# Tool 20: Movie vs. TV Show Deep Comparison
with st.expander("🎬 vs 📺 Tool 20: Movie vs. TV Show Deep Comparison"): # Renumbered (was 24)
    if 'type' in df.columns:
        st.subheader("Movie vs. TV Show Metrics")
        movies_df = df[df['type'] == 'Movie'].copy() # Explicitly create a copy
        tv_shows_df = df[df['type'] == 'TV Show'].copy() # Explicitly create a copy

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Movies")
            st.metric("Total Movies", len(movies_df))
            if 'imdb_score' in movies_df.columns:
                st.metric("Avg. IMDb Score (Movies)", f"{movies_df['imdb_score'].mean():.2f}" if not movies_df.empty else "N/A")
            if 'duration' in movies_df.columns: # Assuming duration for movies is in minutes
                movies_df['duration_numeric'] = movies_df['duration'].str.extract('(\d+)').astype(float)
                st.metric("Avg. Duration (Movies)", f"{movies_df['duration_numeric'].mean():.0f} min" if not movies_df.empty else "N/A")
            if 'rating' in movies_df.columns and not movies_df.empty:
                fig_movie_ratings = px.pie(movies_df, names='rating', title='Movie Rating Distribution',
                                           hole=0.3, # Donut chart
                                           template="plotly_dark")
                st.plotly_chart(fig_movie_ratings, use_container_width=True)

        with col2:
            st.markdown("#### TV Shows")
            st.metric("Total TV Shows", len(tv_shows_df))
            if 'imdb_score' in tv_shows_df.columns:
                st.metric("Avg. IMDb Score (TV Shows)", f"{tv_shows_df['imdb_score'].mean():.2f}" if not tv_shows_df.empty else "N/A")
            if 'duration' in tv_shows_df.columns: # Assuming duration for TV shows is in seasons
                tv_shows_df['duration_numeric'] = tv_shows_df['duration'].str.extract('(\d+)').astype(float)
                st.metric("Avg. Seasons (TV Shows)", f"{tv_shows_df['duration_numeric'].mean():.1f}" if not tv_shows_df.empty else "N/A")
            if 'rating' in tv_shows_df.columns and not tv_shows_df.empty:
                fig_tv_ratings = px.pie(tv_shows_df, names='rating', title='TV Show Rating Distribution',
                                        hole=0.3, # Donut chart
                                        template="plotly_dark")
                st.plotly_chart(fig_tv_ratings, use_container_width=True)
    else:
        st.info("Content 'type' information not available for this comparison.")

# Tool 21: Release Month/Seasonality Analysis
with st.expander("🗓️ Tool 21: Release Month/Seasonality Analysis"): # Renumbered (was 25)
    if 'date_added' in df.columns:
        st.subheader("Content Addition Seasonality")
        df_season = df.copy()
        df_season['date_added'] = pd.to_datetime(df_season['date_added'], errors='coerce')
        df_season.dropna(subset=['date_added'], inplace=True) # Drop rows where date_added couldn't be parsed
        df_season['month_added'] = df_season['date_added'].dt.month_name()
        
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        titles_by_month = df_season['month_added'].value_counts().reindex(month_order, fill_value=0)
        
        fig_month_releases = px.bar(titles_by_month, x=titles_by_month.index, y=titles_by_month.values,
                                    title="Number of Titles Added by Month",
                                    labels={'x': 'Month Added', 'y': 'Number of Titles'},
                                    template="plotly_dark")
        st.plotly_chart(fig_month_releases, use_container_width=True)
    else:
        st.info("'date_added' column not available for seasonality analysis.")

# Tool 22: Keyword Search in Titles
with st.expander("🔑 Tool 22: Keyword Search in Titles"): # Renumbered (was 26)
    if 'title' in df.columns:
        st.subheader("Search Titles by Keyword")
        search_term = st.text_input("Enter keyword to search in titles:", placeholder="e.g., Love, War, Space")
        if search_term:
            # Case-insensitive search
            results_df = df[df['title'].str.contains(search_term, case=False, na=False)]
            if not results_df.empty:
                st.write(f"Found {len(results_df)} titles containing '{search_term}':")
                display_cols = ['title', 'type', 'release_year']
                if 'imdb_score' in results_df.columns:
                    display_cols.append('imdb_score')
                st.dataframe(results_df[display_cols])
            else:
                st.write(f"No titles found containing '{search_term}'.")
    else:
        st.info("'title' column not available for keyword search.")

# Tool 23: Content Rating vs. IMDb Score Analysis
with st.expander("🔞 Tool 23: Content Rating vs. IMDb Score Analysis"): # Renumbered (was 27)
    if 'rating' in df.columns and 'imdb_score' in df.columns: # Checks if both columns are initially present
        st.subheader("Average IMDb Score by Content Rating")
        
        df_tool27 = df.copy()
        # Ensure 'imdb_score' is numeric. If not, it becomes NaN.
        df_tool27['imdb_score'] = pd.to_numeric(df_tool27['imdb_score'], errors='coerce')
        
        # Drop rows where 'rating' or 'imdb_score' (after conversion) is NaN
        df_tool27.dropna(subset=['rating', 'imdb_score'], inplace=True)

        if not df_tool27.empty: # Check if there's data left after cleaning
            avg_score_by_rating = df_tool27.groupby('rating')['imdb_score'].mean().sort_values(ascending=False).reset_index()
            
            if not avg_score_by_rating.empty: # Check if groupby operation yielded results
                fig_rating_score = px.bar(avg_score_by_rating, x='rating', y='imdb_score',
                                          title="Average IMDb Score for Each Content Rating",
                                          labels={'rating': 'Content Rating', 'imdb_score': 'Average IMDb Score'},
                                          color='rating',
                                          template="plotly_dark")
                st.plotly_chart(fig_rating_score, use_container_width=True)

                st.subheader("IMDb Score Distribution by Rating (Box Plot)")
                fig_box_rating_score = px.box(df_tool27, x='rating', y='imdb_score', color='rating',
                                          title="IMDb Score Distribution by Content Rating",
                                          labels={'rating': 'Content Rating', 'imdb_score': 'IMDb Score'},
                                          template="plotly_dark")
                st.plotly_chart(fig_box_rating_score, use_container_width=True)
            else:
                st.write("Not enough valid data to analyze IMDb score by rating after filtering.")
        else:
            st.write("No valid data for 'rating' and 'imdb_score' columns found after attempting to clean 'imdb_score'.")
    elif 'rating' in df.columns and 'imdb_score' not in df.columns:
        st.info("The 'imdb_score' column is missing, which is required for this analysis. Please ensure your dataset includes it.")
    elif 'imdb_score' in df.columns and 'rating' not in df.columns:
        st.info("The 'rating' column is missing, which is required for this analysis. Please ensure your dataset includes it.")
    else: # Neither column is present
        st.info("Both 'rating' and 'imdb_score' columns are missing and required for this analysis.")

# Tool 24: Top Director-Actor Collaborations
with st.expander("🤝 Tool 24: Top Director-Actor Collaborations"): # Renumbered (was 28)
    if 'director' in df.columns and 'cast' in df.columns and 'title' in df.columns:
        st.subheader("Most Frequent Director-Actor Pairs")
        df_collab = df.dropna(subset=['director', 'cast'])
        if not df_collab.empty:
            collaborations = []
            for _, row in df_collab.iterrows():
                directors = [d.strip() for d in str(row['director']).split(',')]
                actors = [a.strip() for a in str(row['cast']).split(',')]
                for director in directors:
                    if director == "Unknown Director": continue # Skip generic unknown directors
                    for actor in actors:
                        if actor == "Unknown Actor": continue # Skip generic unknown actors
                        collaborations.append((director, actor))
            
            if collaborations:
                collab_counts = Counter(collaborations).most_common(10)
                if collab_counts:
                    collab_df = pd.DataFrame(collab_counts, columns=['Director-Actor Pair', 'Number of Collaborations'])
                    collab_df['Director-Actor Pair'] = collab_df['Director-Actor Pair'].apply(lambda x: f"{x[0]} - {x[1]}")
                    
                    fig_collab = px.bar(collab_df, y='Director-Actor Pair', x='Number of Collaborations',
                                        orientation='h', title="Top 10 Director-Actor Collaborations",
                                        labels={'Number of Collaborations': 'Number of Titles Together'},
                                        template="plotly_dark")
                    fig_collab.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_collab, use_container_width=True)
                else:
                    st.write("No significant director-actor collaborations found.")
            else:
                st.write("Not enough data to determine collaborations.")
        else:
            st.write("Director or cast information is largely missing.")
    else:
        st.info("'director', 'cast', and/or 'title' columns not available for collaboration analysis.")

# Tool 25: IMDb Score Trends Over Release Years
with st.expander("📈 Tool 25: IMDb Score Trends Over Release Years"): # Renumbered (was 29)
    if 'release_year' in df.columns and 'imdb_score' in df.columns:
        st.subheader("Average IMDb Score of Content by Release Year")
        avg_score_by_year = df.groupby('release_year')['imdb_score'].mean().reset_index()
        avg_score_by_year = avg_score_by_year.sort_values('release_year')

        if not avg_score_by_year.empty:
            fig_score_trend = px.line(avg_score_by_year, x='release_year', y='imdb_score',
                                      title="Trend of Average IMDb Scores Over Release Years",
                                      labels={'release_year': 'Release Year', 'imdb_score': 'Average IMDb Score'},
                                      template="plotly_dark")
            fig_score_trend.add_trace(go.Scatter(x=avg_score_by_year['release_year'], y=avg_score_by_year['imdb_score'].rolling(window=5, center=True, min_periods=1).mean(),
                                                 mode='lines', name='5-Year Rolling Avg', 
                                                 line=dict(dash='dash', color='rgba(255,255,255,0.5)'))) # Lighter dash for dark theme
            st.plotly_chart(fig_score_trend, use_container_width=True)
        else:
            st.write("Not enough data to analyze IMDb score trends over years.")
    else:
        st.info("'release_year' and/or 'imdb_score' columns not available for this analysis.")

# Tool 26: Multi-Country Content Profile Comparison
with st.expander("🌍 Tool 26: Multi-Country Content Profile Comparison"): # Renumbered (was 30)
    if 'country' in df.columns and 'type' in df.columns and 'listed_in' in df.columns:
        st.subheader("Compare Content Profiles of Top Countries")
        
        # Use primary country for simplicity
        df_country_comp = df.copy()
        df_country_comp['primary_country'] = df_country_comp['country'].astype(str).apply(lambda x: x.split(',')[0].strip())
        
        top_countries_list = df_country_comp['primary_country'].value_counts().nlargest(10).index.tolist()
        selected_countries = st.multiselect("Select up to 3 countries to compare:", top_countries_list, default=top_countries_list[:2] if len(top_countries_list) >=2 else top_countries_list[:1])

        if selected_countries:
            comparison_df = df_country_comp[df_country_comp['primary_country'].isin(selected_countries)]
            
            st.markdown("#### Content Type Distribution by Selected Country")
            fig_type_comp = px.bar(comparison_df.groupby(['primary_country', 'type']).size().reset_index(name='count'),
                                 x='primary_country', y='count', color='type', barmode='group',
                                 title="Movie vs. TV Show Count by Country",
                                 template="plotly_dark")
            st.plotly_chart(fig_type_comp, use_container_width=True)

            if 'imdb_score' in comparison_df.columns:
                st.markdown("#### Average IMDb Score by Selected Country")
                fig_imdb_comp = px.bar(comparison_df.groupby('primary_country')['imdb_score'].mean().reset_index(),
                                     x='primary_country', y='imdb_score', color='primary_country',
                                     title="Average IMDb Score by Country",
                                     template="plotly_dark")
                st.plotly_chart(fig_imdb_comp, use_container_width=True)
        else:
            st.write("Please select at least one country.")
    else:
        st.info("'country', 'type', and/or 'listed_in' columns not available for this analysis.")

# Tool 27: Content Length vs. IMDb Score
with st.expander("📏 Tool 27: Content Length vs. IMDb Score"): # Renumbered (was 31)
    if 'duration' in df.columns and 'imdb_score' in df.columns and 'type' in df.columns:
        st.subheader("Content Length vs. IMDb Score")
        
        df_tool31 = df.copy()
        df_tool31['imdb_score'] = pd.to_numeric(df_tool31['imdb_score'], errors='coerce')
        df_tool31.dropna(subset=['duration', 'imdb_score', 'type'], inplace=True)

        movies_df_31 = df_tool31[df_tool31['type'] == 'Movie'].copy()
        tv_shows_df_31 = df_tool31[df_tool31['type'] == 'TV Show'].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Movies: Duration (min) vs. IMDb Score")
            if 'duration' in movies_df_31.columns and not movies_df_31.empty:
                movies_df_31['duration_numeric'] = movies_df_31['duration'].str.extract('(\d+)').astype(float)
                movies_df_31.dropna(subset=['duration_numeric'], inplace=True)
                if not movies_df_31.empty:
                    fig_movie_len_score = px.scatter(movies_df_31, x='duration_numeric', y='imdb_score',
                                                     title="Movie Duration vs. IMDb Score",
                                                     labels={'duration_numeric': 'Duration (minutes)', 'imdb_score': 'IMDb Score'},
                                                     trendline="ols",
                                                     template="plotly_dark")
                    st.plotly_chart(fig_movie_len_score, use_container_width=True)
                else:
                    st.write("Not enough valid movie duration data for analysis.")
            else:
                st.write("Movie duration data not available.")

        with col2:
            st.markdown("#### TV Shows: Seasons vs. IMDb Score")
            if 'duration' in tv_shows_df_31.columns and not tv_shows_df_31.empty:
                tv_shows_df_31['duration_numeric'] = tv_shows_df_31['duration'].str.extract('(\d+)').astype(float)
                tv_shows_df_31.dropna(subset=['duration_numeric'], inplace=True)
                if not tv_shows_df_31.empty:
                    fig_tv_len_score = px.scatter(tv_shows_df_31, x='duration_numeric', y='imdb_score',
                                                  title="TV Show Seasons vs. IMDb Score",
                                                  labels={'duration_numeric': 'Number of Seasons', 'imdb_score': 'IMDb Score'},
                                                  trendline="ols",
                                                  template="plotly_dark")
                    st.plotly_chart(fig_tv_len_score, use_container_width=True)
                else:
                    st.write("Not enough valid TV show season data for analysis.")
            else:
                st.write("TV show duration data not available.")
    else:
        st.info("'duration', 'imdb_score', and/or 'type' columns not available for this analysis.")

# Tool 28: Genre Co-occurrence Analysis
with st.expander("🤝 Tool 28: Genre Co-occurrence Analysis"): # Renumbered (was 32)
    if 'listed_in' in df.columns:
        st.subheader("Most Frequent Genre Combinations")
        
        df_tool32 = df.copy()
        df_tool32.dropna(subset=['listed_in'], inplace=True)

        if not df_tool32.empty:
            genre_combinations = []
            for genres_str in df_tool32['listed_in']:
                genres = sorted([g.strip() for g in genres_str.split(',') if g.strip()])
                if len(genres) > 1:
                    # Create pairs of genres
                    for i in range(len(genres)):
                        for j in range(i + 1, len(genres)):
                            genre_combinations.append(tuple(sorted((genres[i], genres[j])))) # Ensure consistent order for counting
            
            if genre_combinations:
                co_occurrence_counts = Counter(genre_combinations).most_common(10)
                co_occurrence_df = pd.DataFrame(co_occurrence_counts, columns=['Genre Pair', 'Count'])
                co_occurrence_df['Genre Pair'] = co_occurrence_df['Genre Pair'].apply(lambda x: f"{x[0]} & {x[1]}")

                fig_co_occurrence = px.bar(co_occurrence_df, y='Genre Pair', x='Count',
                                           orientation='h', title="Top 10 Most Frequent Genre Combinations",
                                           labels={'Count': 'Number of Titles'},
                                           template="plotly_dark")
                fig_co_occurrence.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_co_occurrence, use_container_width=True)
            else:
                st.write("No genre combinations found (titles must have more than one genre listed).")
        else:
            st.write("No 'listed_in' data available for analysis.")
    else:
        st.info("'listed_in' column not available for genre co-occurrence analysis.")

# Tool 29: Cast Size vs. Performance
with st.expander("👥 Tool 29: Cast Size vs. Performance"): # Renumbered (was 33)
    if 'cast' in df.columns and ('imdb_score' in df.columns or 'views_millions' in df.columns):
        st.subheader("Cast Size vs. Performance Metrics")
        
        df_tool33 = df.copy()
        df_tool33['cast_size'] = df_tool33['cast'].astype(str).apply(lambda x: len(x.split(',')) if pd.notna(x) and x.strip() else 0)
        df_tool33 = df_tool33[df_tool33['cast_size'] > 0].copy() # Only consider titles with listed cast

        if not df_tool33.empty:
            if 'imdb_score' in df_tool33.columns:
                df_tool33['imdb_score'] = pd.to_numeric(df_tool33['imdb_score'], errors='coerce')
                df_tool33.dropna(subset=['imdb_score'], inplace=True)
                if not df_tool33.empty:
                    fig_cast_score = px.scatter(df_tool33, x='cast_size', y='imdb_score', trendline="ols",
                                                title="Cast Size vs. IMDb Score",
                                                labels={'cast_size': 'Number of Cast Members', 'imdb_score': 'IMDb Score'},
                                                template="plotly_dark")
                    st.plotly_chart(fig_cast_score, use_container_width=True)
                else:
                    st.write("Not enough valid data for Cast Size vs. IMDb Score analysis.")

            if 'views_millions' in df_tool33.columns:
                df_tool33['views_millions'] = pd.to_numeric(df_tool33['views_millions'], errors='coerce')
                df_tool33.dropna(subset=['views_millions'], inplace=True)
                if not df_tool33.empty:
                    fig_cast_views = px.scatter(df_tool33, x='cast_size', y='views_millions', trendline="ols",
                                                title="Cast Size vs. Views (Millions)",
                                                labels={'cast_size': 'Number of Cast Members', 'views_millions': 'Views (Millions)'},
                                                template="plotly_dark")
                    st.plotly_chart(fig_cast_views, use_container_width=True)
                else:
                    st.write("Not enough valid data for Cast Size vs. Views analysis.")
        else:
            st.write("No valid 'cast' data available for analysis.")
    else:
        st.info("'cast' column and at least one performance metric (imdb_score or views_millions) are required for this analysis.")

# Tool 30: Content Addition Trend (Yearly)
with st.expander("📅 Tool 30: Content Addition Trend (Yearly)"): # Renumbered (was 34)
    if 'date_added' in df.columns and 'type' in df.columns:
        st.subheader("Content Added to Netflix by Year and Type")
        df_tool34 = df.copy()
        df_tool34['date_added'] = pd.to_datetime(df_tool34['date_added'], errors='coerce')
        df_tool34.dropna(subset=['date_added', 'type'], inplace=True)
        
        if not df_tool34.empty:
            df_tool34['year_added'] = df_tool34['date_added'].dt.year
            yearly_additions = df_tool34.groupby(['year_added', 'type']).size().reset_index(name='count')
            
            fig_yearly_additions = px.bar(yearly_additions, x='year_added', y='count', color='type',
                                          title="Number of Titles Added to Netflix per Year by Type",
                                          labels={'year_added': 'Year Added', 'count': 'Number of Titles'},
                                          template="plotly_dark")
            st.plotly_chart(fig_yearly_additions, use_container_width=True)
        else:
            st.write("No valid 'date_added' or 'type' data available for analysis.")
    else:
        st.info("'date_added' and 'type' columns are required for this analysis.")

# AI Powered Tools Section
st.header("🧠 AI-Powered Tools")

# Tool 31: AI-Powered Insights
with st.expander("🤖 Tool 31: AI-Powered General Insights"): # Renumbered (was 14)
    if gemini_key:
        analysis_type = st.selectbox("Select analysis type:", 
                                   ["Overall Content Strategy", "Potential Market Gaps", "Key Performance Drivers", "Future Trend Predictions"], key="ai_insights_type") 
        
        if st.button("Generate AI Insights", key="ai_insights_button"):
            prompt = f"""
            Analyze this Netflix dataset summary for {analysis_type}:
            
            Dataset: {len(df)} titles
            Content mix: {df['type'].value_counts().to_dict() if 'type' in df.columns else 'N/A'}
            Top countries: {df['country'].value_counts().head(3).to_dict() if 'country' in df.columns else 'N/A'}
            Release years: {df['release_year'].min()}-{df['release_year'].max() if 'release_year' in df.columns else 'N/A'}
            
            Provide 3-5 concise, actionable insights for {analysis_type} based on the provided Netflix dataset summary. Focus on high-level strategic points.
            """
            
            try:
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating AI insights. Ensure the API key is correct and the model 'gemini-1.5-flash-latest' is available: {e}")
    else:
        st.info("Enter Gemini API key in the sidebar to use AI-Powered Insights.")

# Tool 32: AI Chat with Dataset
with st.expander("💬 Tool 32: AI Chat with Dataset"): # Renumbered (was 21)
    if gemini_key:
        st.subheader("Ask a question about your dataset")
        user_question = st.text_area("Your question:", height=100, placeholder="e.g., What are the top 5 countries with the most titles? or How many movies were released in 2020?", key="ai_chat_question") 

        if st.button("Ask AI 🤖", key="ai_chat_button"): 
            if user_question:
                try:
                    df_summary = f"""
                    Here's a summary of the dataset I'm working with:
                    Column Names: {df.columns.tolist()}
                    Data Types:\n{df.dtypes.to_string()}
                    First 5 Rows:\n{df.head().to_string()}
                    Basic Statistics (Head):\n{df.describe(include='all').head().to_string()} 
                    Total rows: {len(df)}"""

                    prompt = f"""You are a data analysis assistant. Based *only* on the following dataset summary, please answer the user's question. If the information is not present in the summary or cannot be inferred, please state that.
                    Dataset Summary (use only this information):\n{df_summary}\n\nUser's Question: {user_question}\n\nAnswer:"""
                    
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred while querying the AI. Ensure the API key is correct and the model 'gemini-1.5-flash-latest' is available: {e}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please enter your Gemini API key in the sidebar to use the AI Chat feature.")

# Tool 33: AI-Generated Content Summaries
with st.expander("✍️ Tool 33: AI-Generated Content Summaries"):
    if gemini_key:
        if 'title' in df.columns and 'type' in df.columns and 'listed_in' in df.columns:
            st.subheader("Generate a Netflix-Style Summary")
            available_titles = df['title'].dropna().unique().tolist()
            if available_titles:
                selected_title_for_summary = st.selectbox("Select a title from your dataset:", available_titles, key="ai_summary_title_select")
                
                if st.button("Generate Summary", key="ai_generate_summary_button"):
                    if selected_title_for_summary:
                        title_details = df[df['title'] == selected_title_for_summary].iloc[0]
                        content_type = title_details.get('type', 'N/A')
                        genres = title_details.get('listed_in', 'N/A')

                        prompt = f"""
                        Generate a short, engaging, and creative summary (around 2-3 sentences) suitable for a Netflix-style preview for the following content:
                        Title: {selected_title_for_summary}
                        Type: {content_type}
                        Genre(s): {genres}
                        Make it sound exciting and hint at the core themes or plot without giving away major spoilers.
                        """
                        try:
                            model = genai.GenerativeModel("gemini-1.5-flash-latest")
                            response = model.generate_content(prompt)
                            st.markdown("#### Generated Summary:")
                            st.success(response.text)
                        except Exception as e:
                            st.error(f"Error generating summary: {e}")
            else:
                st.info("No titles available in the dataset to generate summaries for.")
        else:
            st.info("Dataset must contain 'title', 'type', and 'listed_in' columns for this feature.")
    else:
        st.info("Enter Gemini API key in the sidebar to use AI-Generated Content Summaries.")

# Tool 34: AI-Generated Title Suggestions
with st.expander("💡 Tool 34: AI-Generated Title Suggestions"):
    if gemini_key:
        st.subheader("Get Creative Title Ideas")
        theme_keywords = st.text_input("Enter a theme, keywords, or a brief concept:", placeholder="e.g., space opera, romantic comedy in Paris, a detective solving ancient mysteries", key="ai_title_gen_theme")
        content_type_suggestion = st.selectbox("Select content type:", ["Movie", "TV Show", "Documentary", "Miniseries"], key="ai_title_gen_type")

        if st.button("Suggest Titles", key="ai_suggest_titles_button"):
            if theme_keywords:
                prompt = f"""
                Generate 5 creative and catchy title suggestions for a new {content_type_suggestion} based on the following theme/keywords/concept: '{theme_keywords}'.
                The titles should be suitable for a streaming platform like Netflix. For each suggestion, provide a brief (1-sentence) rationale or angle.
                Format each suggestion as:
                Title: [Generated Title]
                Rationale: [Brief Rationale]
                """
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(prompt)
                    st.markdown("#### Title Suggestions:")
                    st.markdown(response.text) # Gemini is good at following formatting
                except Exception as e:
                    st.error(f"Error generating title suggestions: {e}")
            else:
                st.warning("Please enter a theme or keywords.")
    else:
        st.info("Enter Gemini API key in the sidebar to use AI-Generated Title Suggestions.")

# Tool 35: AI-Driven Sentiment Analysis of Reviews
with st.expander("😊 Tool 35: AI-Driven Sentiment Analysis (Simulated Review)"):
    if gemini_key:
        st.subheader("Analyze Sentiment of a Hypothetical Review")
        review_text = st.text_area("Paste a review text here:", height=150, placeholder="e.g., 'This movie was absolutely fantastic, the acting was superb!' or 'Terrible plot, I was bored the whole time.'", key="ai_sentiment_review_text")
        if st.button("Analyze Sentiment", key="ai_analyze_sentiment_button"):
            if review_text:
                prompt = f"""Analyze the sentiment of the following review. Classify it as Positive, Negative, or Neutral. Also, provide a brief (1-sentence) explanation for your classification.
Review: "{review_text}"

Sentiment: [Positive/Negative/Neutral]
Explanation: ..."""
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(prompt)
                    st.markdown("#### Sentiment Analysis Result:")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")
            else:
                st.warning("Please enter a review text to analyze.")
    else:
        st.info("Enter Gemini API key in the sidebar to use AI-Driven Sentiment Analysis.")

st.header("🔍 Deeper Analytical Perspectives")

# Tool 36: Content Lifecycle Analysis
with st.expander("🔄 Tool 36: Content Lifecycle, Acquisition & Freshness Analysis"): 
    # Merging original Tool 36, Tool 44 (Content Acquisition Lag), and Tool 51 (Content Pacing & Freshness)
    required_cols_lifecycle = ['release_year', 'date_added', 'imdb_score', 'type'] # 'type' is needed for parts from old Tool 51
    if all(col in df.columns for col in required_cols_lifecycle):
        df_lifecycle = df.copy()
        df_lifecycle['date_added'] = pd.to_datetime(df_lifecycle['date_added'], errors='coerce')
        df_lifecycle.dropna(subset=['release_year', 'date_added', 'imdb_score'], inplace=True)

        if not df_lifecycle.empty:
            df_lifecycle['release_year'] = df_lifecycle['release_year'].astype(int)
            df_lifecycle['year_added'] = df_lifecycle['date_added'].dt.year
            
            df_lifecycle['age_at_addition'] = df_lifecycle['year_added'] - df_lifecycle['release_year']
            # Filter for non-negative age/lag, relevant for all combined analyses
            df_lifecycle = df_lifecycle[df_lifecycle['age_at_addition'] >= 0].copy()
            
            if not df_lifecycle.empty: # Check after age_at_addition filtering
                current_year = datetime.now().year
                df_lifecycle['years_on_platform'] = current_year - df_lifecycle['year_added']

                st.subheader("Distribution of Content Age When Added (Acquisition Lag)")
                fig_age_add_dist = px.histogram(df_lifecycle, x='age_at_addition', nbins=30,
                                           title="Content Age When Added to Netflix (Acquisition Lag)",
                                           labels={'age_at_addition': 'Age of Content When Added (Years)'},
                                           template="plotly_dark")
                st.plotly_chart(fig_age_add_dist, use_container_width=True)

                st.subheader("IMDb Score vs. Content Age at Addition (Acquisition Lag)")
                # Group by lag for clearer trend if many data points, or use raw scatter
                # For consistency with original Tool 44, let's bin if many points, else scatter
                if len(df_lifecycle['age_at_addition'].unique()) > 50: # Heuristic for "many points"
                    bins_lag = np.arange(0, df_lifecycle['age_at_addition'].max() + 5, 5)
                    if len(bins_lag) > 1 : # Ensure there's more than one bin edge
                        avg_score_by_binned_lag = df_lifecycle.groupby(pd.cut(df_lifecycle['age_at_addition'], bins=bins_lag))['imdb_score'].mean().reset_index()
                        avg_score_by_binned_lag['age_at_addition_mid'] = avg_score_by_binned_lag['age_at_addition'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else x)
                        x_col_lag_score = 'age_at_addition_mid'
                        data_for_lag_score_plot = avg_score_by_binned_lag
                        x_label_lag_score = 'Age at Addition (Years - Midpoint of Bin)'
                    else: # Not enough range for binning, use raw data
                        x_col_lag_score = 'age_at_addition'
                        data_for_lag_score_plot = df_lifecycle
                        x_label_lag_score = 'Age at Addition (Years)'
                else:
                    x_col_lag_score = 'age_at_addition'
                    data_for_lag_score_plot = df_lifecycle
                    x_label_lag_score = 'Age at Addition (Years)'

                fig_score_age_add = px.scatter(data_for_lag_score_plot, x=x_col_lag_score, y='imdb_score', trendline="ols",
                                             title="Average IMDb Score vs. Content Age at Addition",
                                             labels={x_col_lag_score: x_label_lag_score, 'imdb_score': 'Average IMDb Score'},
                                             template="plotly_dark")
                st.plotly_chart(fig_score_age_add, use_container_width=True)

                st.subheader("Performance vs. Years on Platform")
                avg_score_by_platform_time = df_lifecycle.groupby('years_on_platform')['imdb_score'].mean().reset_index()
                fig_score_platform_time = px.line(avg_score_by_platform_time, x='years_on_platform', y='imdb_score', markers=True,
                                                  title="Average IMDb Score vs. Years on Netflix Platform",
                                                  labels={'years_on_platform': 'Years on Platform', 'imdb_score': 'Average IMDb Score'},
                                                  template="plotly_dark")
                st.plotly_chart(fig_score_platform_time, use_container_width=True)

                # From original Tool 51: Content Pacing & Freshness Strategy Analysis
                st.subheader("Content Age at Addition: Movies vs. TV Shows")
                fig_pacing_type = px.box(df_lifecycle, x='type', y='age_at_addition', color='type',
                                         title="Freshness: Age of Content When Added (Movies vs. TV Shows)",
                                         labels={'age_at_addition': 'Age at Addition (Years)', 'type': 'Content Type'},
                                         template="plotly_dark")
                st.plotly_chart(fig_pacing_type, use_container_width=True)

                if 'listed_in' in df_lifecycle.columns:
                    st.subheader("Content Age at Addition by Top Genres")
                    pacing_genre_exploded = df_lifecycle.assign(genre=df_lifecycle['listed_in'].str.split(', ')).explode('genre')
                    pacing_genre_exploded['genre'] = pacing_genre_exploded['genre'].str.strip()
                    
                    top_n_genres_pacing = st.slider("Number of Top Genres for Freshness by Genre Analysis:", 3, 10, 5, key="lifecycle_freshness_genre_slider_tool36")
                    common_genres = pacing_genre_exploded['genre'].value_counts().nlargest(top_n_genres_pacing).index.tolist()
                    
                    pacing_top_genres_df = pacing_genre_exploded[pacing_genre_exploded['genre'].isin(common_genres)]

                    if not pacing_top_genres_df.empty:
                        fig_pacing_genre = px.box(pacing_top_genres_df, x='genre', y='age_at_addition', color='genre',
                                                  title=f"Freshness: Age of Content When Added (Top {top_n_genres_pacing} Genres)",
                                                  labels={'age_at_addition': 'Age at Addition (Years)', 'genre': 'Genre'},
                                                  template="plotly_dark")
                        st.plotly_chart(fig_pacing_genre, use_container_width=True)
                    else:
                        st.info("Not enough data for selected top genres to analyze freshness by genre.")
            else:
                st.info("No valid data for lifecycle analysis after filtering for non-negative age at addition.")
        else:
            st.info("Not enough valid data for lifecycle analysis after cleaning 'date_added', 'release_year', and 'imdb_score'.")
    else:
        st.info(f"Required columns ({', '.join(required_cols_lifecycle)}) not available for Content Lifecycle, Acquisition & Freshness Analysis.")

# Tool 37: "Hidden Gems" Detector
with st.expander("💎 Tool 37: 'Hidden Gems' Detector"): # Renumbered (was 34, originally 37)
    if 'imdb_score' in df.columns and ('views_millions' in df.columns or 'budget_millions' in df.columns) and 'title' in df.columns:
        df_gems = df.copy()
        df_gems['imdb_score'] = pd.to_numeric(df_gems['imdb_score'], errors='coerce')
        
        performance_metric_col = None
        performance_label = ""
        if 'views_millions' in df_gems.columns:
            df_gems['views_millions'] = pd.to_numeric(df_gems['views_millions'], errors='coerce')
            if not df_gems['views_millions'].isnull().all():
                performance_metric_col = 'views_millions'
                performance_label = 'Views (Millions)'
        
        if performance_metric_col is None and 'budget_millions' in df_gems.columns: # Fallback to budget if views not usable
            df_gems['budget_millions'] = pd.to_numeric(df_gems['budget_millions'], errors='coerce')
            if not df_gems['budget_millions'].isnull().all():
                performance_metric_col = 'budget_millions'
                performance_label = 'Budget (Millions) - Proxy for Popularity/Exposure'

        if performance_metric_col:
            df_gems.dropna(subset=['imdb_score', performance_metric_col, 'title'], inplace=True)

            if not df_gems.empty:
                st.subheader("Define 'Hidden Gem' Criteria")
                col_gem1, col_gem2 = st.columns(2)
                min_imdb = col_gem1.slider("Minimum IMDb Score for a Gem:", 5.0, 9.5, 7.5, 0.1, key="gem_min_imdb_tool37") 
                
                if performance_metric_col == 'views_millions':
                    max_performance = col_gem2.slider(f"Maximum {performance_label} for a Gem:", 
                                                      float(df_gems[performance_metric_col].min()), 
                                                      float(df_gems[performance_metric_col].quantile(0.75)), # Avoid extreme max
                                                      float(df_gems[performance_metric_col].quantile(0.25)), # Default to lower quartile
                                                      key="gem_max_perf_tool37") 
                else: # budget_millions
                     max_performance = col_gem2.slider(f"Maximum {performance_label} for a Gem:", 
                                                      float(df_gems[performance_metric_col].min()), 
                                                      float(df_gems[performance_metric_col].quantile(0.75)), 
                                                      float(df_gems[performance_metric_col].quantile(0.5)), # Default to median for budget
                                                      key="gem_max_perf_budget_tool37") 

                hidden_gems_df = df_gems[
                    (df_gems['imdb_score'] >= min_imdb) & 
                    (df_gems[performance_metric_col] <= max_performance)
                ]

                fig_gems = px.scatter(df_gems, x=performance_metric_col, y='imdb_score', color='type',
                                      title=f"IMDb Score vs. {performance_label}",
                                      labels={'imdb_score': 'IMDb Score', performance_metric_col: performance_label},
                                      template="plotly_dark", hover_data=['title'])
                
                # Highlight hidden gems
                if not hidden_gems_df.empty:
                    fig_gems.add_trace(go.Scatter(x=hidden_gems_df[performance_metric_col], y=hidden_gems_df['imdb_score'],
                                                  mode='markers', marker=dict(size=12, color='gold', symbol='star'),
                                                  name='Hidden Gem', hoverinfo='skip')) # Use hover_data from main scatter

                st.plotly_chart(fig_gems, use_container_width=True)

                if not hidden_gems_df.empty:
                    st.subheader(f"Identified Hidden Gems ({len(hidden_gems_df)} titles)")
                    st.dataframe(hidden_gems_df[['title', 'type', 'imdb_score', performance_metric_col, 'release_year']].sort_values(by='imdb_score', ascending=False))
                else:
                    st.info("No titles match the current 'Hidden Gem' criteria.")
            else:
                st.info(f"Not enough valid data for 'imdb_score' and '{performance_label}' to detect hidden gems.")
        else:
            st.info("A usable performance metric ('views_millions' or 'budget_millions') is not available or has no valid data.")
    else:
        st.info("Required columns ('imdb_score', 'title', and 'views_millions' or 'budget_millions') not available for Hidden Gems Detector.")

# Tool 38: Genre Popularity vs. Saturation Matrix
with st.expander("🎯 Tool 38: Genre Popularity vs. Saturation Matrix"): # Renumbered (was 35, originally 38)
    if 'listed_in' in df.columns and 'imdb_score' in df.columns:
        df_genre_matrix = df.copy()
        df_genre_matrix['imdb_score'] = pd.to_numeric(df_genre_matrix['imdb_score'], errors='coerce')
        df_genre_matrix.dropna(subset=['listed_in', 'imdb_score'], inplace=True)

        if not df_genre_matrix.empty:
            # Explode genres
            genre_exploded_df = df_genre_matrix.assign(genre=df_genre_matrix['listed_in'].str.split(', ')).explode('genre')
            genre_exploded_df['genre'] = genre_exploded_df['genre'].str.strip()
            genre_exploded_df = genre_exploded_df[genre_exploded_df['genre'] != '']

            if not genre_exploded_df.empty:
                genre_stats = genre_exploded_df.groupby('genre').agg(
                    saturation_count=('title', 'count'),
                    avg_imdb_score=('imdb_score', 'mean'),
                    avg_budget_millions=('budget_millions', 'mean') if 'budget_millions' in genre_exploded_df.columns else pd.NamedAgg(column='title', aggfunc=lambda x: 0) # Placeholder if no budget
                ).reset_index()
                genre_stats.dropna(subset=['avg_imdb_score'], inplace=True) # Ensure genres have scores
                genre_stats = genre_stats[genre_stats['saturation_count'] > 1] # Filter for genres with more than 1 title for meaningful stats

                if not genre_stats.empty:
                    # Cap bubble size for better visualization if budget is used
                    size_col = 'avg_budget_millions' if 'budget_millions' in genre_exploded_df.columns and genre_stats['avg_budget_millions'].sum() > 0 else None
                    if size_col:
                         # Normalize budget for size or use a sensible cap
                        max_budget_for_size = genre_stats[size_col].quantile(0.95) # Cap at 95th percentile to avoid outliers dominating
                        genre_stats['bubble_size'] = genre_stats[size_col].clip(upper=max_budget_for_size) * 0.5 # Scale factor
                        size_col_for_plot = 'bubble_size'
                        hover_data_cols = ['genre', 'saturation_count', 'avg_imdb_score', 'avg_budget_millions']
                    else:
                        size_col_for_plot = None # No bubble size if no budget
                        hover_data_cols = ['genre', 'saturation_count', 'avg_imdb_score']


                    fig_genre_matrix = px.scatter(genre_stats, x='saturation_count', y='avg_imdb_score',
                                                  size=size_col_for_plot, color='genre', 
                                                  hover_name='genre', hover_data=hover_data_cols,
                                                  title="Genre Popularity (Avg. IMDb) vs. Saturation (Title Count)",
                                                  labels={'saturation_count': 'Saturation (Number of Titles)', 
                                                          'avg_imdb_score': 'Popularity (Average IMDb Score)'},
                                                  size_max=60 if size_col_for_plot else 15, # Adjust size_max
                                                  template="plotly_dark")
                    st.plotly_chart(fig_genre_matrix, use_container_width=True)
                    st.caption("Bubble size represents average budget (if available and applicable). Larger bubbles indicate higher average budgets for titles in that genre.")
                else:
                    st.info("Not enough genre data (after filtering for multiple titles per genre) to create the matrix.")
            else:
                st.info("No valid genres found after exploding the 'listed_in' column.")
        else:
            st.info("Not enough valid data for 'listed_in' and 'imdb_score' to create the genre matrix.")
    else:
        st.info("Required columns ('listed_in', 'imdb_score') not available for Genre Popularity vs. Saturation Matrix.")

# Tool 39: N-gram Analysis on Titles
with st.expander("🔑 Tool 39: N-gram Analysis on Titles"): # Renumbered (was 36, originally 39)
    if 'title' in df.columns:
        from sklearn.feature_extraction.text import CountVectorizer

        df_ngram = df.copy()
        df_ngram.dropna(subset=['title'], inplace=True)
        titles_corpus = df_ngram['title'].astype(str).str.lower().tolist()

        if titles_corpus:
            st.subheader("Common Phrases in Content Titles")

            def get_top_n_grams(corpus, ngram_range, n=10):
                try:
                    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
                    bag_of_words = vec.transform(corpus)
                    sum_words = bag_of_words.sum(axis=0)
                    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                    return words_freq[:n]
                except ValueError: # Happens if corpus is empty after stop words or too short
                    return []

            col_ngram1, col_ngram2 = st.columns(2)
            with col_ngram1:
                st.markdown("#### Top Bi-grams (2-word phrases)")
                top_bigrams = get_top_n_grams(titles_corpus, ngram_range=(2,2), n=10)
                if top_bigrams:
                    bigram_df = pd.DataFrame(top_bigrams, columns=['Bi-gram', 'Frequency'])
                    fig_bigram = px.bar(bigram_df, x='Frequency', y='Bi-gram', orientation='h', template="plotly_dark", title="Top 10 Bi-grams in Titles")
                    fig_bigram.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bigram, use_container_width=True)
                else:
                    st.write("Not enough data to extract bi-grams or no common bi-grams found.")

            with col_ngram2:
                st.markdown("#### Top Tri-grams (3-word phrases)")
                top_trigrams = get_top_n_grams(titles_corpus, ngram_range=(3,3), n=10)
                if top_trigrams:
                    trigram_df = pd.DataFrame(top_trigrams, columns=['Tri-gram', 'Frequency'])
                    fig_trigram = px.bar(trigram_df, x='Frequency', y='Tri-gram', orientation='h', template="plotly_dark", title="Top 10 Tri-grams in Titles")
                    fig_trigram.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_trigram, use_container_width=True)
                else:
                    st.write("Not enough data to extract tri-grams or no common tri-grams found.")
        else:
            st.info("No titles available for N-gram analysis.")
    else:
        st.info("'title' column not available for N-gram Analysis.")

# Tool 40: User Persona-Based Recommendations (Simulated)
with st.expander("👤 Tool 40: User Persona-Based Recommendations (Simulated)"): # Renumbered (was 37, originally 40)
    if 'type' in df.columns and 'listed_in' in df.columns and 'imdb_score' in df.columns and 'rating' in df.columns:
        df_persona = df.copy()
        df_persona['imdb_score'] = pd.to_numeric(df_persona['imdb_score'], errors='coerce')
        df_persona.dropna(subset=['type', 'listed_in', 'imdb_score', 'rating', 'title'], inplace=True)

        personas = {
            "Action Enthusiast": {
                "genres": ["Action & Adventure", "Sci-Fi & Fantasy", "Thrillers"],
                "min_imdb": 6.5,
                "ratings": ["PG-13", "R", "TV-14", "TV-MA"],
                "type": None # Any type
            },
            "Documentary Lover": {
                "genres": ["Documentaries", "Science & Nature TV", "Historical Documentaries"], # Add more docu genres if in data
                "min_imdb": 7.0,
                "ratings": ["TV-PG", "TV-14", "PG", "G"],
                "type": None 
            },
            "Family Movie Night": {
                "genres": ["Kids' TV", "Comedies", "Family Movies", "Animation"], # Add more family genres
                "min_imdb": 6.0,
                "ratings": ["G", "PG", "TV-Y", "TV-Y7", "TV-G"],
                "type": "Movie"
            },
            "Critically Acclaimed Seeker": {
                "genres": ["Dramas", "Independent Movies", "International Movies", "Classic Movies"],
                "min_imdb": 8.0,
                "ratings": ["R", "PG-13", "TV-MA", "TV-14"],
                "type": None
            }
        }

        selected_persona_name = st.selectbox("Select a User Persona:", list(personas.keys()), key="persona_select_tool40") 
        
        if selected_persona_name and not df_persona.empty:
            persona_criteria = personas[selected_persona_name]
            st.markdown(f"#### Recommendations for: {selected_persona_name}")
            st.caption(f"Prefers: Genres like {', '.join(persona_criteria['genres'][:3])}..., IMDb >= {persona_criteria['min_imdb']}, Ratings like {', '.join(persona_criteria['ratings'][:2])}...")

            filtered_df = df_persona[
                df_persona['listed_in'].apply(lambda x: any(g in x for g in persona_criteria['genres'])) &
                (df_persona['imdb_score'] >= persona_criteria['min_imdb']) &
                (df_persona['rating'].isin(persona_criteria['ratings']))
            ]

            if persona_criteria['type']:
                filtered_df = filtered_df[filtered_df['type'] == persona_criteria['type']]
            
            if not filtered_df.empty:
                recommendations = filtered_df.nlargest(10, 'imdb_score')
                st.dataframe(recommendations[['title', 'type', 'imdb_score', 'release_year', 'listed_in', 'rating']])
            else:
                st.info(f"No content found matching the '{selected_persona_name}' persona criteria with the current dataset.")
        elif df_persona.empty:
            st.info("Not enough data to provide persona-based recommendations after cleaning.")
    else:
        st.info("Required columns ('type', 'listed_in', 'imdb_score', 'rating', 'title') not available for Persona-Based Recommendations.")

# Tool 41: Award Impact Analysis
with st.expander("🏆 Tool 41: Award Impact Analysis"): # Renumbered (was 38, originally 41)
    required_cols_awards = ['title', 'imdb_score']
    optional_cols_awards = ['awards_won', 'nomination_for_best_picture', 'views_millions']
    
    if all(col in df.columns for col in required_cols_awards):
        df_awards = df.copy()
        df_awards['imdb_score'] = pd.to_numeric(df_awards['imdb_score'], errors='coerce')
        
        if 'views_millions' in df_awards.columns:
            df_awards['views_millions'] = pd.to_numeric(df_awards['views_millions'], errors='coerce')
        if 'awards_won' in df_awards.columns:
            df_awards['awards_won'] = pd.to_numeric(df_awards['awards_won'], errors='coerce')
        if 'nomination_for_best_picture' in df_awards.columns:
            df_awards['nomination_for_best_picture'] = pd.to_numeric(df_awards['nomination_for_best_picture'], errors='coerce').fillna(0).astype(int)

        df_awards.dropna(subset=['imdb_score', 'title'], inplace=True)

        if not df_awards.empty:
            st.subheader("Impact of 'Best Picture' Nomination")
            if 'nomination_for_best_picture' in df_awards.columns:
                best_pic_analysis = df_awards.groupby('nomination_for_best_picture').agg(
                    avg_imdb_score=('imdb_score', 'mean'),
                    avg_views_millions=('views_millions', 'mean') if 'views_millions' in df_awards.columns else pd.NamedAgg(column='title', aggfunc=lambda x: np.nan)
                ).reset_index()
                best_pic_analysis['nomination_for_best_picture'] = best_pic_analysis['nomination_for_best_picture'].map({0: 'Not Nominated', 1: 'Nominated'})
                
                col_bp1, col_bp2 = st.columns(2)
                with col_bp1:
                    fig_bp_imdb = px.bar(best_pic_analysis, x='nomination_for_best_picture', y='avg_imdb_score',
                                         title="Avg. IMDb: Best Picture Nominated vs. Not",
                                         labels={'nomination_for_best_picture': '', 'avg_imdb_score': 'Avg. IMDb Score'},
                                         color='nomination_for_best_picture', template="plotly_dark")
                    st.plotly_chart(fig_bp_imdb, use_container_width=True)
                if 'views_millions' in df_awards.columns and not best_pic_analysis['avg_views_millions'].isnull().all():
                    with col_bp2:
                        fig_bp_views = px.bar(best_pic_analysis, x='nomination_for_best_picture', y='avg_views_millions',
                                             title="Avg. Views: Best Picture Nominated vs. Not",
                                             labels={'nomination_for_best_picture': '', 'avg_views_millions': 'Avg. Views (Millions)'},
                                             color='nomination_for_best_picture', template="plotly_dark")
                        st.plotly_chart(fig_bp_views, use_container_width=True)
            else:
                st.info("'nomination_for_best_picture' column not available for this part of the analysis.")

            if 'awards_won' in df_awards.columns:
                df_awards_filtered = df_awards.dropna(subset=['awards_won'])
                if not df_awards_filtered.empty:
                    st.subheader("Number of Awards Won vs. IMDb Score")
                    fig_awards_imdb = px.scatter(df_awards_filtered, x='awards_won', y='imdb_score', trendline="ols",
                                                 title="Awards Won vs. IMDb Score",
                                                 labels={'awards_won': 'Number of Awards Won', 'imdb_score': 'IMDb Score'},
                                                 template="plotly_dark", hover_data=['title'])
                    st.plotly_chart(fig_awards_imdb, use_container_width=True)

                    st.subheader("Top Titles by Awards Won")
                    top_award_titles = df_awards_filtered.nlargest(10, 'awards_won')[['title', 'awards_won', 'imdb_score']]
                    st.dataframe(top_award_titles)

                    # IMDb Score by Award Brackets
                    bins = [-1, 0, 5, 10, 20, df_awards_filtered['awards_won'].max()] # Define award brackets
                    labels = ['0 Awards', '1-5 Awards', '6-10 Awards', '11-20 Awards', '20+ Awards']
                    df_awards_filtered['award_bracket'] = pd.cut(df_awards_filtered['awards_won'], bins=bins, labels=labels, right=True)
                    avg_imdb_by_bracket = df_awards_filtered.groupby('award_bracket')['imdb_score'].mean().reset_index()
                    
                    fig_bracket_imdb = px.bar(avg_imdb_by_bracket, x='award_bracket', y='imdb_score',
                                              title="Average IMDb Score by Award Count Bracket",
                                              labels={'award_bracket': 'Award Bracket', 'imdb_score': 'Average IMDb Score'},
                                              template="plotly_dark")
                    st.plotly_chart(fig_bracket_imdb, use_container_width=True)
                else:
                    st.info("Not enough data with 'awards_won' to analyze.")
            else:
                st.info("'awards_won' column not available for detailed award analysis.")
        else:
            st.info("Not enough valid data for Award Impact Analysis after cleaning.")
    else:
        st.info(f"Required columns ({', '.join(required_cols_awards)}) not available for Award Impact Analysis.")

# Tool 42: Content Language Diversity & Performance
with st.expander("🌐 Tool 42: Content Language Diversity & Performance"): # Renumbered (was 39, originally 42)
    if 'language' in df.columns and 'imdb_score' in df.columns:
        df_lang = df.copy()
        df_lang['imdb_score'] = pd.to_numeric(df_lang['imdb_score'], errors='coerce')
        if 'views_millions' in df_lang.columns:
            df_lang['views_millions'] = pd.to_numeric(df_lang['views_millions'], errors='coerce')
        
        df_lang.dropna(subset=['language', 'imdb_score'], inplace=True)

        if not df_lang.empty:
            st.subheader("Content Distribution by Language")
            top_n_langs = st.slider("Number of Top Languages to Display:", 3, 10, 5, key="lang_top_n_tool42") 
            lang_counts = df_lang['language'].value_counts().nlargest(top_n_langs)
            
            fig_lang_dist = px.pie(lang_counts, values=lang_counts.values, names=lang_counts.index,
                                   title=f"Top {top_n_langs} Languages by Content Count", template="plotly_dark")
            st.plotly_chart(fig_lang_dist, use_container_width=True)

            df_top_langs = df_lang[df_lang['language'].isin(lang_counts.index)]

            st.subheader(f"Performance Metrics for Top {top_n_langs} Languages")
            avg_imdb_by_lang = df_top_langs.groupby('language')['imdb_score'].mean().reset_index().sort_values(by='imdb_score', ascending=False)
            fig_lang_imdb = px.bar(avg_imdb_by_lang, x='language', y='imdb_score', color='language',
                                   title="Average IMDb Score by Language", template="plotly_dark")
            st.plotly_chart(fig_lang_imdb, use_container_width=True)

            if 'views_millions' in df_top_langs.columns and not df_top_langs['views_millions'].isnull().all():
                avg_views_by_lang = df_top_langs.groupby('language')['views_millions'].mean().reset_index().sort_values(by='views_millions', ascending=False)
                fig_lang_views = px.bar(avg_views_by_lang, x='language', y='views_millions', color='language',
                                       title="Average Views (Millions) by Language", template="plotly_dark")
                st.plotly_chart(fig_lang_views, use_container_width=True)
            
            if 'type' in df_top_langs.columns:
                st.subheader(f"Content Type Distribution within Top {top_n_langs} Languages")
                type_by_lang = df_top_langs.groupby(['language', 'type']).size().reset_index(name='count')
                fig_lang_type = px.bar(type_by_lang, x='language', y='count', color='type', barmode='group',
                                       title="Movie vs. TV Show Count by Language", template="plotly_dark")
                st.plotly_chart(fig_lang_type, use_container_width=True)
        else:
            st.info("Not enough valid data for Language Diversity analysis after cleaning.")
    else:
        st.info("Required columns ('language', 'imdb_score') not available for Language Diversity & Performance Analysis.")

# Tool 43: Director & Actor Genre Affinity (was Tool 45)
with st.expander("🎨 Tool 43: Director & Actor Genre Affinity"): 
    if 'director' in df.columns and 'cast' in df.columns and 'listed_in' in df.columns:
        df_affinity = df.copy()
        df_affinity.dropna(subset=['director', 'cast', 'listed_in'], inplace=True)

        # Explode directors, actors, and genres
        directors_exploded = df_affinity.assign(person=df_affinity['director'].str.split(', ')).explode('person')
        actors_exploded = df_affinity.assign(person=df_affinity['cast'].str.split(', ')).explode('person')
        
        # Combine and get top people
        all_people = pd.concat([directors_exploded['person'], actors_exploded['person']]).value_counts().nlargest(20).index.tolist()
        
        selected_person = st.selectbox("Select a Director or Actor:", all_people, key="person_genre_affinity_tool43") 

        if selected_person:
            person_df = pd.concat([
                directors_exploded[directors_exploded['person'] == selected_person],
                actors_exploded[actors_exploded['person'] == selected_person]
            ]).drop_duplicates(subset=['show_id']) # Avoid double counting if person is director and actor in same title

            if not person_df.empty:
                person_genres_exploded = person_df.assign(genre=person_df['listed_in'].str.split(', ')).explode('genre')
                person_genres_exploded['genre'] = person_genres_exploded['genre'].str.strip()
                genre_counts_person = person_genres_exploded['genre'].value_counts()

                fig_person_genre = px.bar(genre_counts_person, x=genre_counts_person.index, y=genre_counts_person.values,
                                          title=f"Genre Affinity for {selected_person}",
                                          labels={'index': 'Genre', 'y': 'Number of Titles'},
                                          template="plotly_dark")
                st.plotly_chart(fig_person_genre, use_container_width=True)
            else:
                st.info(f"No titles found for {selected_person} to analyze genre affinity.")
    else:
        st.info("Required columns ('director', 'cast', 'listed_in') not available for this analysis.")

# Tool 44: Content Attributes & Technical Details vs. Performance (was Tool 46, absorbs Tool 43)
with st.expander("📝 Tool 44: Content Attributes & Technical Details vs. Performance"):
    # Combines original Tool 46 (Title Characteristics) and Tool 43 (Technical Aspects)
    if 'title' in df.columns and 'imdb_score' in df.columns:
        df_title_char = df.copy()
        df_title_char['imdb_score'] = pd.to_numeric(df_title_char['imdb_score'], errors='coerce')
        df_title_char.dropna(subset=['title', 'imdb_score'], inplace=True)
        
        df_title_char['title_word_count'] = df_title_char['title'].astype(str).apply(lambda x: len(x.split()))
        
        st.subheader("Title Length (Word Count) vs. IMDb Score")
        fig_title_len_score = px.scatter(df_title_char, x='title_word_count', y='imdb_score', trendline="ols",
                                         title="Title Word Count vs. IMDb Score",
                                         labels={'title_word_count': 'Number of Words in Title', 'imdb_score': 'IMDb Score'},
                                         template="plotly_dark")
        st.plotly_chart(fig_title_len_score, use_container_width=True)

        st.subheader("Impact of Numbers in Title on IMDb Score")
        df_title_char['has_number_in_title'] = df_title_char['title'].astype(str).str.contains(r'\d').astype(int)
        avg_score_by_number = df_title_char.groupby('has_number_in_title')['imdb_score'].mean().reset_index()
        avg_score_by_number['has_number_in_title'] = avg_score_by_number['has_number_in_title'].map({0: 'No Number', 1: 'Has Number'})
        fig_title_num_score = px.bar(avg_score_by_number, x='has_number_in_title', y='imdb_score', color='has_number_in_title',
                                     title="Avg. IMDb Score: Titles With vs. Without Numbers", template="plotly_dark")
        st.plotly_chart(fig_title_num_score, use_container_width=True)

        # From original Tool 43: Technical Aspects Analysis
        if 'aspect_ratio' in df_title_char.columns and 'sound_mix' in df_title_char.columns:
            df_tech_merged = df_title_char.dropna(subset=['aspect_ratio', 'sound_mix', 'imdb_score']) # Use df_title_char which already handled imdb_score
            if not df_tech_merged.empty:
                st.markdown("---") # Separator
                st.subheader("Technical Aspects Analysis")
                
                st.markdown("#### Aspect Ratio Analysis")
                col_ar1, col_ar2 = st.columns(2)
                with col_ar1:
                    ar_counts = df_tech_merged['aspect_ratio'].value_counts()
                    fig_ar_dist = px.pie(ar_counts, values=ar_counts.values, names=ar_counts.index,
                                         title="Content Distribution by Aspect Ratio", template="plotly_dark")
                    st.plotly_chart(fig_ar_dist, use_container_width=True)
                with col_ar2:
                    avg_score_by_ar = df_tech_merged.groupby('aspect_ratio')['imdb_score'].mean().reset_index().sort_values(by='imdb_score', ascending=False)
                    fig_ar_score = px.bar(avg_score_by_ar, x='aspect_ratio', y='imdb_score', color='aspect_ratio',
                                          title="Average IMDb Score by Aspect Ratio", template="plotly_dark")
                    st.plotly_chart(fig_ar_score, use_container_width=True)

                st.markdown("#### Sound Mix Analysis")
                col_sm1, col_sm2 = st.columns(2)
                with col_sm1:
                    sm_counts = df_tech_merged['sound_mix'].value_counts()
                    fig_sm_dist = px.pie(sm_counts, values=sm_counts.values, names=sm_counts.index,
                                         title="Content Distribution by Sound Mix", template="plotly_dark")
                    st.plotly_chart(fig_sm_dist, use_container_width=True)
                with col_sm2:
                    avg_score_by_sm = df_tech_merged.groupby('sound_mix')['imdb_score'].mean().reset_index().sort_values(by='imdb_score', ascending=False)
                    fig_sm_score = px.bar(avg_score_by_sm, x='sound_mix', y='imdb_score', color='sound_mix',
                                          title="Average IMDb Score by Sound Mix", template="plotly_dark")
                    st.plotly_chart(fig_sm_score, use_container_width=True)
            else:
                st.info("Not enough valid data for Technical Aspects (Aspect Ratio/Sound Mix) part of this analysis after cleaning.")
        else:
            st.info("Aspect Ratio and/or Sound Mix columns not available for the technical aspects part of this analysis.")
    else:
        st.info("Required columns ('title', 'imdb_score') not available for this analysis.")

# Tool 45: Simulated Franchise/Sequel Analysis (was Tool 47)
with st.expander("🔗 Tool 45: Simulated Franchise/Sequel Analysis"):
    if 'title' in df.columns and 'imdb_score' in df.columns:
        df_franchise = df.copy()
        df_franchise['imdb_score'] = pd.to_numeric(df_franchise['imdb_score'], errors='coerce')
        df_franchise.dropna(subset=['title', 'imdb_score'], inplace=True)

        # Regex patterns to identify potential sequels/franchises
        # This is a heuristic and might need refinement
        franchise_patterns = r'(?i)(\bPart\s*\d+\b|\b\d+\s*of\s*\d+\b|:\s*\w+|\b\d{1,2}\b$|\bII\b|\bIII\b|\bIV\b|\bV\b)'
        df_franchise['is_potential_franchise'] = df_franchise['title'].astype(str).str.contains(franchise_patterns).astype(int)

        st.subheader("Performance: Potential Franchise Titles vs. Standalone")
        avg_score_by_franchise = df_franchise.groupby('is_potential_franchise')['imdb_score'].agg(['mean', 'count']).reset_index()
        avg_score_by_franchise['is_potential_franchise'] = avg_score_by_franchise['is_potential_franchise'].map({0: 'Likely Standalone', 1: 'Potential Franchise/Sequel'})
        
        fig_franchise_score = px.bar(avg_score_by_franchise, x='is_potential_franchise', y='mean', color='is_potential_franchise',
                                     title="Avg. IMDb Score: Potential Franchise vs. Standalone",
                                     labels={'mean': 'Average IMDb Score', 'is_potential_franchise': ''},
                                     text='count', template="plotly_dark")
        fig_franchise_score.update_traces(texttemplate='%{text} titles', textposition='outside')
        st.plotly_chart(fig_franchise_score, use_container_width=True)

        st.caption("Note: Franchise identification is based on simple title patterns and is a simulation.")
        st.write("Examples of titles identified as 'Potential Franchise/Sequel':")
        st.dataframe(df_franchise[df_franchise['is_potential_franchise'] == 1][['title', 'imdb_score']].head())
    else:
        st.info("Required columns ('title', 'imdb_score') not available for this analysis.")

# Tool 46: Genre Performance - Movies vs. TV Shows (was Tool 48)
with st.expander("🎬🆚📺 Tool 46: Genre Performance - Movies vs. TV Shows"):
    if 'listed_in' in df.columns and 'type' in df.columns and 'imdb_score' in df.columns:
        df_genre_type = df.copy()
        df_genre_type['imdb_score'] = pd.to_numeric(df_genre_type['imdb_score'], errors='coerce')
        df_genre_type.dropna(subset=['listed_in', 'type', 'imdb_score'], inplace=True)

        genre_type_exploded = df_genre_type.assign(genre=df_genre_type['listed_in'].str.split(', ')).explode('genre')
        genre_type_exploded['genre'] = genre_type_exploded['genre'].str.strip()
        
        top_genres_for_comp = genre_type_exploded['genre'].value_counts().nlargest(10).index.tolist()
        selected_genre_comp = st.selectbox("Select Genre for Movie vs. TV Show Comparison:", top_genres_for_comp, key="genre_type_comp_select_tool46") 

        if selected_genre_comp:
            genre_specific_df = genre_type_exploded[genre_type_exploded['genre'] == selected_genre_comp]
            
            if not genre_specific_df.empty:
                stats_by_type = genre_specific_df.groupby('type').agg(
                    avg_imdb_score=('imdb_score', 'mean'),
                    title_count=('title', 'count')
                ).reset_index()

                st.subheader(f"Comparison for '{selected_genre_comp}'")
                col_gt1, col_gt2 = st.columns(2)
                with col_gt1:
                    fig_gt_score = px.bar(stats_by_type, x='type', y='avg_imdb_score', color='type',
                                          title=f"Avg. IMDb Score in {selected_genre_comp}", template="plotly_dark")
                    st.plotly_chart(fig_gt_score, use_container_width=True)
                with col_gt2:
                    fig_gt_count = px.bar(stats_by_type, x='type', y='title_count', color='type',
                                          title=f"Number of Titles in {selected_genre_comp}", template="plotly_dark")
                    st.plotly_chart(fig_gt_count, use_container_width=True)
            else:
                st.info(f"No data found for genre '{selected_genre_comp}' to compare Movies vs. TV Shows.")
    else:
        st.info("Required columns ('listed_in', 'type', 'imdb_score') not available for this analysis.")

# Tool 47: Decade-wise Genre Evolution & Dominance (was Tool 49)
with st.expander("📈 Tool 47: Decade-wise Genre Evolution & Dominance"):
    if 'release_year' in df.columns and 'listed_in' in df.columns:
        df_decade_genre = df.copy()
        df_decade_genre['release_year'] = pd.to_numeric(df_decade_genre['release_year'], errors='coerce')
        df_decade_genre.dropna(subset=['release_year', 'listed_in'], inplace=True)
        
        df_decade_genre['decade'] = (df_decade_genre['release_year'] // 10) * 10
        
        genre_decade_exploded = df_decade_genre.assign(genre=df_decade_genre['listed_in'].str.split(', ')).explode('genre')
        genre_decade_exploded['genre'] = genre_decade_exploded['genre'].str.strip()
        
        # Focus on top N overall genres for clarity
        top_n_overall_genres = genre_decade_exploded['genre'].value_counts().nlargest(7).index.tolist()
        genre_decade_filtered = genre_decade_exploded[genre_decade_exploded['genre'].isin(top_n_overall_genres)]
        
        genre_counts_by_decade = genre_decade_filtered.groupby(['decade', 'genre']).size().reset_index(name='count')
        
        if not genre_counts_by_decade.empty:
            fig_decade_genre = px.area(genre_counts_by_decade, x='decade', y='count', color='genre',
                                       title="Evolution of Top Genre Popularity by Decade",
                                       labels={'decade': 'Decade', 'count': 'Number of Titles'},
                                       template="plotly_dark")
            st.plotly_chart(fig_decade_genre, use_container_width=True)
        else:
            st.info("Not enough data to analyze genre evolution by decade for top genres.")
    else:
        st.info("Required columns ('release_year', 'listed_in') not available for this analysis.")

# Tool 48: Budget Efficiency Tiers & ROI Analysis (was Tool 50)
with st.expander("💸 Tool 48: Budget Efficiency Tiers & ROI Analysis"):
    if 'budget_millions' in df.columns and 'views_millions' in df.columns and 'imdb_score' in df.columns:
        df_budget_roi = df.copy()
        df_budget_roi['budget_millions'] = pd.to_numeric(df_budget_roi['budget_millions'], errors='coerce')
        df_budget_roi['views_millions'] = pd.to_numeric(df_budget_roi['views_millions'], errors='coerce')
        df_budget_roi['imdb_score'] = pd.to_numeric(df_budget_roi['imdb_score'], errors='coerce')
        df_budget_roi.dropna(subset=['budget_millions', 'views_millions', 'imdb_score'], inplace=True)

        if not df_budget_roi.empty and df_budget_roi['budget_millions'].max() > 0:
            df_budget_roi['roi'] = np.where(df_budget_roi['budget_millions'] > 0.01, df_budget_roi['views_millions'] / df_budget_roi['budget_millions'], np.nan)
            df_budget_roi.dropna(subset=['roi'], inplace=True)

            if not df_budget_roi.empty:
                # Define budget tiers
                # Using quantiles for dynamic tier definition
                low_budget_threshold = df_budget_roi['budget_millions'].quantile(0.33)
                mid_budget_threshold = df_budget_roi['budget_millions'].quantile(0.66)
                
                bins = [0, low_budget_threshold, mid_budget_threshold, df_budget_roi['budget_millions'].max() + 1]
                labels = ['Low Budget', 'Medium Budget', 'High Budget']
                df_budget_roi['budget_tier'] = pd.cut(df_budget_roi['budget_millions'], bins=bins, labels=labels, right=False)
                df_budget_roi.dropna(subset=['budget_tier'], inplace=True) # Drop if any budget didn't fall into a tier

                if not df_budget_roi.empty:
                    st.subheader("ROI Distribution by Budget Tier")
                    fig_roi_tier = px.box(df_budget_roi, x='budget_tier', y='roi', color='budget_tier',
                                          title="Return on Investment (ROI) by Budget Tier",
                                          labels={'budget_tier': 'Budget Tier', 'roi': 'ROI (Views/Budget)'},
                                          template="plotly_dark")
                    st.plotly_chart(fig_roi_tier, use_container_width=True)

                    st.subheader("Average IMDb Score by Budget Tier")
                    avg_imdb_by_tier = df_budget_roi.groupby('budget_tier')['imdb_score'].mean().reset_index()
                    fig_imdb_tier = px.bar(avg_imdb_by_tier, x='budget_tier', y='imdb_score', color='budget_tier',
                                           title="Average IMDb Score by Budget Tier", template="plotly_dark")
                    st.plotly_chart(fig_imdb_tier, use_container_width=True)
                else:
                    st.info("Could not categorize content into budget tiers or no data left after tiering.")
            else:
                st.info("Not enough valid data to calculate ROI or analyze budget efficiency.")
        else:
            st.info("Not enough valid data for budget, views, or IMDb score to perform budget efficiency analysis.")
    else:
        st.info("Required columns ('budget_millions', 'views_millions', 'imdb_score') not available for this analysis.")

# Tool 49: Emerging Talent Spotlight (Directors/Actors) (was Tool 52)
with st.expander("🌟 Tool 49: Emerging Talent Spotlight (Directors/Actors)"):
    if all(col in df.columns for col in ['director', 'cast', 'title', 'imdb_score']):
        df_talent = df.copy()
        df_talent['imdb_score'] = pd.to_numeric(df_talent['imdb_score'], errors='coerce')
        if 'budget_millions' in df_talent.columns:
            df_talent['budget_millions'] = pd.to_numeric(df_talent['budget_millions'], errors='coerce')
        if 'views_millions' in df_talent.columns:
            df_talent['views_millions'] = pd.to_numeric(df_talent['views_millions'], errors='coerce')
        
        df_talent.dropna(subset=['director', 'cast', 'title', 'imdb_score'], inplace=True)

        st.subheader("Identify High-Potential Talent")
        min_display_count = 5 # Desired minimum number of talents to display
        max_titles_emerging = st.slider("Max Titles for 'Emerging' Status (fewer titles = more 'emerging'):", 1, 20, 10, key="emerging_max_titles_tool49", help="Lower values mean stricter criteria for 'emerging'. Consider dataset size.")
        min_avg_imdb_emerging = st.slider("Min Avg. IMDb Score for Spotlight:", 6.0, 9.5, 7.5, 0.1, key="emerging_min_imdb_tool49") 

        # Directors
        directors_exploded_talent = df_talent.assign(person=df_talent['director'].str.split(', ')).explode('person')
        directors_exploded_talent['person'] = directors_exploded_talent['person'].str.strip()
        director_stats = directors_exploded_talent.groupby('person').agg(
            title_count=('title', 'nunique'),
            avg_imdb_score=('imdb_score', 'mean'),
            avg_budget=('budget_millions', 'mean') if 'budget_millions' in df_talent.columns else pd.NamedAgg(column='title', aggfunc=lambda x: np.nan),
            avg_views=('views_millions', 'mean') if 'views_millions' in df_talent.columns else pd.NamedAgg(column='title', aggfunc=lambda x: np.nan)
        ).reset_index()
        if 'avg_budget' in director_stats.columns and 'avg_views' in director_stats.columns:
             director_stats['avg_roi'] = np.where(director_stats['avg_budget'] > 0.01, director_stats['avg_views'] / director_stats['avg_budget'], np.nan)
        
        emerging_directors_strict = director_stats[
            (director_stats['title_count'] <= max_titles_emerging) &
            (director_stats['avg_imdb_score'] >= min_avg_imdb_emerging)
        ].sort_values(by='avg_imdb_score', ascending=False)

        emerging_directors_display = emerging_directors_strict
        relaxed_directors_criteria = False
        if len(emerging_directors_strict) < min_display_count and not director_stats.empty:
            emerging_directors_display = director_stats.sort_values(
                by=['avg_imdb_score', 'title_count'], 
                ascending=[False, True]
            ).head(min_display_count)
            relaxed_directors_criteria = True

        st.markdown("#### Emerging Directors Spotlight")
        if not emerging_directors_display.empty:
            if relaxed_directors_criteria:
                st.caption(f"Note: Fewer than {min_display_count} directors met the strict 'emerging' criteria. Displaying top talents (up to {min_display_count}) by IMDb score, favoring fewer titles.")
            display_cols_directors = ['person', 'title_count', 'avg_imdb_score']
            if 'avg_roi' in emerging_directors_display.columns: display_cols_directors.append('avg_roi')
            st.dataframe(emerging_directors_display[display_cols_directors].rename(columns={'person': 'Director'}))
        else:
            st.info("No emerging directors found to display based on the current data and criteria.")

        # Actors
        actors_exploded_talent = df_talent.assign(person=df_talent['cast'].str.split(', ')).explode('person')
        actors_exploded_talent['person'] = actors_exploded_talent['person'].str.strip()
        actor_stats = actors_exploded_talent.groupby('person').agg(
            title_count=('title', 'nunique'),
            avg_imdb_score=('imdb_score', 'mean'),
            avg_budget=('budget_millions', 'mean') if 'budget_millions' in df_talent.columns else pd.NamedAgg(column='title', aggfunc=lambda x: np.nan),
            avg_views=('views_millions', 'mean') if 'views_millions' in df_talent.columns else pd.NamedAgg(column='title', aggfunc=lambda x: np.nan)
        ).reset_index()
        if 'avg_budget' in actor_stats.columns and 'avg_views' in actor_stats.columns:
            actor_stats['avg_roi'] = np.where(actor_stats['avg_budget'] > 0.01, actor_stats['avg_views'] / actor_stats['avg_budget'], np.nan)

        emerging_actors_strict = actor_stats[
            (actor_stats['title_count'] <= max_titles_emerging) &
            (actor_stats['avg_imdb_score'] >= min_avg_imdb_emerging)
        ].sort_values(by='avg_imdb_score', ascending=False)

        emerging_actors_display = emerging_actors_strict
        relaxed_actors_criteria = False
        if len(emerging_actors_strict) < min_display_count and not actor_stats.empty:
            emerging_actors_display = actor_stats.sort_values(
                by=['avg_imdb_score', 'title_count'],
                ascending=[False, True]
            ).head(min_display_count)
            relaxed_actors_criteria = True

        st.markdown("#### Emerging Actors Spotlight")
        if not emerging_actors_display.empty:
            if relaxed_actors_criteria:
                st.caption(f"Note: Fewer than {min_display_count} actors met the strict 'emerging' criteria. Displaying top talents (up to {min_display_count}) by IMDb score, favoring fewer titles.")
            display_cols_actors = ['person', 'title_count', 'avg_imdb_score']
            if 'avg_roi' in emerging_actors_display.columns: display_cols_actors.append('avg_roi')
            st.dataframe(emerging_actors_display[display_cols_actors].rename(columns={'person': 'Actor'}))
        else:
            st.info("No emerging actors found to display based on the current data and criteria.")
    else:
        st.info("Required columns ('director', 'cast', 'title', 'imdb_score') not available for Emerging Talent Spotlight.")

# Tool 50: Genre Synergy & Cross-Promotion Opportunities (was Tool 53)
with st.expander("🔗 Tool 50: Genre Synergy & Cross-Promotion Opportunities"):
    if 'listed_in' in df.columns and 'imdb_score' in df.columns:
        df_synergy = df.copy()
        df_synergy['imdb_score'] = pd.to_numeric(df_synergy['imdb_score'], errors='coerce')
        if 'views_millions' in df_synergy.columns:
            df_synergy['views_millions'] = pd.to_numeric(df_synergy['views_millions'], errors='coerce')
        df_synergy.dropna(subset=['listed_in', 'imdb_score'], inplace=True)

        if not df_synergy.empty:
            genre_pairs_data = []
            for index, row in df_synergy.iterrows():
                genres = sorted(list(set(g.strip() for g in str(row['listed_in']).split(',') if g.strip())))
                if len(genres) >= 2:
                    for pair in combinations(genres, 2):
                        genre_pairs_data.append({
                            'pair': tuple(sorted(pair)), 
                            'imdb_score': row['imdb_score'],
                            'views_millions': row.get('views_millions', np.nan) # Safely get views
                        })
            
            if genre_pairs_data:
                genre_pairs_df = pd.DataFrame(genre_pairs_data)
                genre_pair_stats = genre_pairs_df.groupby('pair').agg(
                    count=('imdb_score', 'count'),
                    avg_imdb_score=('imdb_score', 'mean'),
                    avg_views_millions=('views_millions', 'mean')
                ).reset_index()
                
                min_pair_occurrences = st.slider("Min Occurrences for Genre Pair Analysis:", 1, 10, 3, key="synergy_min_occur_tool50")
                genre_pair_stats_filtered = genre_pair_stats[genre_pair_stats['count'] >= min_pair_occurrences]

                st.subheader("Top Performing Genre Pairs by Average IMDb Score")
                top_pairs_imdb = genre_pair_stats_filtered.sort_values(by='avg_imdb_score', ascending=False).head(10)
                top_pairs_imdb['pair_str'] = top_pairs_imdb['pair'].apply(lambda x: f"{x[0]} & {x[1]}")
                fig_synergy_imdb = px.bar(top_pairs_imdb, x='avg_imdb_score', y='pair_str', orientation='h',
                                          title="Top Genre Pairs by Avg. IMDb Score", color='count',
                                          color_continuous_scale='Viridis',
                                          labels={'avg_imdb_score': 'Avg. IMDb Score', 'pair_str': 'Genre Pair', 'count': 'Co-occurrences'},
                                          template="plotly_dark")
                fig_synergy_imdb.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_synergy_imdb, use_container_width=True)
            else:
                st.info("Not enough titles with multiple genres to analyze synergy.")
        else:
            st.info("Not enough valid data for 'listed_in' and 'imdb_score' to perform genre synergy analysis.")
    else:
        st.info("Required columns ('listed_in', 'imdb_score') not available for Genre Synergy Analysis.")

# Tool 51: AI-Driven User Sentiment Analysis (Placed in AI-Powered Tools section)
with st.expander("😊 Tool 51: AI-Driven User Sentiment Analysis (Review Text)"):
    if gemini_key:
        st.subheader("Analyze Sentiment of User Review Text")
        st.markdown("""
        This tool uses AI to analyze the sentiment of a provided text, simulating how user reviews could be processed.
        If your dataset had a 'reviews' column, this AI could be applied to understand audience reception at scale.
        """)
        review_text_tool51 = st.text_area("Paste review text here:", height=150, placeholder="e.g., 'An absolute masterpiece, I was captivated from start to finish!' or 'The plot was predictable and the characters were underdeveloped.'", key="ai_sentiment_review_text_tool51")
        
        if st.button("Analyze Sentiment with AI", key="ai_analyze_sentiment_button_tool51"):
            if review_text_tool51:
                prompt = f"""Analyze the sentiment of the following review. Classify it as Positive, Negative, or Neutral. 
Provide a brief (1-sentence) explanation for your classification and a confidence score (Low, Medium, High).
Review: "{review_text_tool51}"

Sentiment: [Positive/Negative/Neutral]
Confidence: [Low/Medium/High]
Explanation: ..."""
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(prompt)
                    st.markdown("#### Sentiment Analysis Result:")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")
            else:
                st.warning("Please enter a review text to analyze.")
    else:
        st.info("Enter Gemini API key in the sidebar to use AI-Driven Sentiment Analysis.")


st.header("🔬 Advanced Exploratory Tools")

# Tool 52: Clustering for Genre Discovery
with st.expander("🧬 Tool 52: Clustering for Content Discovery"):
    st.subheader("Discover Hidden Content Groups using K-Means Clustering")
    st.markdown("""
    This tool attempts to find natural groupings in your content based on selected features.
    It uses the K-Means clustering algorithm. Clusters might reveal interesting segments,
    e.g., 'high-rated, low-budget movies' or 'long-running, popular TV shows'.
    """)

    if not df.empty:
        # Prepare data for clustering
        df_cluster = df.copy()
        
        # Feature Engineering: Duration
        def parse_duration_for_clustering(row):
            duration_str = str(row['duration'])
            if 'min' in duration_str: # Movie
                return int(re.findall(r'\d+', duration_str)[0])
            elif 'Season' in duration_str: # TV Show
                return int(re.findall(r'\d+', duration_str)[0]) * 50 # Arbitrary scaling for seasons vs minutes
            return np.nan

        if 'duration' in df_cluster.columns and 'type' in df_cluster.columns:
            df_cluster['duration_numeric_clustering'] = df_cluster.apply(parse_duration_for_clustering, axis=1)

        numeric_cols_for_clustering = ['imdb_score', 'release_year', 'budget_millions', 'views_millions', 'duration_numeric_clustering']
        available_features = [col for col in numeric_cols_for_clustering if col in df_cluster.columns and pd.api.types.is_numeric_dtype(df_cluster[col])]
        
        if not available_features:
            st.warning("No suitable numeric features found for clustering (e.g., imdb_score, release_year, budget_millions, views_millions, parsed duration).")
        else:
            selected_features_clustering = st.multiselect("Select features for clustering:", available_features, default=available_features[:2] if len(available_features) >=2 else available_features, key="clustering_features_tool52")

            if len(selected_features_clustering) < 2:
                st.warning("Please select at least two features for clustering.")
            else:
                df_cluster_selected = df_cluster[selected_features_clustering + ['title', 'listed_in']].copy()
                df_cluster_selected.dropna(subset=selected_features_clustering, inplace=True)

                if len(df_cluster_selected) < 10: # Need enough data points
                    st.warning(f"Not enough data ({len(df_cluster_selected)} rows) after cleaning for the selected features. Please select different features or check your dataset.")
                else:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_cluster_selected[selected_features_clustering])

                    num_clusters = st.slider("Select number of clusters (k):", 2, 10, 3, key="num_clusters_tool52")
                    
                    if st.button("Run Clustering", key="run_clustering_tool52"):
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                        df_cluster_selected['cluster'] = kmeans.fit_predict(X_scaled)

                        st.subheader(f"Clustering Results ({num_clusters} Clusters)")

                        # Visualization
                        if len(selected_features_clustering) == 2:
                            fig_cluster = px.scatter(df_cluster_selected, x=selected_features_clustering[0], y=selected_features_clustering[1],
                                                     color='cluster', hover_data=['title'], title="Content Clusters", template="plotly_dark")
                            st.plotly_chart(fig_cluster, use_container_width=True)
                        else: # Use PCA for >2 features
                            pca = PCA(n_components=2, random_state=42)
                            X_pca = pca.fit_transform(X_scaled)
                            df_cluster_selected['pca1'] = X_pca[:, 0]
                            df_cluster_selected['pca2'] = X_pca[:, 1]
                            fig_cluster_pca = px.scatter(df_cluster_selected, x='pca1', y='pca2', color='cluster',
                                                         hover_data=['title'] + selected_features_clustering,
                                                         title="Content Clusters (PCA Reduced)",
                                                         labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'},
                                                         template="plotly_dark")
                            st.plotly_chart(fig_cluster_pca, use_container_width=True)
                            st.caption("Clusters visualized using the first two Principal Components as original features are more than two.")

                        # Cluster Characterization
                        st.subheader("Cluster Characteristics (Mean Values & Top Genres)")
                        for i in range(num_clusters):
                            cluster_data = df_cluster_selected[df_cluster_selected['cluster'] == i]
                            st.markdown(f"#### Cluster {i} ({len(cluster_data)} titles)")
                            st.dataframe(cluster_data[selected_features_clustering].mean().to_frame(name='Mean Value').style.format("{:.2f}"))
                            
                            if 'listed_in' in cluster_data.columns:
                                all_genres_in_cluster = [genre.strip() for sublist in cluster_data['listed_in'].str.split(',') for genre in sublist if genre.strip()]
                                top_genres_cluster = Counter(all_genres_in_cluster).most_common(3)
                                st.write(f"Top Genres: {', '.join([f'{g[0]} ({g[1]})' for g in top_genres_cluster])}")
                            st.dataframe(cluster_data[['title'] + selected_features_clustering].head(3))
    else:
        st.warning("Dataset not loaded. Clustering cannot be performed.")

st.header("🛠️ Utilities & User Features")

# Tool 53: Data Cleaning Showcase
with st.expander("🧹 Tool 53: Data Cleaning & Preparation Showcase"):
    st.subheader("Illustrating Data Cleaning Steps")
    st.markdown("""
    Effective data analysis relies on clean, well-prepared data. This section showcases
    live demonstrations of common cleaning and feature engineering techniques. These operations are performed on a *temporary copy* of your data for this showcase only.
    """)

    if not df.empty:
        df_showcase = df.copy() # Work on a copy for this showcase

        st.markdown("#### 1. Handling Missing Values")
        st.write("Missing values can skew analysis. Let's demonstrate with simulated missing data and imputation:")

        # Simulate missing data if the original df is clean, for demonstration
        simulated_missing = False
        if df_showcase.isnull().sum().sum() == 0: # Check if original df has no NaNs
            simulated_missing = True
            st.caption("ℹ️ _The loaded dataset appears to have no missing values. For demonstration, some missing values will be simulated below._")
            # Introduce some NaNs into df_showcase
            cols_to_mess_up = [] # Columns to introduce NaNs into
            # Prioritize columns that are good candidates for imputation demonstration
            if 'director' in df_showcase.columns and df_showcase['director'].nunique() > 1: # Ensure some variety
                cols_to_mess_up.append('director')
            if 'country' in df_showcase.columns and df_showcase['country'].nunique() > 1:
                 cols_to_mess_up.append('country')
            if 'imdb_score' in df_showcase.columns and pd.api.types.is_numeric_dtype(df_showcase['imdb_score']) and df_showcase['imdb_score'].notna().sum() > 5: # Ensure enough non-NaNs to calculate median
                cols_to_mess_up.append('imdb_score')
            
            if cols_to_mess_up and len(df_showcase) > 0:
                num_rows_to_affect = min(3, len(df_showcase)) # Affect up to 3 rows
                for col in cols_to_mess_up:
                    # Ensure we don't try to sample more than available rows
                    sample_size = min(num_rows_to_affect, len(df_showcase))
                    if sample_size > 0:
                        random_indices = random.sample(range(len(df_showcase)), k=sample_size)
                        df_showcase.loc[random_indices, col] = np.nan
            elif not cols_to_mess_up:
                st.write("Could not find suitable columns (e.g., director, country, imdb_score with sufficient data) to simulate missing data for imputation demo.")

        missing_counts_showcase = df_showcase.isnull().sum()
        missing_df_showcase = missing_counts_showcase[missing_counts_showcase > 0].rename("Missing Count").to_frame()
        
        if not missing_df_showcase.empty:
            st.write("Summary of (potentially simulated) missing values **before** imputation:")
            st.dataframe(missing_df_showcase.T)

            st.write("Let's demonstrate imputation:")
            imputed_cols_display = []

            # Impute 'director' if it was messed up and exists
            if 'director' in df_showcase.columns and df_showcase['director'].isnull().any():
                original_director_col = 'director_original_demo'
                imputed_director_col = 'director_imputed_demo'
                df_showcase[original_director_col] = df_showcase['director'] # Keep original for comparison
                fill_value_director = "Unknown Director"
                df_showcase[imputed_director_col] = df_showcase['director'].fillna(fill_value_director)
                st.markdown(f"- **Categorical Imputation ('director'):** Filled `NaN` with '{fill_value_director}'.")
                imputed_cols_display.extend([original_director_col, imputed_director_col])

            # Impute 'imdb_score' if it was messed up and exists
            if 'imdb_score' in df_showcase.columns and df_showcase['imdb_score'].isnull().any():
                original_imdb_col = 'imdb_score_original_demo'
                imputed_imdb_col = 'imdb_score_imputed_demo'
                df_showcase[original_imdb_col] = df_showcase['imdb_score'] # Keep original
                median_imdb = df_showcase['imdb_score'].median()
                if pd.notna(median_imdb):
                    df_showcase[imputed_imdb_col] = df_showcase['imdb_score'].fillna(median_imdb)
                    st.markdown(f"- **Numerical Imputation ('imdb_score'):** Filled `NaN` with median ({median_imdb:.2f}).")
                    imputed_cols_display.extend([original_imdb_col, imputed_imdb_col])
                else:
                    st.markdown(f"- **Numerical Imputation ('imdb_score'):** Could not calculate median (all values might be NaN after simulation). Skipping imputation demo for this column.")

            if imputed_cols_display:
                st.write("Missing values **after** imputation (for demonstrated columns):")
                st.dataframe(df_showcase[imputed_cols_display].isnull().sum().rename("Missing Count").to_frame().T)
                st.write("Sample of imputed data (showing original and imputed values for a few rows where changes occurred):")
                
                # Find rows where imputation happened for any of the demoed columns
                condition = False
                if 'director_original_demo' in df_showcase.columns and 'director_imputed_demo' in df_showcase.columns:
                    condition = (df_showcase['director_original_demo'].isnull()) & (df_showcase['director_imputed_demo'].notnull())
                if 'imdb_score_original_demo' in df_showcase.columns and 'imdb_score_imputed_demo' in df_showcase.columns:
                    condition_imdb = (df_showcase['imdb_score_original_demo'].isnull()) & (df_showcase['imdb_score_imputed_demo'].notnull())
                    condition = condition | condition_imdb if isinstance(condition, pd.Series) else condition_imdb

                if isinstance(condition, pd.Series) and condition.any():
                    st.dataframe(df_showcase.loc[condition, ['title'] + imputed_cols_display].head())
                else:
                    st.caption("_No rows with visible imputation changes to display in this sample (might be due to random sampling of NaNs)._")
            else:
                st.info("No suitable columns were available or had missing values to demonstrate imputation after simulation.")

        elif simulated_missing and not cols_to_mess_up : # Simulation was intended but no suitable columns
            pass # Message already shown
        else:
            st.success("No missing values found in the loaded dataset, and no suitable columns were identified for simulation in this showcase.")

        st.markdown("#### 2. Feature Engineering Examples")
        st.markdown("Creating new features from existing ones can unlock deeper insights.")

        if 'duration' in df_showcase.columns and 'type' in df_showcase.columns:
            st.markdown("- **Extracting Numeric Duration:** Converting 'duration' (e.g., '1 Season', '90 min') to numeric.")
            df_showcase['movie_duration_minutes_demo'] = df_showcase.apply(lambda row: int(re.findall(r'\d+', str(row['duration']))[0]) if row['type'] == 'Movie' and 'min' in str(row['duration']) else np.nan, axis=1)
            df_showcase['tv_show_seasons_demo'] = df_showcase.apply(lambda row: int(re.findall(r'\d+', str(row['duration']))[0]) if row['type'] == 'TV Show' and 'Season' in str(row['duration']) else np.nan, axis=1)
            st.write("Sample with new duration columns:")
            st.dataframe(df_showcase[['title', 'type', 'duration', 'movie_duration_minutes_demo', 'tv_show_seasons_demo']].head())

        if 'release_year' in df_showcase.columns:
            st.markdown("- **Creating 'Decade':** Grouping by 'release_year'.")
            df_showcase['decade_demo'] = (df_showcase['release_year'].astype(int) // 10) * 10
            st.write("Sample with new 'decade_demo' column:")
            st.dataframe(df_showcase[['title', 'release_year', 'decade_demo']].head())

        if 'listed_in' in df_showcase.columns:
            st.markdown("- **Splitting 'listed_in' (Genres):** Creating one row per genre for a title.")
            # Ensure 'listed_in' is string and handle NaNs before splitting
            df_showcase_genre_safe = df_showcase.dropna(subset=['listed_in'])
            if not df_showcase_genre_safe.empty:
                exploded_genres_demo = df_showcase_genre_safe.assign(genre_demo=df_showcase_genre_safe['listed_in'].str.split(', ')).explode('genre_demo')
                exploded_genres_demo['genre_demo'] = exploded_genres_demo['genre_demo'].str.strip()
                st.write("Sample of exploded genres (first 10 rows):")
                st.dataframe(exploded_genres_demo[['title', 'genre_demo']].head(10))
            else:
                st.write("Not enough data in 'listed_in' to demonstrate genre splitting.")

        st.markdown("#### 3. Data Type Conversion")
        st.write("Ensuring columns have appropriate data types is crucial. Example: converting 'release_year' to integer if it's not already.")
        
        if 'release_year' in df_showcase.columns:
            st.write(f"Original 'release_year' dtype: `{df_showcase['release_year'].dtype}`")
            # Make a copy for conversion to avoid altering the original df_showcase column if it's used elsewhere in this tool
            df_showcase['release_year_converted_demo'] = pd.to_numeric(df_showcase['release_year'], errors='coerce')
            # Check if conversion to numeric resulted in NaNs for all values (e.g. if it was all strings)
            if not df_showcase['release_year_converted_demo'].isnull().all():
                 df_showcase['release_year_converted_demo'] = df_showcase['release_year_converted_demo'].astype('Int64') # Use Int64 to handle potential NaNs
                 st.write(f"Converted 'release_year_converted_demo' dtype: `{df_showcase['release_year_converted_demo'].dtype}`")
                 st.write("Sample of original vs. converted 'release_year':")
                 st.dataframe(df_showcase[['title', 'release_year', 'release_year_converted_demo']].head())
            else:
                st.write("Could not convert 'release_year' to a numeric type for demonstration (all values might be unparseable).")
        else:
            st.info("'release_year' column not found for data type conversion demo.")

        st.markdown("---")
        st.success("Data cleaning and preparation demonstration complete!")
    else:
        st.warning("Dataset not loaded. Cannot showcase data cleaning.")

st.header("🎮 Gamified Analytics")

# Tool 54: "Guess the Views" Challenge (Renumbered from 56)
with st.expander("🎯 Tool 54: 'Guess the Views' Challenge"):
    st.subheader("🔮 Test Your Viewership Clairvoyance!")
    st.markdown("Can you predict the (simulated) views? Flex your analytical intuition!")
    if 'views_millions' not in df.columns or 'title' not in df.columns or 'imdb_score' not in df.columns or 'listed_in' not in df.columns or 'release_year' not in df.columns:
        st.warning("Required columns (views_millions, title, imdb_score, listed_in, release_year) are not available for this game.")
    else:
        if 'game_score_gtv' not in st.session_state:
            st.session_state.game_score_gtv = 0
        if 'current_challenge_gtv' not in st.session_state:
            st.session_state.current_challenge_gtv = None
        if 'gtv_revealed' not in st.session_state:
            st.session_state.gtv_revealed = False
        if 'gtv_difficulty' not in st.session_state:
            st.session_state.gtv_difficulty = "Medium Might" # Default difficulty
        if 'gtv_hint_taken' not in st.session_state:
            st.session_state.gtv_hint_taken = False
        if 'gtv_hint_message' not in st.session_state:
            st.session_state.gtv_hint_message = ""

        df_game_gtv = df.dropna(subset=['views_millions', 'title', 'imdb_score', 'listed_in', 'release_year'])
        if df_game_gtv['views_millions'].isnull().all():
            st.warning("The 'views_millions' column has no valid data for this game after filtering.")
            df_game_gtv = pd.DataFrame() # Empty it to prevent further game logic

        if not df_game_gtv.empty:
            st.session_state.gtv_difficulty = st.radio(
                "Select Difficulty:",
                ["Easy Peasy", "Medium Might", "Hardcore Analyst"],
                index=["Easy Peasy", "Medium Might", "Hardcore Analyst"].index(st.session_state.gtv_difficulty),
                key="gtv_difficulty_selector",
                horizontal=True
            )

        if st.button("Get New Challenge Title 🔮", key="new_challenge_gtv_tool54"):
            if not df_game_gtv.empty:
                st.session_state.current_challenge_gtv = df_game_gtv.sample(1).iloc[0]
                st.session_state.gtv_revealed = False
                st.session_state.user_guess_gtv = None # Reset previous guess
                st.session_state.gtv_hint_taken = False
                st.session_state.gtv_hint_message = ""
                st.rerun()
            else:
                st.error("Not enough data to start the game.")
        
        if st.session_state.current_challenge_gtv is not None:
            challenge_title = st.session_state.current_challenge_gtv
            st.markdown(f"#### Behold! The Title of Destiny: **{challenge_title['title']}**")
            
            # Display info based on difficulty
            info_to_show = {"Type": challenge_title.get('type', 'N/A'), "Release Year": challenge_title['release_year']}
            if st.session_state.gtv_difficulty == "Easy Peasy":
                info_to_show["Genre(s)"] = challenge_title['listed_in']
                info_to_show["IMDb Score"] = f"{challenge_title['imdb_score']:.1f}"
            elif st.session_state.gtv_difficulty == "Medium Might":
                info_to_show["Genre(s)"] = challenge_title['listed_in']
                # IMDb score is hidden
            # For "Hardcore Analyst", only Type and Release Year are shown by default

            for key, value in info_to_show.items():
                st.write(f"- **{key}:** {value}")

            actual_views = challenge_title['views_millions']
            
            if not st.session_state.gtv_revealed:
                st.markdown("---")
                st.markdown("🤔 **Channel your inner psychic...**")
                user_guess = st.number_input("Guess the Views (Millions):", min_value=0.0, value=st.session_state.get('user_guess_gtv', 10.0), step=1.0, key="guess_input_gtv_tool54")
                st.session_state.user_guess_gtv = user_guess # Store current input

                # Hint System
                if not st.session_state.gtv_hint_taken:
                    if st.button("Need a Hint? (-15 points)", key="gtv_hint_button"):
                        st.session_state.gtv_hint_taken = True
                        median_views_sample = df_game_gtv['views_millions'].median()
                        if actual_views > median_views_sample:
                            st.session_state.gtv_hint_message = f"Psst... the actual views are HIGHER than {median_views_sample:.1f} million."
                        else:
                            st.session_state.gtv_hint_message = f"Psst... the actual views are LOWER than {median_views_sample:.1f} million."
                        st.rerun()
                
                if st.session_state.gtv_hint_message:
                    st.info(st.session_state.gtv_hint_message)

                if st.button("✨ Submit Guess & Reveal ✨", key="submit_guess_gtv_tool54"):
                    st.session_state.gtv_revealed = True
                    st.write(f"Your Guess: **{user_guess:.1f} million**")
                    st.write(f"Actual Views: **{actual_views:.1f} million**")
                    
                    difference = abs(user_guess - actual_views)
                    percentage_diff = (difference / actual_views) * 100 if actual_views > 0.01 else float('inf') # Avoid division by zero for very small actual_views

                    points_earned = 0
                    feedback_message = ""

                    if percentage_diff <= 5: # Bullseye
                        points_earned = 150
                        feedback_message = f"🎯 BULLSEYE! Are you a wizard?! Difference: {difference:.1f}M ({percentage_diff:.1f}%). You earned a whopping {points_earned} points!"
                        st.balloons()
                    elif percentage_diff <= 15: # Excellent
                        points_earned = 100
                        feedback_message = f"🎉 Excellent guess, Nostradamus Jr.! Difference: {difference:.1f}M ({percentage_diff:.1f}%). You earned {points_earned} points!"
                    elif percentage_diff <= 30: # Good
                        points_earned = 50
                        feedback_message = f"👍 Good shot! Difference: {difference:.1f}M ({percentage_diff:.1f}%). You earned {points_earned} points!"
                    elif percentage_diff <= 60: # Fair
                        points_earned = 20
                        feedback_message = f"🙂 Fairly close. Difference: {difference:.1f}M ({percentage_diff:.1f}%). You earned {points_earned} points."
                    else:
                        feedback_message = f"😬 My grandma guesses better... just kidding (mostly)! Difference: {difference:.1f}M ({percentage_diff:.1f}%). No points this round."
                    
                    if st.session_state.gtv_hint_taken:
                        points_earned = max(0, points_earned - 15) # Apply penalty
                        feedback_message += " (-15 for the hint!)"

                    if points_earned > 0: st.success(feedback_message)
                    else: st.error(feedback_message)

                    st.session_state.game_score_gtv += points_earned
                    st.rerun() # Rerun to update display and hide input
            else: # Revealed state
                st.write(f"Your Guess was: **{st.session_state.user_guess_gtv:.1f} million**")
                st.write(f"Actual Views: **{actual_views:.1f} million**")
                # The feedback message is already shown on reveal, so no need to repeat unless you want to.
        
        if not df_game_gtv.empty: # Only show score and reset if game is possible
            st.markdown(f"--- \n**Your Total Score: {st.session_state.game_score_gtv} points**")
            if st.button("Reset Score & Game", key="reset_score_gtv_tool54"):
                st.session_state.game_score_gtv = 0
                st.session_state.current_challenge_gtv = None
                st.session_state.gtv_revealed = False
                st.session_state.gtv_hint_taken = False
                st.session_state.gtv_hint_message = ""
                st.rerun()

# Tool 55: Netflix Trivia Challenge (Renumbered from 57)
with st.expander("🧠 Tool 55: Netflix Trivia Challenge"):
    st.subheader("🏆 The Ultimate Netflix Dataset Trivia Showdown!")
    st.markdown("Time to flex those brain-noggin-muscles! Correct answers build your streak for bonus points!")

    if 'trivia_score' not in st.session_state:
        st.session_state.trivia_score = 0
        st.session_state.trivia_questions = []
        st.session_state.current_trivia_idx = 0
        st.session_state.trivia_answer_submitted = False
        st.session_state.trivia_streak = 0
    
    TRIVIA_STREAK_BONUS_THRESHOLD = 3
    TRIVIA_STREAK_BONUS_POINTS = 20

    def generate_trivia_questions(df_trivia, num_questions=5):
        questions = []
        if df_trivia.empty: return questions

        # Q1: Highest IMDb score
        if 'imdb_score' in df_trivia.columns and 'title' in df_trivia.columns:
            top_rated = df_trivia.dropna(subset=['imdb_score', 'title']).nlargest(5, 'imdb_score')
            if not top_rated.empty:
                correct_ans = top_rated.iloc[0]['title']
                options = random.sample(df_trivia['title'].dropna().unique().tolist(), 3) + [correct_ans]
                random.shuffle(options)
                questions.append({"question": "Which title has the highest IMDb score in this dataset sample?", "options": options, "answer": correct_ans, "type": "mcq"})

        # Q2: Release year of a random title
        if 'release_year' in df_trivia.columns and 'title' in df_trivia.columns:
            sample_q2_df = df_trivia.dropna(subset=['release_year', 'title'])
            if not sample_q2_df.empty:
                sample_q2 = sample_q2_df.sample(1).iloc[0]
            correct_ans = str(int(sample_q2['release_year']))
            other_years = [str(int(y)) for y in df_trivia['release_year'].dropna().unique() if int(y) != int(sample_q2['release_year'])]
            options = random.sample(other_years, min(3, len(other_years))) + [correct_ans]
            random.shuffle(options)
            questions.append({"question": f"In what year was '{sample_q2['title']}' released?", "options": options, "answer": correct_ans, "type": "mcq"})

        # Q3: Most common genre
        if 'listed_in' in df_trivia.columns:
            all_genres_trivia = [g.strip() for sublist in df_trivia['listed_in'].dropna().str.split(',') for g in sublist if g.strip()]
            if all_genres_trivia:
                genre_counts_trivia = Counter(all_genres_trivia)
                correct_ans = genre_counts_trivia.most_common(1)[0][0]
                other_genres = [g for g,c in genre_counts_trivia.most_common(10) if g != correct_ans]
                options = random.sample(other_genres, min(3, len(other_genres))) + [correct_ans]
                random.shuffle(options)
                questions.append({"question": "What is the most common genre in this dataset?", "options": options, "answer": correct_ans, "type": "mcq"})
        
        # Q4: Country with most titles
        if 'country' in df.columns:
            country_counts_trivia = df_trivia['country'].astype(str).apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else "Unknown").value_counts()
            if not country_counts_trivia.empty:
                correct_ans = country_counts_trivia.index[0]
                other_countries = country_counts_trivia.index[1:min(10, len(country_counts_trivia))].tolist()
                options = random.sample(other_countries, min(3, len(other_countries))) + [correct_ans]
                random.shuffle(options)
                questions.append({"question": "Which primary country has produced the most titles in this dataset?", "options": options, "answer": correct_ans, "type": "mcq"})
        
        # Q5: Actor in most X genre titles
        if 'cast' in df_trivia.columns and 'listed_in' in df_trivia.columns:
            df_q5 = df_trivia.dropna(subset=['cast', 'listed_in'])
            if not df_q5.empty:
                random_genre_q5 = random.choice(df_q5['listed_in'].dropna().unique()).split(',')[0].strip()
                actor_genre_df = df_q5[df_q5['listed_in'].str.contains(random_genre_q5, na=False)]
                if not actor_genre_df.empty:
                    actor_genre_exploded = actor_genre_df.assign(actor=actor_genre_df['cast'].str.split(',')).explode('actor')
                    actor_genre_exploded['actor'] = actor_genre_exploded['actor'].str.strip()
                    actor_counts_in_genre = actor_genre_exploded['actor'].value_counts()
                    if len(actor_counts_in_genre) >= 4:
                        correct_ans = actor_counts_in_genre.index[0]
                        options = random.sample(actor_counts_in_genre.index[1:].tolist(), 3) + [correct_ans]
                        random.shuffle(options)
                        questions.append({"question": f"Which actor appeared in the most '{random_genre_q5}' titles in this dataset?", "options": options, "answer": correct_ans, "type": "mcq"})

        # Q6: Title NOT from a specific decade
        if 'release_year' in df_trivia.columns and 'title' in df_trivia.columns:
            df_q6 = df_trivia.dropna(subset=['release_year', 'title'])
            if len(df_q6['release_year'].unique()) > 10 : # Ensure enough variety for decades
                random_decade_q6 = (random.choice(df_q6['release_year'].unique().astype(int)) // 10) * 10
                titles_in_decade = df_q6[(df_q6['release_year'] >= random_decade_q6) & (df_q6['release_year'] < random_decade_q6 + 10)]['title'].unique().tolist()
                titles_not_in_decade = df_q6[~((df_q6['release_year'] >= random_decade_q6) & (df_q6['release_year'] < random_decade_q6 + 10))]['title'].unique().tolist()
                if len(titles_in_decade) >=3 and titles_not_in_decade:
                    correct_ans = random.choice(titles_not_in_decade)
                    options = random.sample(titles_in_decade, 3) + [correct_ans]
                    random.shuffle(options)
                    questions.append({"question": f"Which of these titles was NOT released in the {random_decade_q6}s?", "options": options, "answer": correct_ans, "type": "mcq"})

        return random.sample(questions, min(num_questions, len(questions))) if questions else []

    if st.button("Start/Restart Trivia Challenge", key="start_trivia_tool55"):
        st.session_state.trivia_questions = generate_trivia_questions(df, num_questions=5)
        st.session_state.current_trivia_idx = 0
        st.session_state.trivia_score = 0
        st.session_state.trivia_streak = 0
        st.session_state.trivia_answer_submitted = False
        if not st.session_state.trivia_questions:
            st.warning("Could not generate trivia questions. Dataset might be too small or lack required columns.")
        st.rerun()

    if st.session_state.trivia_questions and st.session_state.current_trivia_idx < len(st.session_state.trivia_questions):
        current_q = st.session_state.trivia_questions[st.session_state.current_trivia_idx]
        st.markdown(f"**Question {st.session_state.current_trivia_idx + 1}/{len(st.session_state.trivia_questions)}:** {current_q['question']}")
        
        if not st.session_state.trivia_answer_submitted:
            user_answer_trivia = st.radio("Your answer:", current_q['options'], key=f"trivia_q_{st.session_state.current_trivia_idx}")
            if st.button("Submit Answer", key=f"submit_trivia_{st.session_state.current_trivia_idx}"):
                st.session_state.trivia_user_choice = user_answer_trivia # Store choice
                st.session_state.trivia_answer_submitted = True
                if user_answer_trivia == current_q['answer']:
                    st.session_state.trivia_streak += 1
                    st.session_state.trivia_score += 10
                    feedback = f"Correct! You're a data wizard! The answer was {current_q['answer']}. You earned 10 points."
                    if st.session_state.trivia_streak > 0 and st.session_state.trivia_streak % TRIVIA_STREAK_BONUS_THRESHOLD == 0:
                        st.session_state.trivia_score += TRIVIA_STREAK_BONUS_POINTS
                        feedback += f" 🔥 Streak Bonus! +{TRIVIA_STREAK_BONUS_POINTS} points for {st.session_state.trivia_streak} correct in a row!"
                        st.balloons()
                    st.success(feedback)
                else:
                    st.session_state.trivia_streak = 0
                    st.error(f"Oof, that one was a curveball from the data dimension! The correct answer was {current_q['answer']}.")
                st.rerun()
        else: # Answer submitted, show result and next button
            st.write(f"You chose: {st.session_state.trivia_user_choice}")
            if st.session_state.trivia_user_choice == current_q['answer']:
                 st.markdown(f"**Result:** <span style='color:lightgreen;'>Correct!</span> The answer was **{current_q['answer']}**.", unsafe_allow_html=True)
            else:
                 st.markdown(f"**Result:** <span style='color:salmon;'>Incorrect.</span> The correct answer was **{current_q['answer']}**.", unsafe_allow_html=True)

            if st.button("Next Question", key=f"next_trivia_{st.session_state.current_trivia_idx}"):
                st.session_state.current_trivia_idx += 1
                st.session_state.trivia_answer_submitted = False
                st.rerun()

    # Check if questions were generated and if the game has started
    elif st.session_state.trivia_questions and st.session_state.current_trivia_idx >= len(st.session_state.trivia_questions) and len(st.session_state.trivia_questions) > 0:
        st.balloons()
        st.subheader("Trivia Challenge Complete!")
        st.write(f"Your final score: {st.session_state.trivia_score} out of {len(st.session_state.trivia_questions) * 10} points.")

    st.markdown(f"--- \n**Current Trivia Score: {st.session_state.trivia_score} points**")

st.markdown("---")
st.markdown("**Netflix Data Analytics Dashboard** - Comprehensive toolkit for data analysis capstone projects")
