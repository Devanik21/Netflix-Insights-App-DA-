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

st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide")
st.title("🎬 Netflix Data Analytics Dashboard")
st.markdown("**Advanced Analytics Suite for Data Analyst Capstone Project**")

# Enhanced sample Netflix dataset
@st.cache_data
def load_sample_netflix_data():
    """Create comprehensive Netflix dataset for analysis"""
    sample_data = {
        'show_id': [f's{i}' for i in range(1, 51)],
        'type': ['Movie', 'TV Show'] * 25,
        'title': ['Dark', 'Stranger Things', 'The Irishman', 'Money Heist', 'Bird Box', 'Roma', 'The Crown', 'Extraction', 'Ozark', 'The Platform',
                 'Narcos', 'Black Mirror', 'The Witcher', 'Orange is the New Black', 'House of Cards', 'Mindhunter', 'Breaking Bad', 'Better Call Saul',
                 'The Office', 'Friends', 'Squid Game', 'Lupin', 'Emily in Paris', 'Bridgerton', 'The Queen\'s Gambit', 'Tiger King', 'Making a Murderer',
                 'Wild Wild Country', 'Our Planet', 'Chef\'s Table', 'The Movies That Made Us', 'High Score', 'The Social Dilemma', 'My Octopus Teacher',
                 'American Factory', 'Icarus', 'Won\'t You Be My Neighbor?', 'RBG', 'Free Solo', 'The Great Hack', 'Explained', 'Abstract', 'Salt Fat Acid Heat',
                 'Ugly Delicious', 'Street Food', 'The Mind, Explained', 'Sex Education', 'Elite', 'Cable Girls', 'Money Heist: Korea'],
        'director': ['Baran bo Odar', 'The Duffer Brothers', 'Martin Scorsese', 'Álex Pina', 'Susanne Bier'] * 10,
        'country': ['Germany', 'United States', 'United States', 'Spain', 'United States', 'Mexico', 'United Kingdom', 'United States', 'United States', 'Spain'] * 5,
        'release_year': [2017, 2016, 2019, 2017, 2018, 2018, 2016, 2020, 2017, 2019, 2015, 2011, 2019, 2013, 2013, 2017, 2008, 2015, 2005, 1994,
                        2021, 2021, 2020, 2020, 2020, 2020, 2015, 2018, 2019, 2017, 2019, 2020, 2020, 2020, 2019, 2017, 2018, 2018, 2018, 2019,
                        2018, 2017, 2017, 2019, 2019, 2019, 2019, 2017, 2017, 2021],
        'rating': ['TV-14', 'TV-14', 'R', 'TV-MA', 'R', 'R', 'TV-MA', 'R', 'TV-MA', 'TV-MA'] * 5,
        'duration': ['1 Season', '4 Seasons', '209 min', '4 Seasons', '124 min', '135 min', '6 Seasons', '116 min', '4 Seasons', '94 min'] * 5,
        'listed_in': ['Crime TV Shows, International TV Shows, TV Dramas', 'TV Horror, TV Sci-Fi & Fantasy, TV Thrillers', 'Crime Movies, Dramas', 
                     'Crime TV Shows, International TV Shows, Spanish-Language TV Shows', 'Horror Movies, Sci-Fi Movies, Thrillers'] * 10,
        'imdb_score': np.random.uniform(6.0, 9.5, 50).round(1),
        'budget_millions': np.random.uniform(10, 200, 50).round(1),
        'views_millions': np.random.uniform(50, 500, 50).round(1)
    }
    return pd.DataFrame(sample_data)

# Sidebar
st.sidebar.header("📂 Data Source")
file = st.sidebar.file_uploader("Upload Netflix dataset (CSV)", type="csv")

if file is None:
    st.sidebar.info("Using sample dataset")
    df = load_sample_netflix_data()
else:
    df = pd.read_csv(file)
    st.success("Custom dataset loaded!")

# Gemini API
gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Titles", len(df))
with col2:
    st.metric("Movies", len(df[df['type'] == 'Movie']) if 'type' in df.columns else 0)
with col3:
    st.metric("TV Shows", len(df[df['type'] == 'TV Show']) if 'type' in df.columns else 0)
with col4:
    st.metric("Countries", df['country'].nunique() if 'country' in df.columns else 0)

# Tool 1: Content Performance Analytics
with st.expander("📊 Tool 1: Content Performance Analytics"):
    if 'imdb_score' in df.columns and 'views_millions' in df.columns:
        fig = px.scatter(df, x='imdb_score', y='views_millions', color='type', size='budget_millions',
                        title="Content Performance: Rating vs Viewership",
                        labels={'imdb_score': 'IMDB Score', 'views_millions': 'Views (Millions)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        top_performers = df.nlargest(5, 'views_millions')[['title', 'views_millions', 'imdb_score']]
        st.subheader("Top 5 Most Viewed")
        st.dataframe(top_performers)

# Tool 2: Genre Trend Analysis
with st.expander("📈 Tool 2: Genre Trend Analysis"):
    if 'release_year' in df.columns and 'listed_in' in df.columns:
        genre_data = []
        for _, row in df.iterrows():
            genres = [g.strip() for g in str(row['listed_in']).split(',')]
            for genre in genres:
                genre_data.append({'release_year': row['release_year'], 'genre': genre})
        
        genre_df = pd.DataFrame(genre_data)
        genre_trends = genre_df.groupby(['release_year', 'genre']).size().reset_index(name='count')
        top_genres = genre_df['genre'].value_counts().head(6).index.tolist()
        
        fig = px.line(genre_trends[genre_trends['genre'].isin(top_genres)], 
                     x='release_year', y='count', color='genre',
                     title="Genre Popularity Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)

# Tool 3: Geographic Content Distribution
with st.expander("🌍 Tool 3: Geographic Content Distribution"):
    if 'country' in df.columns:
        country_data = df['country'].value_counts().head(10)
        fig = px.bar(x=country_data.values, y=country_data.index, orientation='h',
                    title="Content Production by Country")
        st.plotly_chart(fig, use_container_width=True)
        
        # Market penetration analysis
        st.subheader("Market Analysis")
        market_share = (country_data / country_data.sum() * 100).round(2)
        st.write("Market Share (%):", market_share.to_dict())

# Tool 4: Content Duration Analysis
with st.expander("⏱️ Tool 4: Content Duration Analysis"):
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
                                 labels={'x': 'Duration (minutes)', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if tv_seasons:
                fig = px.histogram(x=tv_seasons, title="TV Show Seasons Distribution",
                                 labels={'x': 'Number of Seasons', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

# Tool 5: Rating Distribution Analysis
with st.expander("🏆 Tool 5: Rating Distribution Analysis"):
    if 'rating' in df.columns:
        rating_counts = df['rating'].value_counts()
        fig = px.pie(values=rating_counts.values, names=rating_counts.index,
                    title="Content Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age demographic insights
        mature_content = len(df[df['rating'].isin(['R', 'TV-MA'])])
        family_content = len(df[df['rating'].isin(['G', 'PG', 'TV-G', 'TV-Y'])])
        st.write(f"Mature Content: {mature_content} ({mature_content/len(df)*100:.1f}%)")
        st.write(f"Family-Friendly: {family_content} ({family_content/len(df)*100:.1f}%)")

# Tool 6: Release Year Timeline
with st.expander("📅 Tool 6: Release Year Timeline"):
    if 'release_year' in df.columns:
        yearly_releases = df['release_year'].value_counts().sort_index()
        fig = px.area(x=yearly_releases.index, y=yearly_releases.values,
                     title="Content Release Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Decade analysis
        df['decade'] = (df['release_year'] // 10) * 10
        decade_counts = df['decade'].value_counts().sort_index()
        st.write("Content by Decade:", decade_counts.to_dict())

# Tool 7: Budget vs Performance ROI
with st.expander("💰 Tool 7: Budget vs Performance ROI"):
    if 'budget_millions' in df.columns and 'views_millions' in df.columns:
        df['roi'] = df['views_millions'] / df['budget_millions']
        
        fig = px.scatter(df, x='budget_millions', y='roi', color='type', size='imdb_score',
                        title="Budget vs ROI Analysis",
                        labels={'budget_millions': 'Budget (Millions)', 'roi': 'ROI (Views/Budget)'})
        st.plotly_chart(fig, use_container_width=True)
        
        high_roi = df.nlargest(5, 'roi')[['title', 'budget_millions', 'views_millions', 'roi']]
        st.subheader("Best ROI Content")
        st.dataframe(high_roi)

# Tool 8: Content Correlation Matrix
with st.expander("🔗 Tool 8: Content Correlation Matrix"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Correlations")
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.5:
                    st.write(f"{col1} ↔ {col2}: {corr_val:.3f}")

# Tool 9: Content Gap Analysis
with st.expander("📊 Tool 9: Content Gap Analysis"):
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
with st.expander("🔮 Tool 10: Predictive Analytics Dashboard"):
    if 'imdb_score' in df.columns and 'views_millions' in df.columns:
        # Simple trend prediction
        from sklearn.linear_model import LinearRegression
        
        X = df[['imdb_score', 'budget_millions']].fillna(df[['imdb_score', 'budget_millions']].mean())
        y = df['views_millions'].fillna(df['views_millions'].mean())
        
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=predictions, mode='markers', name='Predicted vs Actual'))
        fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                mode='lines', name='Perfect Prediction'))
        fig.update_layout(title="Viewership Prediction Model",
                         xaxis_title="Actual Views", yaxis_title="Predicted Views")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model insights
        st.write(f"Model Score: {model.score(X, y):.3f}")
        st.write("Feature Importance:")
        st.write(f"IMDB Score: {model.coef_[0]:.2f}")
        st.write(f"Budget: {model.coef_[1]:.2f}")

# Advanced Analytics Tools
st.header("🔬 Advanced Analytics")

# Tool 11: Statistical Summary
with st.expander("📋 Tool 11: Statistical Summary Report"):
    st.subheader("Dataset Overview")
    st.write(df.describe())
    
    st.subheader("Data Quality Report")
    missing_data = df.isnull().sum()
    st.write("Missing Values:", missing_data[missing_data > 0].to_dict())
    
    st.subheader("Categorical Analysis")
    for col in df.select_dtypes(include=['object']).columns:
        st.write(f"**{col}**: {df[col].nunique()} unique values")

# Tool 12: Content Recommendation Engine
with st.expander("🎯 Tool 12: Content Recommendation Engine"):
    if 'listed_in' in df.columns:
        user_genre = st.selectbox("Select preferred genre:", 
                                 ['Drama', 'Comedy', 'Action', 'Horror', 'Sci-Fi', 'Crime'])
        
        # Simple content-based filtering
        genre_matches = df[df['listed_in'].str.contains(user_genre, na=False)]
        
        if not genre_matches.empty:
            if 'imdb_score' in df.columns:
                recommendations = genre_matches.nlargest(5, 'imdb_score')
            else:
                recommendations = genre_matches.head(5)
            
            st.subheader(f"Top {user_genre} Recommendations")
            st.dataframe(recommendations[['title', 'country', 'release_year']])

# Tool 13: Executive Summary Generator
with st.expander("📄 Tool 13: Executive Summary Generator"):
    summary_data = {
        'Total Content': len(df),
        'Content Mix': f"{len(df[df['type'] == 'Movie'])} Movies, {len(df[df['type'] == 'TV Show'])} TV Shows",
        'Geographic Reach': f"{df['country'].nunique()} countries",
        'Release Timeline': f"{df['release_year'].min()}-{df['release_year'].max()}",
        'Top Genre': df['listed_in'].str.split(', ').explode().value_counts().index[0] if 'listed_in' in df.columns else 'N/A'
    }
    
    st.subheader("Executive Summary")
    for key, value in summary_data.items():
        st.write(f"**{key}**: {value}")

# Tool 14: AI-Powered Insights
with st.expander("🤖 Tool 14: AI-Powered Insights"):
    if gemini_key:
        analysis_type = st.selectbox("Select analysis type:", 
                                   ["Content Strategy", "Market Gaps", "Performance Insights", "Trend Predictions"])
        
        if st.button("Generate AI Insights"):
            prompt = f"""
            Analyze this Netflix dataset summary for {analysis_type}:
            
            Dataset: {len(df)} titles
            Content mix: {df['type'].value_counts().to_dict() if 'type' in df.columns else 'N/A'}
            Top countries: {df['country'].value_counts().head(3).to_dict() if 'country' in df.columns else 'N/A'}
            Release years: {df['release_year'].min()}-{df['release_year'].max() if 'release_year' in df.columns else 'N/A'}
            
            Provide 3-5 actionable insights for {analysis_type}.
            """
            
            try:
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter Gemini API key to use AI insights")

# Tool 15: Data Export & Reporting
with st.expander("📤 Tool 15: Data Export & Reporting"):
    export_format = st.selectbox("Export format:", ["CSV", "JSON", "Excel Summary"])
    
    if st.button("Generate Export"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "netflix_analysis.csv", "text/csv")
        elif export_format == "JSON":
            json_data = df.to_json(orient='records')
            st.download_button("Download JSON", json_data, "netflix_analysis.json", "application/json")
        else:
            st.success("Excel summary prepared (implementation would generate comprehensive report)")

# Tool 16: Director Performance Analysis
with st.expander("🎬 Tool 16: Director Performance Analysis"):
    if 'director' in df.columns and 'title' in df.columns:
        st.subheader("Director Analysis")
        # Filter out rows where director is NaN or 'Unknown' if necessary, though sample data is clean
        # For this example, we'll assume directors are mostly single individuals or known groups
        # In a real dataset, director column might need more cleaning (e.g., splitting multiple directors)
        
        director_counts = df['director'].value_counts().head(10)
        fig_director_titles = px.bar(director_counts, x=director_counts.index, y=director_counts.values,
                                     labels={'x': 'Director', 'y': 'Number of Titles'},
                                     title="Top 10 Directors by Number of Titles")
        st.plotly_chart(fig_director_titles, use_container_width=True)

        if 'imdb_score' in df.columns:
            # Calculate average IMDb score per director
            # For simplicity, considering only directors with at least 2 titles for score analysis
            director_title_counts = df['director'].value_counts()
            directors_for_score_analysis = director_title_counts[director_title_counts >= 2].index
            
            if not directors_for_score_analysis.empty:
                avg_score_by_director = df[df['director'].isin(directors_for_score_analysis)].groupby('director')['imdb_score'].mean().sort_values(ascending=False).head(10)
                fig_director_score = px.bar(avg_score_by_director, x=avg_score_by_director.index, y=avg_score_by_director.values,
                                             labels={'x': 'Director', 'y': 'Average IMDb Score'},
                                             title="Top Directors by Average IMDb Score (min. 2 titles)")
                st.plotly_chart(fig_director_score, use_container_width=True)
            else:
                st.write("Not enough data for director IMDb score analysis (requires directors with >= 2 titles).")
    else:
        st.info("Director and/or title information not available for this analysis.")

# Tool 17: Title Word Cloud
with st.expander("☁️ Tool 17: Title Word Cloud"):
    if 'title' in df.columns:
        st.subheader("Word Cloud from Content Titles")
        text = " ".join(title for title in df['title'].astype(str))
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No titles available to generate a word cloud.")
    else:
        st.info("Title information not available for word cloud generation.")

# Tool 18: Content Type Evolution Over Time
with st.expander("🔄 Tool 18: Content Type Evolution Over Time"):
    if 'release_year' in df.columns and 'type' in df.columns:
        content_type_evolution = df.groupby(['release_year', 'type']).size().reset_index(name='count')
        fig = px.line(content_type_evolution, x='release_year', y='count', color='type',
                     title="Content Type Releases Over Time", labels={'release_year': 'Release Year', 'count': 'Number of Titles'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Release year and/or type information not available for this analysis.")

st.markdown("---")
st.markdown("**Netflix Data Analytics Dashboard** - Comprehensive toolkit for data analysis capstone projects")
