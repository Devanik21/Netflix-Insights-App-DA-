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
        'director': ['Baran bo Odar', 'The Duffer Brothers', 'Martin Scorsese', 'Álex Pina', 'Susanne Bier',
                     'Alfonso Cuarón', 'Peter Morgan', 'Sam Hargrave', 'Bill Dubuque', 'Galder Gaztelu-Urrutia'] * 5, # Adjusted for 50 titles
        'cast': [
            'Louis Hofmann, Karoline Eichhorn, Lisa Vicari', 'Millie Bobby Brown, Finn Wolfhard, Winona Ryder', 'Robert De Niro, Al Pacino, Joe Pesci',
            'Úrsula Corberó, Álvaro Morte, Itziar Ituño', 'Sandra Bullock, Trevante Rhodes, John Malkovich',
            'Yalitza Aparicio, Marina de Tavira, Fernando Grediaga', 'Olivia Colman, Tobias Menzies, Helena Bonham Carter', 'Chris Hemsworth, Rudhraksh Jaiswal, Randeep Hooda',
            'Jason Bateman, Laura Linney, Sofia Hublitz', 'Iván Massagué, Zorion Eguileor, Antonia San Juan'] * 5, # 50 cast entries
        'country': ['Germany', 'United States', 'United States', 'Spain', 'United States', 'Mexico', 'United Kingdom', 'United States', 'United States', 'Spain'] * 5,
        'release_year': [2017, 2016, 2019, 2017, 2018, 2018, 2016, 2020, 2017, 2019, 2015, 2011, 2019, 2013, 2013, 2017, 2008, 2015, 2005, 1994,
                        2021, 2021, 2020, 2020, 2020, 2020, 2015, 2018, 2019, 2017, 2019, 2020, 2020, 2020, 2019, 2017, 2018, 2018, 2018, 2019,
                        2018, 2017, 2017, 2019, 2019, 2019, 2019, 2017, 2017, 2021],
        'rating': ['TV-14', 'TV-14', 'R', 'TV-MA', 'R', 'R', 'TV-MA', 'R', 'TV-MA', 'TV-MA'] * 5,
        'duration': ['1 Season', '4 Seasons', '209 min', '4 Seasons', '124 min', '135 min', '6 Seasons', '116 min', '4 Seasons', '94 min'] * 5,
        'listed_in': ['Crime TV Shows, International TV Shows, TV Dramas', 'TV Horror, TV Sci-Fi & Fantasy, TV Thrillers', 'Crime Movies, Dramas', 
                     'Crime TV Shows, International TV Shows, Spanish-Language TV Shows', 'Horror Movies, Sci-Fi Movies, Thrillers'] * 10,
        'imdb_score': np.clip(np.random.normal(7.5, 0.8, 50), 4.0, 9.8).round(1), # More realistic IMDb scores
        'date_added': pd.to_datetime([f'{np.random.randint(2015, 2023)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}' for _ in range(50)]),
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
        'Total Content': len(df) if not df.empty else 0,
        'Content Mix': f"{len(df[df['type'] == 'Movie'])} Movies, {len(df[df['type'] == 'TV Show'])} TV Shows" if 'type' in df.columns and not df.empty else "N/A",
        'Geographic Reach': f"{df['country'].nunique()} countries" if 'country' in df.columns and not df.empty else "N/A",
        'Release Timeline': f"{df['release_year'].min()}-{df['release_year'].max()}" if 'release_year' in df.columns and not df.empty else "N/A",
        'Top Genre': df['listed_in'].str.split(', ', expand=True).stack().value_counts().index[0] if 'listed_in' in df.columns and not df.empty and not df['listed_in'].dropna().empty else 'N/A'
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
                model = genai.GenerativeModel("gemini-2.0-flash")
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

# Tool 19: Actor/Cast Performance Analysis
with st.expander("🎭 Tool 19: Actor/Cast Performance Analysis"):
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
                                          title="Top 10 Actors by Number of Titles Appeared In")
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
                                                 title="Top Actors by Average IMDb Score (min. 1 title)")
                        st.plotly_chart(fig_actor_score, use_container_width=True)
                    else:
                        st.write("Could not calculate average IMDb scores for actors.")
                else:
                    st.write("Not enough data for actor IMDb score analysis (requires actors with >= 1 title).")
        else:
            st.write("No cast information available to analyze.")
    else:
        st.info("Cast, title, and/or IMDb score information not available for this analysis.")

# Tool 20: Genre Deep Dive
with st.expander("🔎 Tool 20: Genre Deep Dive"):
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
                fig_genre_trend = px.line(genre_release_trend, x='release_year', y='count', title=f"Release Trend for {selected_genre}")
                st.plotly_chart(fig_genre_trend, use_container_width=True)

                st.subheader(f"Top 5 Titles in {selected_genre} (by IMDb Score)")
                st.dataframe(genre_df.nlargest(5, 'imdb_score')[['title', 'release_year', 'imdb_score', 'type']])
            else:
                st.write(f"No titles found for the genre: {selected_genre}")
    else:
        st.info("Required columns (listed_in, release_year, imdb_score, title) not available for Genre Deep Dive.")

# Tool 21: AI Chat with Dataset
with st.expander("💬 Tool 21: AI Chat with Dataset"):
    if gemini_key:
        st.subheader("Ask a question about your dataset")
        user_question = st.text_area("Your question:", height=100, placeholder="e.g., What are the top 5 countries with the most titles? or How many movies were released in 2020?")

        if st.button("Ask AI 🤖"):
            if user_question:
                # Prepare a summary of the DataFrame for the AI
                # Using .to_string() to get a string representation
                # Limiting the head() and describe() output to keep the prompt concise
                try:
                    df_summary = f"""
                    Here's a summary of the dataset I'm working with:
                    Column Names: {df.columns.tolist()}
                    Data Types:
{df.dtypes.to_string()}
                    First 5 Rows:
{df.head().to_string()}
                    Basic Statistics:
{df.describe(include='all').head().to_string()} 
                    Total rows: {len(df)}
                    """

                    prompt = f"""
                    You are a data analysis assistant. Based *only* on the following dataset summary, please answer the user's question.
                    If the information is not present in the summary or cannot be inferred, please state that.
                    
                    Dataset Summary:
                    {df_summary}
                    
                    User's Question: {user_question}
                    
                    Answer:
                    """
                    model = genai.GenerativeModel("gemini-2.0-flash") 
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred while querying the AI: {e}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please enter your Gemini API key in the sidebar to use the AI Chat feature.")

# Tool 22: Content Freshness Analysis
with st.expander("⏳ Tool 22: Content Freshness Analysis"):
    if 'release_year' in df.columns:
        st.subheader("Content Age Analysis")
        current_year = datetime.now().year
        df_copy = df.copy() # Work on a copy to avoid modifying the original df
        df_copy['content_age'] = current_year - df_copy['release_year']

        fig_age_dist = px.histogram(df_copy, x='content_age', nbins=20,
                                    title="Distribution of Content Age (Years)",
                                    labels={'content_age': 'Content Age (Years)'})
        st.plotly_chart(fig_age_dist, use_container_width=True)

        if 'imdb_score' in df_copy.columns:
            fig_age_score = px.scatter(df_copy, x='content_age', y='imdb_score', trendline="ols",
                                       title="Content Age vs. IMDb Score",
                                       labels={'content_age': 'Content Age (Years)', 'imdb_score': 'IMDb Score'})
            st.plotly_chart(fig_age_score, use_container_width=True)

        if 'views_millions' in df_copy.columns:
            fig_age_views = px.scatter(df_copy, x='content_age', y='views_millions', trendline="ols",
                                       title="Content Age vs. Views (Millions)",
                                       labels={'content_age': 'Content Age (Years)', 'views_millions': 'Views (Millions)'})
            st.plotly_chart(fig_age_views, use_container_width=True)
    else:
        st.info("Release year information not available for content freshness analysis.")

# Tool 23: Interactive World Map of Content Production
with st.expander("🗺️ Tool 23: Interactive World Map of Content Production"):
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
                                    title="Number of Titles Produced by Country")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.write("No country data available to display on the map.")
    else:
        st.info("Country information not available for map visualization.")

# Tool 24: Movie vs. TV Show Deep Comparison
with st.expander("🎬 vs 📺 Tool 24: Movie vs. TV Show Deep Comparison"):
    if 'type' in df.columns:
        st.subheader("Movie vs. TV Show Metrics")
        movies_df = df[df['type'] == 'Movie']
        tv_shows_df = df[df['type'] == 'TV Show']

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
                fig_movie_ratings = px.pie(movies_df, names='rating', title='Movie Rating Distribution')
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
                fig_tv_ratings = px.pie(tv_shows_df, names='rating', title='TV Show Rating Distribution')
                st.plotly_chart(fig_tv_ratings, use_container_width=True)
    else:
        st.info("Content 'type' information not available for this comparison.")

# Tool 25: Release Month/Seasonality Analysis
with st.expander("🗓️ Tool 25: Release Month/Seasonality Analysis"):
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
                                    labels={'x': 'Month Added', 'y': 'Number of Titles'})
        st.plotly_chart(fig_month_releases, use_container_width=True)
    else:
        st.info("'date_added' column not available for seasonality analysis.")

# Tool 26: Keyword Search in Titles
with st.expander("🔑 Tool 26: Keyword Search in Titles"):
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

# Tool 27: Content Rating vs. IMDb Score Analysis
with st.expander("🔞 Tool 27: Content Rating vs. IMDb Score Analysis"):
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
                                      color='rating')
            st.plotly_chart(fig_rating_score, use_container_width=True)

            st.subheader("IMDb Score Distribution by Rating (Box Plot)")
                fig_box_rating_score = px.box(df_tool27, x='rating', y='imdb_score', color='rating',
                                          title="IMDb Score Distribution by Content Rating",
                                          labels={'rating': 'Content Rating', 'imdb_score': 'IMDb Score'})
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

# Tool 28: Top Director-Actor Pairs
with st.expander("🤝 Tool 28: Top Director-Actor Collaborations"):
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
                                        labels={'Number of Collaborations': 'Number of Titles Together'})
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

# Tool 29: Time Series Analysis of IMDb Scores
with st.expander("📈 Tool 29: IMDb Score Trends Over Release Years"):
    if 'release_year' in df.columns and 'imdb_score' in df.columns:
        st.subheader("Average IMDb Score of Content by Release Year")
        avg_score_by_year = df.groupby('release_year')['imdb_score'].mean().reset_index()
        avg_score_by_year = avg_score_by_year.sort_values('release_year')

        if not avg_score_by_year.empty:
            fig_score_trend = px.line(avg_score_by_year, x='release_year', y='imdb_score',
                                      title="Trend of Average IMDb Scores Over Release Years",
                                      labels={'release_year': 'Release Year', 'imdb_score': 'Average IMDb Score'})
            fig_score_trend.add_trace(go.Scatter(x=avg_score_by_year['release_year'], y=avg_score_by_year['imdb_score'].rolling(window=5, center=True, min_periods=1).mean(),
                                                 mode='lines', name='5-Year Rolling Avg', line=dict(dash='dash')))
            st.plotly_chart(fig_score_trend, use_container_width=True)
        else:
            st.write("Not enough data to analyze IMDb score trends over years.")
    else:
        st.info("'release_year' and/or 'imdb_score' columns not available for this analysis.")

# Tool 30: Comparative Analysis of Top N Countries
with st.expander("🌍 Tool 30: Multi-Country Content Profile Comparison"):
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
                                 title="Movie vs. TV Show Count by Country")
            st.plotly_chart(fig_type_comp, use_container_width=True)

            if 'imdb_score' in comparison_df.columns:
                st.markdown("#### Average IMDb Score by Selected Country")
                fig_imdb_comp = px.bar(comparison_df.groupby('primary_country')['imdb_score'].mean().reset_index(),
                                     x='primary_country', y='imdb_score', color='primary_country',
                                     title="Average IMDb Score by Country")
                st.plotly_chart(fig_imdb_comp, use_container_width=True)
        else:
            st.write("Please select at least one country.")
    else:
        st.info("'country', 'type', and/or 'listed_in' columns not available for this analysis.")

st.markdown("---")
st.markdown("**Netflix Data Analytics Dashboard** - Comprehensive toolkit for data analysis capstone projects")
