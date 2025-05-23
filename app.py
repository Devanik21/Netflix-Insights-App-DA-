import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io
import os

st.set_page_config(page_title="Netflix Content Insights Tool", layout="wide")
st.title("🎬 Netflix Content Insights Tool")
st.markdown("Analyze genre popularity, trends, and gaps across the globe using Netflix data ✨")

# Create sample Netflix dataset
@st.cache_data
def load_sample_netflix_data():
    """Create a sample Netflix dataset for demonstration"""
    sample_data = {
        'show_id': ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10'],
        'type': ['Movie', 'TV Show', 'Movie', 'TV Show', 'Movie', 'Movie', 'TV Show', 'Movie', 'TV Show', 'Movie'],
        'title': ['Dark', 'Stranger Things', 'The Irishman', 'Money Heist', 'Bird Box', 'Roma', 'The Crown', 'Extraction', 'Ozark', 'The Platform'],
        'director': ['Baran bo Odar', 'The Duffer Brothers', 'Martin Scorsese', 'Álex Pina', 'Susanne Bier', 'Alfonso Cuarón', 'Peter Morgan', 'Sam Hargrave', 'Bill Dubuque', 'Galder Gaztelu-Urrutia'],
        'cast': ['Louis Hofmann, Oliver Masucci', 'Winona Ryder, David Harbour', 'Robert De Niro, Al Pacino', 'Úrsula Corberó, Álvaro Morte', 'Sandra Bullock, Trevante Rhodes', 'Yalitza Aparicio, Marina de Tavira', 'Claire Foy, Matt Smith', 'Chris Hemsworth, Rudhraksh Jaiswal', 'Jason Bateman, Laura Linney', 'Ivan Massagué, Zorion Eguileor'],
        'country': ['Germany', 'United States', 'United States', 'Spain', 'United States', 'Mexico', 'United Kingdom', 'United States', 'United States', 'Spain'],
        'date_added': ['December 1, 2017', 'July 15, 2016', 'November 27, 2019', 'December 20, 2017', 'December 21, 2018', 'December 14, 2018', 'November 4, 2016', 'April 24, 2020', 'July 21, 2017', 'March 20, 2020'],
        'release_year': [2017, 2016, 2019, 2017, 2018, 2018, 2016, 2020, 2017, 2019],
        'rating': ['TV-14', 'TV-14', 'R', 'TV-MA', 'R', 'R', 'TV-MA', 'R', 'TV-MA', 'TV-MA'],
        'duration': ['1 Season', '4 Seasons', '209 min', '4 Seasons', '124 min', '135 min', '6 Seasons', '116 min', '4 Seasons', '94 min'],
        'listed_in': ['Crime TV Shows, International TV Shows, TV Dramas', 'TV Horror, TV Sci-Fi & Fantasy, TV Thrillers', 'Crime Movies, Dramas', 'Crime TV Shows, International TV Shows, Spanish-Language TV Shows', 'Horror Movies, Sci-Fi Movies, Thrillers', 'Dramas, Independent Movies, International Movies', 'British TV Shows, Docuseries, International TV Shows', 'Action & Adventure, Thrillers', 'Crime TV Shows, TV Dramas, TV Thrillers', 'Horror Movies, International Movies, Sci-Fi Movies']
    }
    return pd.DataFrame(sample_data)

# Sidebar
st.sidebar.header("📂 Upload Netflix Dataset")
file = st.sidebar.file_uploader("Upload your Netflix dataset (CSV)", type="csv")

# Load sample dataset if user doesn't upload
if file is None:
    st.sidebar.info("No file uploaded. Using sample Netflix dataset.")
    df = load_sample_netflix_data()
    st.info("📊 Using sample Netflix dataset for demonstration")
else:
    df = pd.read_csv(file)
    st.success("Dataset loaded successfully!")

# Gemini API
gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Load Data Analysis
if df is not None:
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head())
        st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    with st.expander("📈 Genre Popularity Over Time"):
        if 'release_year' in df.columns and 'listed_in' in df.columns:
            # Clean and prepare data
            df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
            df_clean = df.dropna(subset=['release_year'])
            df_clean['release_year'] = df_clean['release_year'].astype(int)
            
            # Split genres and create year-genre combinations
            genre_data = []
            for _, row in df_clean.iterrows():
                genres = [g.strip() for g in str(row['listed_in']).split(',')]
                for genre in genres:
                    genre_data.append({'release_year': row['release_year'], 'genre': genre})
            
            genre_df = pd.DataFrame(genre_data)
            genre_year_counts = genre_df.groupby(['release_year', 'genre']).size().reset_index(name='count')
            
            # Get top 5 genres overall
            top_genres = genre_df['genre'].value_counts().head(5).index.tolist()
            filtered_data = genre_year_counts[genre_year_counts['genre'].isin(top_genres)]
            
            if not filtered_data.empty:
                fig = px.line(filtered_data, x='release_year', y='count', color='genre', 
                             title="Top 5 Genre Trends Over Years",
                             labels={'count': 'Number of Titles', 'release_year': 'Release Year'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to show genre trends")
        else:
            st.warning("Required columns: 'release_year' and 'listed_in'")
    
    with st.expander("🌍 Country-wise Content Distribution"):
        if 'country' in df.columns and 'listed_in' in df.columns:
            # Country distribution
            country_counts = df['country'].value_counts().head(10)
            fig_country = px.bar(x=country_counts.index, y=country_counts.values,
                               title="Top 10 Countries by Content Count",
                               labels={'x': 'Country', 'y': 'Number of Titles'})
            st.plotly_chart(fig_country, use_container_width=True)
            
            # Genre distribution by top countries
            st.subheader("Genre Distribution by Country")
            top_countries = country_counts.head(5).index.tolist()
            country_genre_data = []
            
            for _, row in df[df['country'].isin(top_countries)].iterrows():
                genres = [g.strip() for g in str(row['listed_in']).split(',')]
                for genre in genres:
                    country_genre_data.append({'country': row['country'], 'genre': genre})
            
            if country_genre_data:
                cg_df = pd.DataFrame(country_genre_data)
                country_genre_pivot = cg_df.groupby(['country', 'genre']).size().unstack(fill_value=0)
                st.dataframe(country_genre_pivot)
        else:
            st.warning("Required columns: 'country' and 'listed_in'")
    
    with st.expander("📊 Content Type Analysis"):
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            fig_type = px.pie(values=type_counts.values, names=type_counts.index,
                             title="Distribution of Movies vs TV Shows")
            st.plotly_chart(fig_type, use_container_width=True)
    
    with st.expander("💬 Ask Gemini About Your Data"):
        if gemini_key:
            user_q = st.text_input("Ask a question like 'Which genre is most popular?' or 'Show content trends by country'")
            if user_q:
                # Compose prompt from data summary
                data_summary = f"""
                Dataset Overview:
                - Total titles: {len(df)}
                - Columns: {', '.join(df.columns.tolist())}
                - Sample data:
                {df.head().to_string()}
                
                Key Statistics:
                - Content types: {df['type'].value_counts().to_dict() if 'type' in df.columns else 'N/A'}
                - Top countries: {df['country'].value_counts().head(5).to_dict() if 'country' in df.columns else 'N/A'}
                - Release year range: {df['release_year'].min()}-{df['release_year'].max() if 'release_year' in df.columns else 'N/A'}
                """
                
                prompt = f"""
                Here is a summary of a Netflix dataset:
                {data_summary}
                
                User Question: {user_q}
                
                Please provide insights based on this Netflix data. Be specific and reference the actual data patterns.
                """
                
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    response = model.generate_content(prompt)
                    st.markdown("**Gemini Response:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error from Gemini: {e}")
        else:
            st.info("Please enter your Gemini API key in the sidebar to ask questions ✨")
else:
    st.info("Upload a dataset to begin ✨")
