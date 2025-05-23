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

# Enhanced sample Netflix dataset
@st.cache_data
def load_sample_netflix_data():
    """Create a more comprehensive Netflix dataset for analysis."""
    num_titles = 200  # Increased number of titles
    # Generate 'type' data first
    types_list = np.random.choice(['Movie', 'TV Show'], num_titles, p=[0.6, 0.4])
    # Generate 'duration' based on the pre-generated 'types_list'
    durations_list = [
        f'{np.random.randint(60, 240)} min' if t == 'Movie' else f'{np.random.randint(1, 10)} Seasons'
        for t in types_list
    ]
    # More realistic titles
    realistic_titles = [
        # Existing Titles
        "Stranger Things", "The Crown", "Bridgerton", "Money Heist", "The Witcher", "Ozark", "Narcos",
        "Black Mirror", "Squid Game", "Lupin", "Emily in Paris", "The Queen's Gambit", "Dark",
        "You", "Sex Education", "Cobra Kai", "Outer Banks", "Never Have I Ever", "Lucifer", "Elite",
        "The Umbrella Academy", "Dead to Me", "Russian Doll", "Mindhunter", "Peaky Blinders",
        "Extraction", "The Irishman", "Bird Box", "Roma", "Marriage Story", "The Platform", "Enola Holmes",
        "Project Power", "The Old Guard", "Spenser Confidential", "6 Underground", "Murder Mystery",
        "The Kissing Booth", "To All the Boys I've Loved Before", "Always Be My Maybe", "Set It Up", # Approx 40
        "The Social Dilemma", "My Octopus Teacher", "American Factory", "Tiger King", "Making a Murderer",
        "Chef's Table", "Our Planet", "Explained", "Unorthodox", "When They See Us", "The Haunting of Hill House",
        "Midnight Mass", "Maid", "Clickbait", "Behind Her Eyes", "Shadow and Bone", "Sweet Tooth",
        "Virgin River", "Firefly Lane", "Ginny & Georgia", "The Circle", "Too Hot to Handle", "Love is Blind",
        "Selling Sunset", "Bling Empire", "Floor Is Lava", "Nailed It!", "Queer Eye", "Rhythm + Flow",
        "The Great British Baking Show", "Anne with an E", "Atypical", "BoJack Horseman", "Big Mouth",
        "The Dragon Prince", "She-Ra and the Princesses of Power", "Hilda", "Klaus", "Over the Moon",
        "The Mitchells vs. the Machines", "Vivo", "Wish Dragon", "I Lost My Body", "The Willoughbys",
        "Next Gen", "Fear Street Trilogy", "Army of the Dead", "The Midnight Sky", "Don't Look Up", # Approx 100
        "Red Notice", "The Adam Project", "The Gray Man", "Glass Onion: A Knives Out Mystery", "Hustle",
        "The Sea Beast", "Slumberland", "Guillermo del Toro's Pinocchio", "Matilda the Musical",
        "All Quiet on the Western Front", "Blonde", "White Noise", "The Pale Blue Eye", "You People",
        "Luther: The Fallen Sun", "Murder Mystery 2", "A Tourist's Guide to Love", "The Mother",
        "Extraction 2", "They Cloned Tyrone", "Heart of Stone", "Reptile", "The Killer", "Leave the World Behind",
        "Rebel Moon: Part One – A Child of Fire", "Lift", "Damsel", "Spaceman", "Irish Wish",
        "Wednesday", "The Diplomat", "The Night Agent", "Beef", "Queen Charlotte: A Bridgerton Story",
        "XO, Kitty", "FUBAR", "Glamorous", "The Lincoln Lawyer", "Manifest", "Vikings: Valhalla",
        "One Piece", "Avatar: The Last Airbender (Live Action)", "3 Body Problem", "The Gentlemen (Series)", # Approx 140
        # Adding ~100 more diverse titles
        "Breaking Bad", "Game of Thrones", "Friends", "The Office (US)", "Parks and Recreation", "Seinfeld",
        "The Sopranos", "The Wire", "Chernobyl", "Band of Brothers", "Planet Earth II", "Blue Planet II",
        "Cosmos: A Spacetime Odyssey", "Rick and Morty", "Attack on Titan", "Death Note", "Fullmetal Alchemist: Brotherhood",
        "Cowboy Bebop", "Avatar: The Last Airbender (Animated)", "The Legend of Korra", "Sherlock", "Doctor Who",
        "Better Call Saul", "Fargo", "True Detective", "Westworld", "The Mandalorian", "Ted Lasso", "Succession",
        "The Boys", "Invincible", "Arcane", "Cyberpunk: Edgerunners", "Blue Eye Samurai", "The Bear", "Severance",
        "Yellowjackets", "House of the Dragon", "The Last of Us (Series)", "Andor", "Loki", "WandaVision",
        "The Simpsons", "South Park", "Family Guy", "Bob's Burgers", "Futurama", "King of the Hill",
        "Pulp Fiction", "The Shawshank Redemption", "The Dark Knight", "Forrest Gump", "Inception", "The Matrix",
        "Goodfellas", "Fight Club", "The Lord of the Rings: The Fellowship of the Ring", "Spirited Away", "Parasite",
        "Interstellar", "Gladiator", "Saving Private Ryan", "The Green Mile", "City of God", "Amelie",
        "Pan's Labyrinth", "Oldboy (2003)", "A Separation", "The Lives of Others", "Train to Busan", "Shoplifters",
        "Portrait of a Lady on Fire", "Everything Everywhere All at Once", "CODA", "Nomadland", "Drive My Car",
        "Dune (2021)", "Blade Runner 2049", "Mad Max: Fury Road", "Whiplash", "La La Land", "Get Out",
        "Moonlight", "Lady Bird", "Little Women (2019)", "The Grand Budapest Hotel", "Her", "Ex Machina",
        "Arrival", "Sicario", "Hell or High Water", "Knives Out", "Once Upon a Time in Hollywood", "Joker",
        "1917", "Tenet", "The Batman", "Top Gun: Maverick", "Oppenheimer", "Barbie", "Poor Things",
        "Killers of the Flower Moon", "Anatomy of a Fall", "The Holdovers", "Past Lives", "Spider-Man: Into the Spider-Verse",
        "Spider-Man: Across the Spider-Verse", "The Boy and the Heron", "Howl's Moving Castle", "Princess Mononoke",
        "Your Name.", "Weathering with You", "A Silent Voice", "Grave of the Fireflies", "Perfect Blue",
        "The Handmaiden", "Memories of Murder", "Burning", "Minari", "Sound of Metal", "Another Round",
        "The Father", "Judas and the Black Messiah", "Promising Young Woman", "Nomadland", "The Power of the Dog",
        "Drive", "Lost in Translation", "Eternal Sunshine of the Spotless Mind", "No Country for Old Men",
        "There Will Be Blood", "The Social Network", "Zodiac", "Inglourious Basterds", "Django Unchained" # Added ~110 more
    ]

    sample_data = {
        'show_id': [f's{i}' for i in range(1, num_titles + 1)],
        'type': types_list,
        'title': np.random.choice(realistic_titles, num_titles, replace=True), # Use realistic titles
        'director': np.random.choice([
            'Martin Scorsese', 'Steven Spielberg', 'Christopher Nolan', 'Quentin Tarantino',
            'Alfred Hitchcock', 'Stanley Kubrick', 'Greta Gerwig', 'Bong Joon-ho', 'Akira Kurosawa',
            'Ingmar Bergman', 'Ridley Scott', 'Francis Ford Coppola', 'Pedro Almodóvar'
        ], num_titles),
        'cast': [', '.join(np.random.choice([
            'Robert De Niro', 'Leonardo DiCaprio', 'Meryl Streep', 'Tom Hanks', 'Brad Pitt', 'Scarlett Johansson',
            'Denzel Washington', 'Cate Blanchett', 'Morgan Freeman', 'Natalie Portman', 'Joaquin Phoenix',
            'Kate Winslet', 'Samuel L. Jackson', 'Julia Roberts', 'Johnny Depp'
        ], size=np.random.randint(2, 5), replace=False)) for _ in range(num_titles)],
        'country': np.random.choice([
            'United States', 'United Kingdom', 'Canada', 'India', 'South Korea', 'Japan', 'France', 'Spain',
            'Germany', 'Mexico', 'Brazil', 'Australia', 'China', 'Italy', 'Argentina'
        ], num_titles, p=[0.32, 0.15, 0.1, 0.1, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01]), # Sum is now 1.0
        'release_year': np.random.randint(2000, 2024, num_titles),
        'rating': np.random.choice([
            'G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'UR'
        ], num_titles, p=[0.05, 0.08, 0.12, 0.15, 0.02, 0.05, 0.06, 0.07, 0.1, 0.1, 0.18, 0.02]),
        'duration': durations_list,
        'listed_in': [', '.join(np.random.choice([
            'Dramas', 'Comedies', 'Action & Adventure', 'Horror Movies', 'Thrillers', 'Sci-Fi & Fantasy',
            'International Movies', 'Independent Movies', 'Romantic Movies', 'Documentaries', 'Crime Movies',
            'Kids\' TV', 'TV Dramas', 'TV Comedies', 'Reality TV', 'Anime Series', 'Spanish-Language TV Shows',
            'Korean TV Shows', 'British TV Shows', 'TV Action & Adventure', 'Classic Movies', 'Cult Movies',
            'LGBTQ Movies', 'Music & Musicals', 'Sports Movies', 'Faith & Spirituality', 'Teen TV Shows',
            'Romantic TV Shows', 'Science & Nature TV', 'Stand-Up Comedy', 'Talk Shows', 'Variety & Game Shows'
        ], size=np.random.randint(1, 4), replace=False)) for _ in range(num_titles)],
        'imdb_score': np.clip(np.random.normal(6.5, 1.2, num_titles), 2.0, 9.5).round(1),
        'date_added': pd.to_datetime([
            f'{np.random.randint(2015, 2024)}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}'
            for _ in range(num_titles)
        ]),
        'budget_millions': np.random.uniform(1, 300, num_titles).round(1),
        'views_millions': np.random.uniform(1, 800, num_titles).round(1),
        'awards_won': np.random.randint(0, 50, num_titles),  # Awards won
        'nomination_for_best_picture': np.random.choice([0, 1], num_titles, p=[0.9, 0.1]), #Binary: 0 for No, 1 for Yes
        'nominations': np.random.randint(0, 100, num_titles), # Number of nominations        
        'language': np.random.choice(['English', 'Spanish', 'French', 'German', 'Korean', 'Japanese', 'Hindi', 'Italian', 'Mandarin', 'Portuguese'], num_titles, p=[0.6, 0.1, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.03]),  # Language, sum is now 1.0
        'aspect_ratio': np.random.choice(['16:9', '2.39:1', '4:3', '1.85:1'], num_titles), # Aspect Ratio
        'sound_mix': np.random.choice(['Dolby Digital', 'Dolby Atmos', 'Stereo', 'Mono'], num_titles), # Sound Mix
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
with st.expander("📊 Tool 1: Content Performance Analytics"):
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
                     title="Genre Popularity Trends Over Time",
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Tool 3: Geographic Content Distribution
with st.expander("🌍 Tool 3: Geographic Content Distribution"):
    if 'country' in df.columns:
        country_data = df['country'].value_counts().head(10)
        fig = px.bar(country_data, x=country_data.values, y=country_data.index, orientation='h',
                    title="Content Production by Country", labels={'x': 'Number of Titles', 'y': 'Country'},
                    template="plotly_dark")
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
with st.expander("🏆 Tool 5: Rating Distribution Analysis"):
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
with st.expander("📅 Tool 6: Release Year Timeline"):
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
with st.expander("💰 Tool 7: Budget vs Performance ROI"):
    if 'budget_millions' in df.columns and 'views_millions' in df.columns:
        df['roi'] = df['views_millions'] / df['budget_millions']
        
        fig = px.scatter(df, x='budget_millions', y='roi', color='type', size='imdb_score',
                        title="Budget vs ROI Analysis",
                        labels={'budget_millions': 'Budget (Millions)', 'roi': 'ROI (Views/Budget)'},
                        template="plotly_dark")
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
                       color_continuous_scale='RdBu_r', text_auto=True,
                       template="plotly_dark")
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
                         xaxis_title="Actual Views", yaxis_title="Predicted Views",
                         template="plotly_dark")
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

# Tool 17: Title Word Cloud
with st.expander("☁️ Tool 17: Title Word Cloud"):
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

# Tool 18: Content Type Evolution Over Time
with st.expander("🔄 Tool 18: Content Type Evolution Over Time"):
    if 'release_year' in df.columns and 'type' in df.columns:
        content_type_evolution = df.groupby(['release_year', 'type']).size().reset_index(name='count')
        fig = px.line(content_type_evolution, x='release_year', y='count', color='type',
                     title="Content Type Releases Over Time", 
                     labels={'release_year': 'Release Year', 'count': 'Number of Titles'},
                     template="plotly_dark")
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
                                    title="Number of Titles Produced by Country",
                                    template="plotly_dark")
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
                                    labels={'x': 'Month Added', 'y': 'Number of Titles'},
                                    template="plotly_dark")
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

# Tool 29: Time Series Analysis of IMDb Scores
with st.expander("📈 Tool 29: IMDb Score Trends Over Release Years"):
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

# Tool 31: Content Length vs. IMDb Score
with st.expander("📏 Tool 31: Content Length vs. IMDb Score"):
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

# Tool 32: Genre Co-occurrence Analysis
with st.expander("🤝 Tool 32: Genre Co-occurrence Analysis"):
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

# Tool 33: Cast Size vs. Performance
with st.expander("👥 Tool 33: Cast Size vs. Performance"):
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

# Tool 34: Content Addition Trend (Yearly)
with st.expander("📅 Tool 34: Content Addition Trend (Yearly)"):
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

# Tool 35: Description Keyword Analysis
with st.expander("📝 Tool 35: Description Keyword Analysis"):
    if 'description' in df.columns:
        st.subheader("Most Frequent Keywords in Descriptions")
        
        df_tool35 = df.copy()
        df_tool35.dropna(subset=['description'], inplace=True)

        if not df_tool35.empty:
            # Simple tokenization and cleaning
            text = " ".join(df_tool35['description'].str.lower().tolist())
            words = re.findall(r'\b\w+\b', text) # Extract words
            
            # Basic stop words (can be expanded)
            stop_words = set([
                'the', 'a', 'an', 'is', 'it', 'in', 'of', 'for', 'with', 'and', 'to', 'on', 'by', 'about',
                'from', 'as', 'at', 'be', 'this', 'that', 'have', 'has', 'he', 'she', 'it', 'they', 'we',
                'you', 'his', 'her', 'its', 'their', 'our', 'will', 'can', 'just', 'get', 'when', 'where',
                'who', 'what', 'how', 'which', 'if', 'or', 'not', 'no', 'yes', 'out', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
                'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
                'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'film', 'series', 'story', 'life', 'world',
                'new', 'young', 'man', 'woman', 'family', 'find', 'take', 'make', 'come', 'go', 'back',
                'two', 'one', 'time', 'show', 'tv', 'movie', 'about', 'their', 'into', 'through', 'after',
                'before', 'during', 'below', 'above', 'between', 'among', 'across', 'behind', 'beside',
                'down', 'into', 'off', 'out', 'over', 'under', 'up', 'with', 'within', 'without', 'throughout'
            ])
            
            # Filter out stop words and short words
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            if filtered_words:
                word_counts = Counter(filtered_words).most_common(20)
                word_counts_df = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency'])

                fig_keywords = px.bar(word_counts_df, y='Keyword', x='Frequency',
                                      orientation='h', title="Top 20 Most Frequent Keywords in Descriptions",
                                      labels={'Frequency': 'Number of Occurrences'},
                                      template="plotly_dark")
                fig_keywords.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_keywords, use_container_width=True)
            else:
                st.write("No significant keywords found in descriptions after filtering.")
        else:
            st.write("No 'description' data available for analysis.")
    else:
        st.info("'description' column not available for keyword analysis.")

st.markdown("---")
st.markdown("**Netflix Data Analytics Dashboard** - Comprehensive toolkit for data analysis capstone projects")
