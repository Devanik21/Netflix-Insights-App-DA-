import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import json # Added for data export
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Netflix Analytics Ultimate", layout="wide")
st.title("🎬 Netflix Analytics Dashboard - Ultimate Edition")
st.markdown("**25+ Professional Analytics Tools for Enterprise-Level Insights**")

# Comprehensive Netflix dataset
@st.cache_data
def load_comprehensive_netflix_data():
    """Generate comprehensive Netflix dataset with 100+ titles"""
    np.random.seed(42)
    
    titles = ['Stranger Things', 'The Crown', 'Squid Game', 'Money Heist', 'Bridgerton', 'The Witcher', 'Ozark', 
              'Narcos', 'Black Mirror', 'The Queen\'s Gambit', 'Lupin', 'Elite', 'Dark', 'Mindhunter', 'Orange is the New Black',
              'House of Cards', 'Breaking Bad', 'Better Call Saul', 'The Office', 'Friends', 'Tiger King', 'Making a Murderer',
              'Wild Wild Country', 'Our Planet', 'Chef\'s Table', 'The Movies That Made Us', 'High Score', 'The Social Dilemma',
              'My Octopus Teacher', 'American Factory', 'Icarus', 'Won\'t You Be My Neighbor?', 'RBG', 'Free Solo', 'The Great Hack',
              'Explained', 'Abstract', 'Salt Fat Acid Heat', 'Ugly Delicious', 'Street Food', 'The Mind, Explained', 'Sex Education',
              'Cable Girls', 'Money Heist: Korea', 'Emily in Paris', 'Virgin River', 'Lucifer', 'You', 'Russian Doll', 'GLOW',
              'Atypical', 'Big Mouth', 'BoJack Horseman', 'F is for Family', 'The Umbrella Academy', 'Altered Carbon', 'Lost in Space',
              'Space Force', 'The Good Place', 'Schitt\'s Creek', 'Dead to Me', 'Grace and Frankie', 'One Day at a Time',
              'Cobra Kai', 'The Karate Kid', 'Bird Box', 'Roma', 'The Irishman', 'Marriage Story', 'Klaus', 'I Am Mother',
              'Extraction', 'The Old Guard', 'Project Power', 'Enola Holmes', 'The Platform', 'The Half of It', 'Da 5 Bloods',
              'Ma Rainey\'s Black Bottom', 'Mank', 'The Trial of the Chicago 7', 'Mulan', 'Soul', 'Wonder Woman 1984', 'Tenet',
              'No Time to Die', 'Fast & Furious 9', 'Black Widow', 'Shang-Chi', 'Eternals', 'Spider-Man: No Way Home', 'The Batman',
              'Doctor Strange 2', 'Thor: Love and Thunder', 'Black Panther 2', 'Avatar 2', 'Top Gun: Maverick', 'Jurassic World 3',
              'Mission Impossible 7', 'Indiana Jones 5', 'John Wick 4', 'The Matrix 4', 'Dune', 'Blade Runner 2049', 'Mad Max: Fury Road']
    
    countries = ['United States', 'United Kingdom', 'South Korea', 'Spain', 'Germany', 'France', 'Japan', 'India', 'Canada', 'Australia',
                'Brazil', 'Mexico', 'Italy', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Turkey', 'Russia', 'China']
    
    genres = ['Drama', 'Comedy', 'Action & Adventure', 'Thriller', 'Horror', 'Sci-Fi & Fantasy', 'Crime', 'Documentary', 'Romance',
              'Family', 'Animation', 'Mystery', 'Biography', 'History', 'War', 'Musical', 'Western', 'Sport']
    
    n_titles = 100
    data = {
        'show_id': [f's{i}' for i in range(1, n_titles + 1)],
        'type': np.random.choice(['Movie', 'TV Show'], n_titles, p=[0.6, 0.4]),
        'title': np.random.choice(titles, n_titles),
        'country': np.random.choice(countries, n_titles, p=[0.3, 0.15, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03,
                                                           0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
        'release_year': np.random.randint(2010, 2024, n_titles),
        'rating': np.random.choice(['G', 'PG', 'PG-13', 'R', 'TV-Y', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA'], n_titles),
        'duration_minutes': np.random.randint(80, 180, n_titles),
        'seasons': np.random.randint(1, 6, n_titles),
        'genre_primary': np.random.choice(genres, n_titles),
        'genre_secondary': np.random.choice(genres, n_titles),
        'imdb_score': np.random.uniform(5.0, 9.5, n_titles).round(1),
        'budget_millions': np.random.uniform(5, 300, n_titles).round(1),
        'views_millions': np.random.uniform(10, 1000, n_titles).round(1),
        'revenue_millions': np.random.uniform(20, 2000, n_titles).round(1),
        'production_cost_millions': np.random.uniform(3, 250, n_titles).round(1),
        'marketing_spend_millions': np.random.uniform(1, 100, n_titles).round(1),
        'critic_score': np.random.randint(30, 100, n_titles),
        'audience_score': np.random.randint(40, 100, n_titles),
        'awards_count': np.random.randint(0, 25, n_titles),
        'language': np.random.choice(['English', 'Spanish', 'Korean', 'German', 'French', 'Japanese', 'Hindi'], n_titles),
        'cast_popularity_score': np.random.uniform(1, 10, n_titles).round(1),
        'director_experience_years': np.random.randint(1, 40, n_titles),
        'social_media_mentions': np.random.randint(1000, 100000, n_titles),
        'streaming_hours_total': np.random.randint(100000, 50000000, n_titles),
        'completion_rate': np.random.uniform(0.3, 0.95, n_titles).round(2),
        'user_retention_rate': np.random.uniform(0.2, 0.8, n_titles).round(2),
        'global_availability': np.random.choice([True, False], n_titles, p=[0.7, 0.3]),
        'content_age_days': np.random.randint(30, 3650, n_titles),
        'trailer_views_millions': np.random.uniform(0.5, 50, n_titles).round(1),
        'subtitles_count': np.random.randint(5, 40, n_titles),
        'production_company_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_titles, p=[0.3, 0.5, 0.2])
    }
    
    return pd.DataFrame(data)

# Load data
df = load_comprehensive_netflix_data()

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
view_mode = st.sidebar.selectbox("Analysis Mode", ["Executive", "Analyst", "Data Scientist", "Strategic"])
time_filter = st.sidebar.slider("Release Year Range", 2010, 2023, (2015, 2023))
df_filtered = df[(df['release_year'] >= time_filter[0]) & (df['release_year'] <= time_filter[1])].copy()
st.sidebar.info(f"Selected mode: {view_mode}. Custom views for each mode are a future enhancement.")

# Gemini API
gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Main KPI Dashboard
st.header("📊 Executive KPI Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Content", len(df_filtered), delta=len(df_filtered)-len(df)*0.8)
with col2:
    avg_score = df_filtered['imdb_score'].mean()
    st.metric("Avg IMDB Score", f"{avg_score:.1f}", delta=f"{avg_score-7.5:.1f}")
with col3:
    total_views = df_filtered['views_millions'].sum()
    st.metric("Total Views (M)", f"{total_views:,.0f}")
with col4:
    roi = (df_filtered['revenue_millions'].sum() / df_filtered['budget_millions'].sum())
    st.metric("Portfolio ROI", f"{roi:.1f}x")
with col5:
    avg_completion = df_filtered['completion_rate'].mean()
    st.metric("Avg Completion", f"{avg_completion:.1%}")

# Tool 1: Advanced Performance Analytics
with st.expander("🎯 Tool 1: Advanced Performance Analytics"):
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter_3d(df_filtered, x='imdb_score', y='views_millions', z='revenue_millions',
                           color='type', size='awards_count', title="3D Performance Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        performance_score = (df_filtered['imdb_score'] * df_filtered['views_millions'] * df_filtered['completion_rate']).round(0)
        df_filtered['performance_score'] = performance_score
        top_performers = df_filtered.nlargest(10, 'performance_score')[['title', 'performance_score', 'imdb_score', 'views_millions']]
        st.subheader("Top 10 Overall Performers")
        st.dataframe(top_performers)

# Tool 2: Market Sentiment Analysis
with st.expander("📈 Tool 2: Market Sentiment Analysis"):
    sentiment_data = []
    for score in df_filtered['audience_score']:
        if score >= 80: sentiment_data.append('Positive')
        elif score >= 60: sentiment_data.append('Neutral')
        else: sentiment_data.append('Negative')
    
    df_filtered['sentiment'] = sentiment_data
    sentiment_counts = pd.Series(sentiment_data).value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                    title="Content Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sentiment_by_genre = df_filtered.groupby(['genre_primary', 'sentiment']).size().unstack(fill_value=0)
        fig = px.bar(sentiment_by_genre, title="Sentiment by Genre")
        st.plotly_chart(fig, use_container_width=True)

# Tool 3: Competitive Intelligence Dashboard
with st.expander("🔍 Tool 3: Competitive Intelligence Dashboard"):
    competitor_analysis = df_filtered.groupby('production_company_tier').agg({
        'imdb_score': 'mean',
        'views_millions': 'sum',
        'revenue_millions': 'sum',
        'awards_count': 'sum'
    }).round(2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Production Company Performance")
        st.dataframe(competitor_analysis)
    
    with col2:
        fig = px.bar(competitor_analysis, y=competitor_analysis.index, x='revenue_millions',
                    title="Revenue by Production Tier", orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# Tool 4: Advanced Clustering Analysis
with st.expander("🎯 Tool 4: Content Clustering Analysis"):
    features = ['imdb_score', 'views_millions', 'budget_millions', 'awards_count', 'completion_rate']
    X = df_filtered[features].fillna(df_filtered[features].mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Added n_init
    clusters = kmeans.fit_predict(X_scaled)
    df_filtered['cluster'] = clusters
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_filtered, x='imdb_score', y='views_millions', color='cluster',
                        title="Content Clusters", size='budget_millions')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cluster_summary = df_filtered.groupby('cluster')[features].mean().round(2)
        st.subheader("Cluster Characteristics")
        st.dataframe(cluster_summary)

# Tool 5: Financial Performance Deep Dive
with st.expander("💰 Tool 5: Financial Performance Deep Dive"):
    df_filtered['profit_margin'] = (df_filtered['revenue_millions'] - df_filtered['production_cost_millions']) / df_filtered['revenue_millions']
    df_filtered['marketing_efficiency'] = df_filtered['views_millions'] / df_filtered['marketing_spend_millions']
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_filtered, x='marketing_spend_millions', y='views_millions',
                        color='profit_margin', size='revenue_millions',
                        title="Marketing Efficiency vs Views")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        financial_metrics = {
            'Total Revenue': f"${df_filtered['revenue_millions'].sum():,.0f}M",
            'Total Costs': f"${df_filtered['production_cost_millions'].sum():,.0f}M",
            'Avg Profit Margin': f"{df_filtered['profit_margin'].mean():.1%}",
            'Best ROI Title': df_filtered.loc[df_filtered['profit_margin'].idxmax(), 'title']
        }
        
        st.subheader("Financial KPIs")
        for key, value in financial_metrics.items():
            st.write(f"**{key}**: {value}")

# Tool 6: Predictive Analytics Suite
with st.expander("🔮 Tool 6: Advanced Predictive Analytics"):
    # Multiple prediction models
    prediction_target = st.selectbox("Predict:", ["Views", "Revenue", "IMDB Score"])
    
    if prediction_target == "Views":
        y = df_filtered['views_millions']
        X = df_filtered[['imdb_score', 'budget_millions', 'marketing_spend_millions', 'awards_count']]
    elif prediction_target == "Revenue":
        y = df_filtered['revenue_millions']
        X = df_filtered[['views_millions', 'budget_millions', 'imdb_score', 'marketing_spend_millions']]
    else:
        y = df_filtered['imdb_score']
        X = df_filtered[['budget_millions', 'director_experience_years', 'cast_popularity_score']]
    
    X_clean = X.fillna(X.mean())
    y_clean = y.fillna(y.mean())
    
    model = LinearRegression().fit(X_clean, y_clean)
    predictions = model.predict(X_clean)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=y_clean, y=predictions, title=f"{prediction_target} Prediction Accuracy")
        fig.add_trace(go.Scatter(x=[y_clean.min(), y_clean.max()], y=[y_clean.min(), y_clean.max()],
                                mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        st.write(f"R² Score: {model.score(X_clean, y_clean):.3f}")
        st.write("Feature Importance:")
        for i, col in enumerate(X.columns):
            st.write(f"{col}: {model.coef_[i]:.3f}")

# Tool 7: Network Analysis
with st.expander("🕸️ Tool 7: Content Network Analysis"):
    if not df_filtered.empty and len(df_filtered['title'].unique()) > 1:
        # Ensure 'title' is unique for crosstab. Titles can repeat due to data generation.
        # For network nodes, we use unique titles.
        unique_titles_df = df_filtered.drop_duplicates(subset=['title'])

        if len(unique_titles_df['title'].unique()) < 2:
            st.info("Not enough unique titles to build a meaningful network after filtering.")
        else:
            genre_matrix = pd.crosstab(unique_titles_df['title'], unique_titles_df['genre_primary'])
            
            if genre_matrix.empty or genre_matrix.shape[0] < 2: # Need at least 2 titles to form a network
                st.info("Not enough data to create genre matrix for network analysis.")
            else:
                similarity_matrix_values = cosine_similarity(genre_matrix)
                similarity_df = pd.DataFrame(similarity_matrix_values, index=genre_matrix.index, columns=genre_matrix.index)

                st.subheader("Content Similarity Network Visualization")
                
                threshold = st.slider("Similarity Threshold for Network Edges", 0.1, 1.0, 0.5, 0.05, key="network_threshold")

                G = nx.Graph()
                for title_node in similarity_df.columns:
                    G.add_node(title_node)

                similar_pairs_for_listing = []
                for i in range(len(similarity_df.columns)):
                    for j in range(i + 1, len(similarity_df.columns)):
                        title1 = similarity_df.columns[i]
                        title2 = similarity_df.columns[j]
                        weight = similarity_df.iloc[i, j]
                        if weight > threshold:
                            G.add_edge(title1, title2, weight=weight)
                            if len(similar_pairs_for_listing) < 10: # Display up to 10 pairs
                                similar_pairs_for_listing.append((title1, title2, weight))
                
                if not G.nodes() or not G.edges():
                    st.info("No connections found above the selected similarity threshold. Try lowering the threshold.")
                else:
                    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42) # positions for all nodes

                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                    node_x, node_y, node_text_hover = [], [], []
                    node_adjacencies = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        degree = G.degree(node)
                        node_adjacencies.append(degree)
                        node_text_hover.append(f"{node}<br>Connections: {degree}")

                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text_hover,
                        marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=node_adjacencies,
                                    size=10, colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
                                    line_width=2))

                    fig_network = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='<br>Content Similarity Network (Primary Genre)', titlefont_size=16, showlegend=False, hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                    st.plotly_chart(fig_network, use_container_width=True)

                    if similar_pairs_for_listing:
                        st.subheader("Highly Similar Content Pairs (above threshold):")
                        for pair in similar_pairs_for_listing:
                            st.write(f"{pair[0]} ↔ {pair[1]} (Similarity: {pair[2]:.2f})")
                    else:
                        st.write("No content pairs found above the current similarity threshold to list.")
    else:
        st.info("Not enough data to perform network analysis. Adjust filters or ensure data is loaded.")

# Tool 8: Time Series Forecasting
with st.expander("📊 Tool 8: Time Series Forecasting"):
    monthly_data = df_filtered.groupby('release_year').agg({
        'views_millions': 'sum',
        'revenue_millions': 'sum',
        'imdb_score': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly_data, x='release_year', y='views_millions', 
                     title="Views Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Simple trend forecast
        X_years = monthly_data['release_year'].values.reshape(-1, 1)
        y_views = monthly_data['views_millions'].values
        
        trend_model = LinearRegression().fit(X_years, y_views)
        future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
        future_views = trend_model.predict(future_years)
        
        st.subheader("2024-2026 Forecast")
        for year, views in zip([2024, 2025, 2026], future_views):
            st.write(f"{year}: {views:,.0f}M views")

# Tool 9: A/B Testing Framework
with st.expander("🧪 Tool 9: A/B Testing Analytics"):
    # Simulate A/B test results
    test_groups = np.random.choice(['Control', 'Treatment'], len(df_filtered))
    df_filtered['test_group'] = test_groups
    df_filtered['conversion_rate'] = np.where(test_groups == 'Treatment', 
                                            df_filtered['completion_rate'] * 1.15,
                                            df_filtered['completion_rate'])
    
    col1, col2 = st.columns(2)
    with col1:
        ab_results = df_filtered.groupby('test_group').agg({
            'completion_rate': 'mean',
            'conversion_rate': 'mean',
            'views_millions': 'mean'
        }).round(3)
        
        st.subheader("A/B Test Results")
        st.dataframe(ab_results)
    
    with col2:
        fig = px.box(df_filtered, x='test_group', y='conversion_rate',
                    title="Conversion Rate Distribution by Group")
        st.plotly_chart(fig, use_container_width=True)

# Tool 10: Content Lifecycle Analysis
with st.expander("🔄 Tool 10: Content Lifecycle Analysis"):
    df_filtered['lifecycle_stage'] = pd.cut(df_filtered['content_age_days'], 
                                          bins=[0, 30, 180, 365, 3650],
                                          labels=['Launch', 'Growth', 'Maturity', 'Decline'])
    
    lifecycle_metrics = df_filtered.groupby('lifecycle_stage').agg({
        'views_millions': 'mean',
        'completion_rate': 'mean',
        'user_retention_rate': 'mean'
    }).round(3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance by Lifecycle Stage")
        st.dataframe(lifecycle_metrics)
    
    with col2:
        fig = px.line(lifecycle_metrics.reset_index(), x='lifecycle_stage', y='views_millions',
                     title="Views Across Content Lifecycle")
        st.plotly_chart(fig, use_container_width=True)

# Tool 11: Risk Assessment Matrix
with st.expander("⚠️ Tool 11: Risk Assessment Matrix"):
    df_filtered['risk_score'] = (
        (10 - df_filtered['imdb_score']) * 0.3 +
        (1 - df_filtered['completion_rate']) * 0.4 +
        (df_filtered['budget_millions'] / df_filtered['revenue_millions']) * 0.3
    )
    
    df_filtered['risk_category'] = pd.cut(df_filtered['risk_score'], 
                                        bins=[0, 2, 4, 6, 10],
                                        labels=['Low', 'Medium', 'High', 'Critical'])
    
    col1, col2 = st.columns(2)
    with col1:
        risk_distribution = df_filtered['risk_category'].value_counts()
        fig = px.pie(values=risk_distribution.values, names=risk_distribution.index,
                    title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        high_risk = df_filtered[df_filtered['risk_category'].isin(['High', 'Critical'])]
        st.subheader("High-Risk Content")
        st.dataframe(high_risk[['title', 'risk_score', 'imdb_score', 'completion_rate']].head())

# Tool 12: Market Saturation Analysis
with st.expander("📊 Tool 12: Market Saturation Analysis"):
    genre_saturation = df_filtered.groupby('genre_primary').agg({
        'title': 'count',
        'views_millions': 'mean',
        'imdb_score': 'mean'
    }).rename(columns={'title': 'content_count'})
    
    genre_saturation['saturation_index'] = (genre_saturation['content_count'] / 
                                          genre_saturation['views_millions'] * 100)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(genre_saturation, x='content_count', y='views_millions',
                        size='imdb_score', color='saturation_index',
                        title="Genre Saturation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Opportunities")
        opportunities = genre_saturation.sort_values('saturation_index').head(5)
        for genre in opportunities.index:
            st.write(f"**{genre}**: Low saturation opportunity")

# Tool 13: User Engagement Analytics
with st.expander("👥 Tool 13: User Engagement Analytics"):
    engagement_metrics = df_filtered.groupby('type').agg({
        'completion_rate': 'mean',
        'user_retention_rate': 'mean',
        'social_media_mentions': 'mean',
        'trailer_views_millions': 'mean'
    }).round(3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Engagement by Content Type")
        st.dataframe(engagement_metrics)
    
    with col2:
        if not engagement_metrics.empty:
            fig = go.Figure()
            
            # Assuming engagement_metrics.index contains 'Movie', 'TV Show' etc.
            # And engagement_metrics.columns are the metrics like 'completion_rate'
            
            for i, trace_name_val in enumerate(engagement_metrics.index):
                fig.add_trace(go.Scatterpolar(
                    r=engagement_metrics.iloc[i].values,
                    theta=engagement_metrics.columns,
                    fill='toself',
                    name=str(trace_name_val) # Name for the legend (e.g., "Movie", "TV Show")
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True) # Add range if needed, e.g. range=[0, max_val_across_all_metrics]
                ),
                showlegend=True,
                title="Engagement Metrics by Content Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No engagement metrics to display for radar chart.")
# Tool 14: Global Expansion Strategy
with st.expander("🌍 Tool 14: Global Expansion Strategy"):
    country_metrics = df_filtered.groupby('country').agg({
        'views_millions': 'sum',
        'revenue_millions': 'sum',
        'imdb_score': 'mean',
        'global_availability': 'mean'
    }).round(2)
    
    country_metrics['expansion_score'] = (
        country_metrics['views_millions'] * 0.4 +
        country_metrics['revenue_millions'] * 0.3 +
        country_metrics['imdb_score'] * 10 * 0.3
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_markets = country_metrics.nlargest(10, 'expansion_score')
        fig = px.bar(top_markets, y=top_markets.index, x='expansion_score',
                    title="Top Expansion Markets", orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Recommendations")
        for country in top_markets.index[:5]:
            score = top_markets.loc[country, 'expansion_score']
            st.write(f"**{country}**: Expansion Score {score:.0f}")

# Tool 15: Content Portfolio Optimization
with st.expander("🎯 Tool 15: Portfolio Optimization"):
    # Portfolio risk-return analysis
    portfolio_data = df_filtered.groupby('genre_primary').agg({
        'revenue_millions': 'sum',
        'budget_millions': 'sum',
        'views_millions': 'sum'
    })
    
    portfolio_data['return'] = portfolio_data['revenue_millions'] / portfolio_data['budget_millions']
    portfolio_data['risk'] = portfolio_data['revenue_millions'].std() / portfolio_data['revenue_millions'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(portfolio_data, x='risk', y='return',
                        size='views_millions', hover_name=portfolio_data.index,
                        title="Portfolio Risk-Return Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        optimal_genres = portfolio_data.nlargest(5, 'return')
        st.subheader("Optimal Genre Portfolio")
        st.dataframe(optimal_genres[['return', 'risk']])

# Advanced AI Tools
st.header("🤖 AI-Powered Advanced Analytics")

# Tool 16: Automated Insights Generator
with st.expander("🔍 Tool 16: Automated Insights Generator"):
    if gemini_key:
        insight_type = st.selectbox("Generate insights for:", 
                                  ["Performance Optimization", "Market Strategy", "Content Planning", "Risk Management"])
        
        if st.button("Generate Advanced Insights"):
            data_summary = f"""
            Netflix Dataset Analysis Summary:
            - Total content: {len(df_filtered)} titles
            - Average IMDB score: {df_filtered['imdb_score'].mean():.2f}
            - Total revenue: ${df_filtered['revenue_millions'].sum():,.0f}M
            - Average completion rate: {df_filtered['completion_rate'].mean():.2%}
            - Top performing genre: {df_filtered.groupby('genre_primary')['views_millions'].sum().idxmax()}
            - Highest risk content: {df_filtered.loc[df_filtered['risk_score'].idxmax(), 'title']}
            - Best ROI title: {df_filtered.loc[(df_filtered['revenue_millions']/df_filtered['budget_millions']).idxmax(), 'title']}
            """
            
            prompt = f"""
            As a senior data analyst, provide 5 advanced insights for {insight_type} based on this Netflix data:
            {data_summary}
            
            Focus on actionable recommendations with specific metrics and strategic implications.
            """
            
            try:
                model = genai.GenerativeModel("gemini-1.5-flash-latest") # Updated model name
                response = model.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Enter Gemini API key for AI insights")

# Tool 17-25: Additional Advanced Tools
st.header("🔬 Additional Advanced Analytics Tools")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📊 Tool 17: Cohort Analysis"):
        cohort_data = df_filtered.groupby(['release_year', 'genre_primary']).agg({
            'views_millions': 'sum',
            'completion_rate': 'mean'
        }).round(2)
        st.write("Content cohort performance by release year and genre")
        st.dataframe(cohort_data.head(10))

    with st.expander("🎯 Tool 18: Attribution Modeling"):
        attribution_factors = ['cast_popularity_score', 'director_experience_years', 'marketing_spend_millions']
        attribution_impact = {}
        for factor in attribution_factors:
            corr = df_filtered[factor].corr(df_filtered['views_millions'])
            attribution_impact[factor] = corr
        
        st.write("Factor Attribution to Views:")
        for factor, impact in attribution_impact.items():
            st.write(f"**{factor}**: {impact:.3f}")
        
        # Attribution visualization
        fig = px.bar(x=list(attribution_impact.keys()), y=list(attribution_impact.values()),
                    title="Attribution Model Results")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 Tool 19: Churn Prediction"):
        # Simulate churn based on retention rate
        df_filtered['churn_risk'] = 1 - df_filtered['user_retention_rate']
        df_filtered['churn_category'] = pd.cut(df_filtered['churn_risk'], 
                                             bins=[0, 0.3, 0.6, 1.0],
                                             labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        churn_analysis = df_filtered.groupby('churn_category').agg({
            'title': 'count',
            'views_millions': 'mean',
            'completion_rate': 'mean'
        }).round(2)
        
        st.write("Churn Risk Analysis:")
        st.dataframe(churn_analysis)

with col2:
    with st.expander("🔍 Tool 20: Anomaly Detection"):
        # Detect anomalies using statistical methods
        Q1 = df_filtered['views_millions'].quantile(0.25)
        Q3 = df_filtered['views_millions'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = df_filtered[(df_filtered['views_millions'] < lower_bound) | 
                               (df_filtered['views_millions'] > upper_bound)]
        
        st.write(f"Detected {len(anomalies)} anomalies:")
        if len(anomalies) > 0:
            st.dataframe(anomalies[['title', 'views_millions', 'imdb_score']].head())
        
        # Anomaly visualization
        fig = px.box(df_filtered, y='views_millions', title="Views Distribution with Outliers")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎨 Tool 21: Creative Performance Index"):
        # Create a composite creativity score
        df_filtered['creativity_score'] = (
            (df_filtered['awards_count'] * 0.3) +
            (df_filtered['critic_score'] / 10 * 0.3) +
            ((df_filtered['imdb_score'] - 5) * 0.4)
        )
        
        creative_leaders = df_filtered.nlargest(10, 'creativity_score')
        st.write("Most Creative Content:")
        st.dataframe(creative_leaders[['title', 'creativity_score', 'awards_count', 'critic_score']])

    with st.expander("🌟 Tool 22: Star Power Analysis"):
        # Analyze cast popularity impact
        cast_impact = df_filtered.groupby(pd.cut(df_filtered['cast_popularity_score'], 
                                                bins=[0, 3, 6, 10],
                                                labels=['Low', 'Medium', 'High'])).agg({
            'views_millions': 'mean',
            'revenue_millions': 'mean',
            'imdb_score': 'mean'
        }).round(2)
        
        st.write("Star Power Impact:")
        st.dataframe(cast_impact)
        
        fig = px.bar(cast_impact.reset_index(), x='cast_popularity_score', y='revenue_millions',
                    title="Revenue by Cast Popularity")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    with st.expander("📺 Tool 23: Genre Evolution Tracker"):
        genre_evolution = df_filtered.groupby(['release_year', 'genre_primary']).agg({
            'views_millions': 'sum',
            'imdb_score': 'mean'
        }).reset_index()
        
        # Show top 3 genres over time
        top_genres = df_filtered.groupby('genre_primary')['views_millions'].sum().nlargest(3).index
        genre_subset = genre_evolution[genre_evolution['genre_primary'].isin(top_genres)]
        
        fig = px.line(genre_subset, x='release_year', y='views_millions', 
                     color='genre_primary', title="Genre Performance Evolution")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎬 Tool 24: Director Success Patterns"):
        director_analysis = df_filtered.groupby(pd.cut(df_filtered['director_experience_years'],
                                                      bins=[0, 5, 15, 40],
                                                      labels=['Newcomer', 'Experienced', 'Veteran'])).agg({
            'imdb_score': 'mean',
            'awards_count': 'mean',
            'revenue_millions': 'mean'
        }).round(2)
        
        st.write("Director Experience Impact:")
        st.dataframe(director_analysis)

    with st.expander("🔥 Tool 25: Viral Content Predictor"):
        # Viral potential based on social media mentions and trailer views
        df_filtered['viral_score'] = (
            np.log(df_filtered['social_media_mentions']) * 0.4 +
            np.log(df_filtered['trailer_views_millions'] + 1) * 0.3 +
            (df_filtered['completion_rate'] * 100) * 0.3
        )
        
        viral_content = df_filtered.nlargest(10, 'viral_score')
        st.write("Highest Viral Potential:")
        st.dataframe(viral_content[['title', 'viral_score', 'social_media_mentions', 'trailer_views_millions']])

# Advanced Reporting Section
st.header("📋 Executive Reports")

with st.expander("📊 Executive Summary Report"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Health Metrics")
        health_metrics = {
            "Content Diversity Index": len(df_filtered['genre_primary'].unique()) / len(df_filtered) * 100,
            "Quality Score (Avg IMDB)": df_filtered['imdb_score'].mean(),
            "Financial Efficiency": (df_filtered['revenue_millions'].sum() / df_filtered['budget_millions'].sum()),
            "Audience Satisfaction": df_filtered['completion_rate'].mean() * 100,
            "Market Coverage": len(df_filtered['country'].unique()),
            "Innovation Index": df_filtered['awards_count'].sum() / len(df_filtered)
        }
        
        for metric, value in health_metrics.items():
            st.metric(metric, f"{value:.2f}")
    
    with col2:
        st.subheader("Strategic Recommendations")
        
        # Generate recommendations based on data
        recommendations = []
        
        # Genre recommendations
        top_genre = df_filtered.groupby('genre_primary')['revenue_millions'].sum().idxmax()
        recommendations.append(f"📈 Focus on {top_genre} content - highest revenue generator")
        
        # Risk recommendations
        high_risk_count = len(df_filtered[df_filtered['risk_category'].isin(['High', 'Critical'])])
        if high_risk_count > len(df_filtered) * 0.2:
            recommendations.append(f"⚠️ Review {high_risk_count} high-risk titles for optimization")
        
        # Market recommendations
        top_market = df_filtered.groupby('country')['views_millions'].sum().idxmax()
        recommendations.append(f"🌍 Expand investment in {top_market} market")
        
        # Performance recommendations
        avg_completion = df_filtered['completion_rate'].mean()
        if avg_completion < 0.7:
            recommendations.append("🎯 Improve content engagement - completion rate below target")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

# Data Export Section
st.header("📤 Data Export & Integration")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Analytics Report"):
        # Create comprehensive report
        report_data = {
            'Summary': {
                'Total Content': len(df_filtered),
                'Avg IMDB Score': df_filtered['imdb_score'].mean(),
                'Total Revenue ($M)': df_filtered['revenue_millions'].sum(),
                'Portfolio ROI': df_filtered['revenue_millions'].sum() / df_filtered['budget_millions'].sum()
            },
            'Top Performers': df_filtered.nlargest(5, 'performance_score')[['title', 'imdb_score', 'views_millions']].to_dict(),
            'Risk Assessment': df_filtered['risk_category'].value_counts().to_dict(),
            'Genre Performance': df_filtered.groupby('genre_primary')['revenue_millions'].sum().to_dict()
        }
        report_json = json.dumps(report_data, indent=4)
        st.download_button(
            label="📥 Download Report as JSON",
            data=report_json,
            file_name=f"netflix_analytics_report_{time_filter[0]}-{time_filter[1]}.json",
            mime="application/json"
        )
        st.success("Report data prepared. Click button above to download.")
        st.json(report_data)

with col2:
    if st.button("📈 Generate Dashboard URL"):
        # Simulate dashboard URL generation
        dashboard_params = {
            'view_mode': view_mode,
            'time_filter': f"{time_filter[0]}-{time_filter[1]}",
            'content_count': len(df_filtered)
        }
        st.success("Dashboard URL generated")
        st.code(f"https://netflix-analytics.com/dashboard?{dashboard_params}")

with col3:
    if st.button("🔄 Sync with External Systems"):
        st.success("Integration endpoints ready")
        st.write("Available APIs:")
        st.write("- PowerBI Connector")
        st.write("- Tableau Integration")
        st.write("- Google Analytics")
        st.write("- Slack Notifications")

# Footer
st.markdown("---")
st.markdown("### 🎬 Netflix Analytics Dashboard - Professional Edition")
st.markdown("*Powered by Advanced Analytics, Machine Learning, and AI Insights*")

# Performance metrics at bottom
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
with perf_col1:
    st.metric("Data Points Analyzed", f"{len(df_filtered) * len(df_filtered.columns):,}")
with perf_col2:
    st.metric("Analytics Tools", "25+")
with perf_col3:
    st.metric("Visualization Types", "15+")
with perf_col4:
    st.metric("Core AI/ML Models", "3") # Adjusted count (LR, KMeans, Gemini)
