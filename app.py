import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

st.set_page_config(page_title="Netflix Content Insights Tool", layout="wide")
st.title("🎬 Netflix Content Insights Tool")
st.markdown("Analyze genre popularity, trends, and gaps across the globe using Netflix data ✨")

# Sidebar
st.sidebar.header("📂 Upload Netflix Dataset")
file = st.sidebar.file_uploader("Upload your Netflix dataset (CSV)", type="csv")

gemini_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
if gemini_key:
    genai.configure(api_key=gemini_key)

# Load Data
if file:
    df = pd.read_csv(file)
    st.success("Dataset loaded successfully!")

    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head())

    with st.expander("📈 Genre Popularity Over Time"):
        if 'release_year' in df.columns and 'listed_in' in df.columns:
            df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
            df = df.dropna(subset=['release_year'])
            df['release_year'] = df['release_year'].astype(int)

            genre_year = df.explode('listed_in').groupby(['release_year'])['listed_in'].value_counts().reset_index(name='count')
            top_genres = genre_year['listed_in'].value_counts().head(5).index
            filtered = genre_year[genre_year['listed_in'].isin(top_genres)]
            fig = px.line(filtered, x='release_year', y='count', color='listed_in', title="Top Genre Trends Over Years")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns: 'release_year' and 'listed_in'")

    with st.expander("🌍 Country-wise Content Gaps"):
        if 'country' in df.columns and 'listed_in' in df.columns:
            country_genre = df.explode('listed_in').groupby(['country'])['listed_in'].value_counts().unstack(fill_value=0)
            st.dataframe(country_genre)
        else:
            st.warning("Required columns: 'country' and 'listed_in'")

    with st.expander("💬 Ask Gemini About Your Data"):
        if gemini_key:
            user_q = st.text_input("Ask a question like 'Which genre is rising in India?' or 'Show thriller trends'")
            if user_q:
                # Compose prompt from data summary
                summary_buf = io.StringIO()
                df.describe(include='all').to_csv(summary_buf)
                prompt = f"""
                Here is a data summary of a Netflix dataset:
                {summary_buf.getvalue()}

                Now answer the question: {user_q}
                """
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    st.markdown("**Gemini Response:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error from Gemini: {e}")
        else:
            st.info("Please enter your Gemini API key in the sidebar to ask questions ✨")
else:
    st.info("Upload a dataset to begin ✨")
