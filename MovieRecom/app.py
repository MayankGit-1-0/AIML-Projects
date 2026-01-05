import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Settings
# -----------------------------
DATA_PATH = "imdb_top_1000.csv"
EMBEDDINGS_PATH = "movie_embeddings.npy"
TITLES_PATH = "movie_titles.npy"

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_embeddings():
    return np.load(EMBEDDINGS_PATH)

@st.cache_data
def load_titles():
    with open("movie_titles.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


# -----------------------------
# App UI
# -----------------------------
st.title("🎬 IMDb Top 1000 Movie Recommender")
st.write("Get recommendations based on movie overview similarity.")

df = load_data()
titles = load_titles()
embeddings = load_embeddings()

# -----------------------------
# Movie selection
# -----------------------------
movie_name = st.selectbox("Choose a movie:", titles)

movie_index = list(titles).index(movie_name)

# -----------------------------
# Recommend
# -----------------------------
if st.button("Recommend Similar Movies"):
    st.subheader(f"Selected Movie: {movie_name}")

    # compute similarity
    sims = cosine_similarity(
        embeddings[movie_index].reshape(1, -1),
        embeddings
    )[0]

    indices = sims.argsort()[-6:-1][::-1]  # top 5

    st.subheader("Recommended Movies:")
    for idx in indices:
        st.write(f"🎬 **{titles[idx]}**")
        st.write(df.loc[idx, "Overview"])
        st.write("---")
