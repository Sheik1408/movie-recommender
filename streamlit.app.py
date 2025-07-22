# app.py

import streamlit as st
import pandas as pd
from recommender import recommend

# Load movie data for dropdown
movies = pd.read_csv("movies.csv.csv")

st.title("🎬 Movie Recommendation System")
st.markdown("Get similar movie suggestions based on overview content!")

movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    with st.spinner("Finding movies..."):
        recommendations = recommend(selected_movie)
        st.success("Top Recommendations:")
        for i, title in enumerate(recommendations, 1):
            st.write(f"{i}. {title}")


