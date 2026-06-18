import pandas as pd
import streamlit as st
from recommender import build_similarity_matrix


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_movies.csv")


@st.cache_data
def get_similarity(data):
    return build_similarity_matrix(data)


def get_recommendations(movie_title, data, similarity, number_of_recommendations=5):
    movie_title = movie_title.lower()

    matching_movies = data[data["title"].str.lower().str.contains(movie_title, na=False)]

    if matching_movies.empty:
        return None

    movie_index = matching_movies.index[0]

    distances = list(enumerate(similarity[movie_index]))
    movies_sorted = sorted(distances, reverse=True, key=lambda x: x[1])

    recommendations = []

    for movie in movies_sorted[1:number_of_recommendations + 1]:
        recommended_movie_index = movie[0]
        recommended_movie_title = data.iloc[recommended_movie_index]["title"]
        similarity_score = movie[1]

        recommendations.append(
            {
                "title": recommended_movie_title,
                "similarity_score": round(similarity_score, 3)
            }
        )

    return recommendations


st.title("Movie Recommendation System")

st.write(
    "Enter a movie title and this app will recommend similar movies "
    "based on genres, keywords, cast, director, and movie overview."
)

data = load_data()
similarity = get_similarity(data)

movie_title = st.text_input("Enter a movie title", placeholder="Example: Avatar")

number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Recommend Movies"):
    if movie_title.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        recommendations = get_recommendations(
            movie_title,
            data,
            similarity,
            number_of_recommendations
        )

        if recommendations is None:
            st.error("Movie not found. Try another title.")
        else:
            st.subheader(f"Movies similar to: {movie_title}")

            for index, movie in enumerate(recommendations, start=1):
                st.write(f"{index}. {movie['title']} — Similarity score: {movie['similarity_score']}")