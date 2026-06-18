import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_cleaned_data(file_path):
    """
    Loads the cleaned movie dataset.
    """
    return pd.read_csv(file_path)


def build_similarity_matrix(data):
    """
    Converts movie tags into numbers and calculates similarity between movies.
    """
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(data["tags"]).toarray()

    similarity = cosine_similarity(vectors)
    return similarity


def recommend_movies(movie_title, data, similarity, number_of_recommendations=5):
    """
    Recommends movies similar to the given movie title.
    """
    movie_title = movie_title.lower()

    matching_movies = data[data["title"].str.lower() == movie_title]

    if matching_movies.empty:
        print(f"Movie '{movie_title}' was not found.")
        return

    movie_index = matching_movies.index[0]

    distances = list(enumerate(similarity[movie_index]))
    movies_sorted = sorted(distances, reverse=True, key=lambda x: x[1])

    print(f"\nMovies similar to {data.iloc[movie_index]['title']}:\n")

    for i in movies_sorted[1:number_of_recommendations + 1]:
        print(data.iloc[i[0]]["title"])


if __name__ == "__main__":
    data = load_cleaned_data("data/processed/cleaned_movies.csv")
    similarity = build_similarity_matrix(data)

    recommend_movies("Iron Man", data, similarity)