import pandas as pd
from recommender import build_similarity_matrix, recommend_movies


def main():
    data = pd.read_csv("data/processed/cleaned_movies.csv")
    similarity = build_similarity_matrix(data)

    print("Movie Recommendation System")
    print("Type a movie title to get recommendations.")
    print("Type 'quit' to exit.\n")

    while True:
        movie_title = input("Enter a movie title: ")

        if movie_title.lower() == "quit":
            print("Goodbye!")
            break

        recommend_movies(movie_title, data, similarity)
        print()


if __name__ == "__main__":
    main()