import pandas as pd
import ast
from pathlib import Path

def load_data():
    movies = pd.read_csv("data/raw/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/raw/tmdb_5000_credits.csv")
    return movies, credits

def convert_json_column(text):
    # Converts columns that look like JSON strings into a list of names.
    # Ex: genres, keywords, cast
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except:
        return[]

def get_top_cast(text, limit=3):
    # Gets the first few cast members from the cast column.
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items[:limit]]
    except:
        return []
def get_director(text):
    """
    Gets the director from the crew column.
    """
    try:
        crew = ast.literal_eval(text)
        for person in crew:
            if person["job"] == "Director":
                return person["name"]
        return ""
    except:
        return ""


def clean_text_list(items):
    """
    Removes spaces from names/phrases.
    Example: 'Science Fiction' becomes 'ScienceFiction'
    """
    return [item.replace(" ", "") for item in items]


def create_tags(row):
    """
    Combines overview, genres, keywords, cast, and director into one text field.
    """
    overview = row["overview"].split()
    genres = row["genres"]
    keywords = row["keywords"]
    cast = row["cast"]
    director = [row["director"]]

    return " ".join(overview + genres + keywords + cast + director).lower()


def clean_movie_data(movies, credits):
    # Merge both files on title
    data = movies.merge(credits, on="title")

    # Keep useful columns
    data = data[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    # Remove rows with missing overview
    data = data.dropna(subset=["overview"])

    # Convert JSON-like text columns into lists
    data["genres"] = data["genres"].apply(convert_json_column)
    data["keywords"] = data["keywords"].apply(convert_json_column)
    data["cast"] = data["cast"].apply(get_top_cast)
    data["director"] = data["crew"].apply(get_director)

    # Clean spaces
    data["genres"] = data["genres"].apply(clean_text_list)
    data["keywords"] = data["keywords"].apply(clean_text_list)
    data["cast"] = data["cast"].apply(clean_text_list)
    data["director"] = data["director"].apply(lambda x: x.replace(" ", ""))

    # Create combined tags column
    data["tags"] = data.apply(create_tags, axis=1)

    # Final dataset
    cleaned_data = data[["movie_id", "title", "tags"]]

    return cleaned_data


if __name__ == "__main__":
    movies, credits = load_data()
    cleaned_data = clean_movie_data(movies, credits)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    cleaned_data.to_csv("data/processed/cleaned_movies.csv", index=False)

    print(cleaned_data.head())
    print("Cleaned movie data saved to data/processed/cleaned_movies.csv")