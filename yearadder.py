import csv
import requests
import os

# Replace with your actual TMDb API key
TMDB_API_KEY = "b6e03e5f7bf3a6786e5312a121bb1bb9"
BASE_URL = "https://api.themoviedb.org/3/find/"

# Input and output file paths
INPUT_CSV = "CineChatCSV_cleaned.csv"
OUTPUT_CSV = "output_with_year_rating.csv"

# Function to fetch year and rating from TMDb API
def fetch_movie_details(imdb_id):
    params = {
        "api_key": TMDB_API_KEY,
        "external_source": "imdb_id",
    }
    url = f"{BASE_URL}{imdb_id}"
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract year and rating from the first movie result, if available
        if "movie_results" in data and data["movie_results"]:
            movie = data["movie_results"][0]
            release_date = movie.get("release_date", "")
            year = release_date.split("-")[0] if release_date else "Unknown"
            rating = movie.get("vote_average", "Unknown")
            return year, rating
        return "Unknown", "Unknown"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for IMDb ID {imdb_id}: {e}")
        return "Unknown", "Unknown"

# Process the CSV file
if os.path.exists(INPUT_CSV):
    with open(INPUT_CSV, mode="r", encoding="utf-8") as infile, open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["year", "rating"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            imdb_id = row.get("imdb_id")
            if imdb_id:
                year, rating = fetch_movie_details(imdb_id)
                row["year"] = year
                row["rating"] = rating
            else:
                row["year"] = "Unknown"
                row["rating"] = "Unknown"
            writer.writerow(row)

    print(f"Processing complete. Output saved to {OUTPUT_CSV}")
else:
    print(f"Input file {INPUT_CSV} not found.")