from flask import Flask, render_template, request, jsonify
import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Make sure to load your API key from environment variables
TMDB_BASE_URL = "https://api.themoviedb.org/3"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %%
def load_vectorstore(persist_directory="movie_vectorstore"):
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

vectorstore2 = load_vectorstore()

# %%
def recommend_movies(query, top_k=5):
    results = vectorstore2.similarity_search_by_vector(embedding=embeddings.embed_query(query), k=top_k)
    recommendations = []
    for result in results:
        metadata = result.metadata
        recommendations.append({
            "title": metadata["title"],
            "director": metadata["director"],
            "actors": metadata["actors"],
            "genres": metadata["genres"],
            "plot": metadata["plot"],
            "poster_path": metadata["poster_path"],
            "imdb_id": metadata["imdb_id"],
        })
    return recommendations

# Function to get movie details (including rating) from TMDb by IMDb ID
def get_movie_rating(imdb_id):
    url = f"{TMDB_BASE_URL}/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "movie_results" in data and len(data["movie_results"]) > 0:
            movie_id = data["movie_results"][0]["id"]
            movie_details_url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
            movie_details_response = requests.get(movie_details_url)
            if movie_details_response.status_code == 200:
                movie_data = movie_details_response.json()
                rating = movie_data.get("vote_average", 0)  # Get the rating (vote_average)
                return rating
    return None


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return response

# Function to get movie details (including rating and summary) from TMDb by IMDb ID
def get_movie_details(imdb_id):
    url = f"{TMDB_BASE_URL}/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "movie_results" in data and len(data["movie_results"]) > 0:
            movie_id = data["movie_results"][0]["id"]
            movie_details_url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
            movie_details_response = requests.get(movie_details_url)
            if movie_details_response.status_code == 200:
                movie_data = movie_details_response.json()
                rating = movie_data.get("vote_average", 0)  # Get the rating (vote_average)
                summary = movie_data.get("overview", "No summary available.")  # Get the summary
                release_date = movie_data.get("release_date", "")  # Get the release date
                year = release_date[:4] if release_date else "Unknown"  # Extract year from release_date
                return rating, summary, year
    return None, "No summary available.", "Unknown"

# Modified function to get chat response
def get_chat_response(text):
    try:
        results = recommend_movies(text)
        if not results:
            response = {"message": "No results found for your query."}
        else:
            movie_details = []
            for doc in results:
                rating, summary, year = get_movie_details(doc['imdb_id'])  # Fetch rating, summary, and year
                movie_details.append({
                    "title": doc["title"],
                    "poster_url": f"https://image.tmdb.org/t/p/w500/{doc['poster_path']}",
                    "rating": rating,
                    "summary": summary,  # Include the summary
                    "year": year,  # Include the year from TMDb
                    "genre": doc.get("genres", "Unknown"),  # Add the genre
                    "actors": ", ".join(doc.get("actors", "").split(",")[:3])  # Add up to 3 actors
                })
            response = {"movies": movie_details}
    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response)

if __name__ == '__main__':
     # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Bind to all interfaces
    app.run(host="0.0.0.0", port=port)
