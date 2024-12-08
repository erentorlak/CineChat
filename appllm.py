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

# Global variables for LLM and retriever
retriever = None
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Make sure to load your API key from environment variables
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Function to initialize the LLM and retriever
def initialize_retriever():
    global retriever

    # Load and preprocess the CSV file
    df = pd.read_csv("CineChatCSV_cleaned_new.csv")

    def combine_fields(row):
        # Safely retrieve and convert the summary to a string
        summary = str(row.get('summary', ''))
        
        # Calculate the midpoint
        midpoint = len(summary) * 3 // 4
        
        # Slice the summary to include only the first half
        half_summary = summary[:midpoint]
        
        # Format the content string
        content = f"Title: {row.get('title', '')}\nSummary: {half_summary}"
        
        return content


    df["content"] = df.apply(combine_fields, axis=1)

    documents = []
    for _, row in df.iterrows():
        metadata = {
            "title": row["title"],
            "director": row["director"],
            "actors": row["actors"],
            "genres": row["genres"],
            "plot": row["plot"],
            "id": row["id"],
            "imdb_id": row["imdb_id"],
            "poster_path": row["poster_path"],
            "year": row["year"],
            "rating": row["rating"],
            #"summary": row["summary"]
        }
        doc = Document(page_content=row["content"], metadata=metadata)
        documents.append(doc)
        print(doc.metadata["imdb_id"])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in documents],
    )

    metadata_field_info = [
        AttributeInfo(name="title", description="The title of the movie", type="string"),
        AttributeInfo(name="director", description="The director of the movie", type="string"),
        AttributeInfo(name="actors", description="The actors in the movie", type="string"),
        AttributeInfo(name="genres", description="The genres of the movie", type="string"),
        AttributeInfo(name="year", description="The release year of the movie", type="integer"),
        AttributeInfo(name="rating", description="The rating of the movie", type="float"),
        #AttributeInfo(name="id", description="The internal ID of the movie", type="integer"),
        #AttributeInfo(name="summary", description="The detailed summary of the movie", type="string"),
    ]

    document_content_description = (
        "Detailed information about movies, including title, director, actors, genres, plot, and summary. "
        #"If user wants recommendations, it can also provide recommendations based on genres, actors, or directors."#
    )

    llm = ChatOpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_filters=True,
    )

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

# Initialize retriever during app startup
initialize_retriever()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg.title())
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
    print("query: " + text)
    global retriever
    try:
        results = retriever.invoke(text)
        if not results:
            response = {"message": "No results found for your query."}
        else:
            movie_details = []
            for doc in results:
                rating, summary, year = get_movie_details(doc.metadata['imdb_id'])  # Fetch rating, summary, and year
                movie_details.append({
                    "title": doc.metadata["title"],
                    "poster_url": f"https://image.tmdb.org/t/p/w500/{doc.metadata['poster_path']}",
                    "rating": rating,
                    "summary": summary,  # Include the summary
                    "year": year,  # Include the year from TMDb
                    "genre": doc.metadata.get("genres", "Unknown"),  # Add the genre
                    "actors": ", ".join(doc.metadata.get("actors", "").split(",")[:3])  # Add up to 3 actors
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
