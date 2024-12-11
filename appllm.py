from flask import Flask, render_template, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables for LLM
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Make sure to load your API key from environment variables
TMDB_BASE_URL = "https://api.themoviedb.org/3"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Load OpenAI API key from environment variables

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vectorstore = Chroma.from_documents(
#    documents=documents,
#    embedding=embeddings,
#    persist_directory="movie_vectorstore5openai",
#    ids=[doc.metadata["id"] for doc in documents],
#    #use_jsonb=True
#)

def load_vectorstore(persist_directory="movie_vectorstore5openai"):
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

vectorstore = load_vectorstore()

def recommend_movies(query, top_k=3):
    results = vectorstore.similarity_search_by_vector(embedding=embeddings.embed_query(query), k=top_k)
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

# Function to call GPT API to generate response
#def call_gpt_api(query, movie_titles):
#    prompt = (
#        f"Based on the query '{query}', the recommended movies are {', '.join(movie_titles)}. "
#        "Can you suggest two more movies that are similar to this query?"
#    )
#
#    url = "https://api.openai.com/v1/chat/completions"
#    headers = {
#        "Content-Type": "application/json",
#        "Authorization": f"Bearer {OPENAI_API_KEY}"
#    }
#    payload = {
#        "model": "gpt-4",
#        "messages": [{"role": "system", "content": "You are a movie recommendation assistant."},
#                     {"role": "user", "content": prompt}],
#        "temperature": 0.7
#    }
#
#    response = requests.post(url, headers=headers, json=payload)
#    if response.status_code == 200:
#        return response.json()["choices"][0]["message"]["content"]
#    else:
#        raise Exception(f"GPT API Error: {response.status_code}, {response.text}")
#    


def call_groq_llm(query, movie_titles):
    system = """You are a movie recommendation assistant. 
    You have been asked to evaluate a movie recommendation according to question and retrieved movie titles.
    If not enough movie titles are retrieved, you can recommend movies based on the user question.
    """
    grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved Movie Titles: \n\n {movie_titles} \n\n User question: {question}"),
    ]
)
    llm = ChatGroq(model="mixtral-8x7b-32768")
    last_llm = grade_prompt | llm
    
    response = last_llm.invoke({"movie_titles": movie_titles, "question": query})

    return response.content



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
                rating = movie_data.get("vote_average", 0)
                summary = movie_data.get("overview", "No summary available.")
                release_date = movie_data.get("release_date", "")
                year = release_date[:4] if release_date else "Unknown"
                return rating, summary, year
            
    else:
        print(f"Error: {response.status_code}, {response.text}")

    return None, "No summary available.", "Unknown"

# Modified function to get chat response
def get_chat_response(text):
    print("query: " + text)
    global retriever
    movie_details = []
    movie_titles = []
    try:
        results = recommend_movies(text)
        #print("results: " + str(results))
              
        for doc in results:
            rating, summary, year = get_movie_details(doc['imdb_id'])
            movie_titles.append(doc["title"])
            movie_details.append({
                "title": doc["title"],
                "poster_url": f"https://image.tmdb.org/t/p/w500/{doc['poster_path']}",
                "rating": rating,
                "summary": summary,
                "year": year,
                "genre": doc.get("genres", "Unknown"),
                "actors": ", ".join(doc.get("actors", "").split(",")[:3])
            })
        gpt_response = call_groq_llm(text, movie_titles)
        response = {"gpt_response": gpt_response, "movies": movie_details}
    except Exception as e:
        print("exception: " + str(e))
        gpt_response = call_groq_llm(text, movie_titles)
        response = {"gpt_response": gpt_response, "movies": movie_details}

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# %%
