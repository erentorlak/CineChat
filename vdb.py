import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Load the CSV file
df = pd.read_csv("CineChatCSV_cleaned.csv")

# Step 1: More compact content format
def combine_fields(row):
    content = f"""Title: {row['title']}

Plot: {row['plot']}""".strip()
    return content

# Create content column
df["content"] = df.apply(combine_fields, axis=1)

# Step 2: Create documents with explicit control
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
    }

    doc = Document(page_content=row["content"], metadata=metadata)
    documents.append(doc)
    print(doc.metadata["poster_path"])

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    # Step 3: Create vector store with explicit chunking control
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in documents],
    )
    print("Vector store created successfully")
except Exception as e:
    print(f"Error occurred while creating the vector store: {e}")

# Metadata field information
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the movie",
        type="string",
    ),
    AttributeInfo(
        name="director",
        description="The director of the movie",
        type="string",
    ),
    AttributeInfo(
        name="actors",
        description="The actors in the movie",
        type="string",
    ),
    AttributeInfo(
        name="genres",
        description="The genres of the movie",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The release year of the movie",
        type="integer",
    ),
    AttributeInfo(
        name="id",
        description="The internal ID of the movie",
        type="integer",
    ),
]

# Document content description
document_content_description = document_content_description = (
    "Detailed information about movies, including title, director, actors, "
    "genres, plot, and summary. If user wants recommendation, it can also provide recommendations based on genres, "
    "actors, or directors. "
)

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Create the SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_filters=True,        # metadata filters
)

# Query the retriever
query = "I love visually stunning films. What should I watch?"
results = retriever.invoke(query)

# Check if any results were found
if not results:
    print("No results found.")
else:
    # Print the results (movie titles)
    titles = [doc.metadata["title"] for doc in results]
    print("Found movies:", titles)
