import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
df = pd.read_csv('data/Symptom2Disease.csv')

# Initialize Chroma client
client = chromadb.Client()

# Create a new collection for disease symptoms
collection = client.create_collection(name="disease_symptoms")

# Initialize a sentence transformer model to generate text embeddings
# You can use 'all-MiniLM-L6-v2' or another transformer-based model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to add data into Chroma
def add_to_chroma(collection, text, label, id):
    # Convert text (symptom description) into an embedding using the model
    embedding = model.encode(text).tolist()

    # Insert the embedding into the collection with metadata (label)
    collection.add(
        embeddings=[embedding],        # Embedding generated from text
        metadatas=[{"label": label}],  # Metadata (disease label)
        ids=[id]                       # Unique ID for each record
    )

# Iterate through the dataframe and insert each row into Chroma
for index, row in df.iterrows():
    text = row['text']  # The symptom description text
    label = row['label']  # The disease condition label
    add_to_chroma(collection, text, label, str(index))  # Add each entry with unique ID

print("Data inserted into Chroma successfully!")

# Example user question (symptoms)
query = "I have fever, headache, and muscle pain"

# Convert question to embedding
query_embedding = model.encode(query).tolist()

# Search in Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5  # top 5 most similar symptom entries
)

# Prepare context for the LLM
retrieved_info = ""
for meta in results["metadatas"][0]:
    retrieved_info += f"Disease: {meta['label']}\n"
    print(f" {retrieved_info}")


prompt = f"""
You are a medical assistant (not a doctor).

User symptoms:
{query}

Possible diseases retrieved from database:
{retrieved_info}

Explain which disease is most likely and why.
"""
import requests
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

print(response.json()["response"])