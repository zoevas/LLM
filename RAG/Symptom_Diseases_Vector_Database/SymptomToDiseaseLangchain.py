from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import pandas as pd

# ------------------
# Embeddings
# ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------
# Vector Stores
# ------------------
knowledge_db = Chroma(
    collection_name="disease_symptoms",
    persist_directory="./chroma_db/knowledge",
    embedding_function=embeddings
)

memory_db = Chroma(
    collection_name="agent_memory",
    persist_directory="./chroma_db/memory",
    embedding_function=embeddings
)

# ------------------
# Load the dataset https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
# ------------------
df = pd.read_csv("data/Symptom2Disease.csv")

if knowledge_db._collection.count() == 0:
    print("📥 Loading disease knowledge...")
    knowledge_db.add_texts(
        texts=df["text"].tolist(),
        metadatas=[{"label": l} for l in df["label"]]
    )

# ------------------
# LLM
# ------------------
llm = ChatOllama(model="llama3")

# ------------------
# Prompt
# ------------------
prompt = PromptTemplate(
    input_variables=["memory", "symptoms", "diseases"],
    template="""
You are a medical assistant (not a doctor).

Relevant past memory:
{memory}

User symptoms:
{symptoms}

Possible diseases:
{diseases}

Explain which disease is most likely and why.
"""
)

# ------------------
# Memory functions
# ------------------
def recall_memory(query, k=2):
    docs = memory_db.similarity_search(query, k=k)
    return "\n".join([d.page_content for d in docs]) if docs else "None"

def save_memory(text):
    memory_db.add_texts([text])

# ------------------
# Query
# ------------------
query = "I have dizziness"

# Disease retrieval
docs = knowledge_db.similarity_search(query, k=5)
unique_diseases = sorted({d.metadata["label"] for d in docs})
disease_text = "\n".join(unique_diseases)

# Recall memory
memory_text = recall_memory(query)

# Chain
chain = (
    {
        "memory": lambda _: memory_text,
        "symptoms": RunnablePassthrough(),
        "diseases": lambda _: disease_text
    }
    | prompt
    | llm
)

response = chain.invoke(query)

# Save memory
save_memory(
    f"Symptoms: {query}\nDiseases: {disease_text}\nAnswer: {response.content}"
)

print("\n🤖 LLM Response:")
print(response.content)
