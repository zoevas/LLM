# -------------------------
# Symptom-to-Disease Medical Agent
# LangChain 1.2.6 Compatible
# -------------------------

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import hashlib
import requests
import os

# -------------------------
# Setup ChromaDB
# -------------------------
os.makedirs("./chroma_db", exist_ok=True)

client = chromadb.PersistentClient(path="./chroma_db")

# Collections
knowledge_db = client.get_or_create_collection("disease_symptoms")
memory_db = client.get_or_create_collection("agent_memory")

# -------------------------
# Sentence Embedding Model
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Load Symptom→Disease Dataset
# -------------------------
df = pd.read_csv('data/Symptom2Disease.csv')

print("🔄 Loading disease knowledge into ChromaDB...")

for index, row in df.iterrows():
    text = row['text']
    label = row['label']
    doc_id = hashlib.sha256(f"{text}_{label}_{index}".encode()).hexdigest()
    embedding = model.encode(text).tolist()
    knowledge_db.add(embeddings=[embedding], metadatas=[{"label": label}], ids=[doc_id])

print(f"✅ Knowledge DB count: {knowledge_db.count()}")

# -------------------------
# Memory Functions
# -------------------------
def recall_memory(query, top_k=2):
    emb = model.encode(query).tolist()
    results = memory_db.query(query_embeddings=[emb], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

def save_memory(text):
    emb = model.encode(text).tolist()
    doc_id = hashlib.sha256(text.encode()).hexdigest()
    memory_db.add(embeddings=[emb], documents=[text], ids=[doc_id])
    print(f"💾 Memory saved: {doc_id[:8]}...")

# -------------------------
# Medical Agent Class
# -------------------------
class MedicalAgent:
    def __init__(self, llm_url="http://localhost:11434/api/generate"):
        self.llm_url = llm_url

    def retrieve_diseases(self, query, top_k=5):
        emb = model.encode(query).tolist()
        results = knowledge_db.query(query_embeddings=[emb], n_results=top_k)
        # remove duplicates
        diseases = list({meta["label"] for meta in results["metadatas"][0]})
        return diseases

    def generate_prompt(self, query, diseases, past_memory):
        memory_text = "\n".join([f"- {m}" for m in past_memory]) if past_memory else "None"
        disease_text = "\n".join(diseases)
        return f"""
You are a medical assistant (not a doctor).

Relevant past memory:
{memory_text}

User symptoms:
{query}

Possible diseases retrieved from database:
{disease_text}

Explain which disease is most likely and why.
"""

    def call_llm(self, prompt):
        response = requests.post(
            self.llm_url,
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    def run(self, query):
        # 1️⃣ Recall memory
        past_memory = recall_memory(query)

        # 2️⃣ Retrieve diseases
        diseases = self.retrieve_diseases(query)

        # 3️⃣ Generate prompt for LLM
        prompt = self.generate_prompt(query, diseases, past_memory)

        # 4️⃣ Call LLM
        answer = self.call_llm(prompt)

        # 5️⃣ Save answer in memory
        save_memory(f"Symptoms: {query}\nDiseases: {diseases}\nAnswer: {answer}")

        return answer

# -------------------------
# Run Agent
# -------------------------
if __name__ == "__main__":
    agent = MedicalAgent()
    query = "I have dizziness and nausea"
    answer = agent.run(query)

    print("\n🤖 Agent Response:")
    print(answer)

    # Optional: show all memory entries
    print("\n🧠 All stored memory entries:")
    all_memories = memory_db.get(include=["documents"])
    for i, doc in enumerate(all_memories["documents"]):
        print(f"{i+1}. {doc[:150]}...")
