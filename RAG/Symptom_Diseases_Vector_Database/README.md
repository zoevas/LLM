# Symptom-to-Disease Retrieval System

This project demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline for mapping user-reported symptoms to possible diseases using
**vector databases**, **sentence embeddings**, and a **local LLM**.

It uses: - **ChromaDB** for vector storage and similarity search -
**Sentence Transformers** for text embeddings - **A medical symptoms
dataset from Kaggle** - **LLaMA 3 via Ollama** for natural language
explanations

> **Disclaimer**: This project is for educational purposes only. It
> is **not a medical diagnostic tool**.

------------------------------------------------------------------------

## Features

-   Loads a symptom-to-disease dataset\
-   Generates embeddings for symptom descriptions\
-   Stores embeddings in ChromaDB\
-   Performs similarity search based on user symptoms\
-   Uses an LLM to explain likely diseases based on retrieved data

------------------------------------------------------------------------

##  Dataset

Dataset used: - **Symptom2Disease** from Kaggle\
https://www.kaggle.com/datasets/niyarrbarman/symptom2disease

Expected CSV structure: - `text` -- symptom description\
- `label` -- disease name

Place the dataset at:

    data/Symptom2Disease.csv

------------------------------------------------------------------------

##  Installation

### 1. Clone the repository

``` bash
git clone https://github.com/your-username/symptom-disease-rag.git
cd symptom-disease-rag
```

### 2. Create a virtual environment (optional but recommended)

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install chromadb pandas sentence-transformers requests
```

### 4. Install & run Ollama

Ensure Ollama is installed and running:

``` bash
ollama run llama3
```

------------------------------------------------------------------------

##  How It Works

1.  **Load Dataset**\
    Reads symptom descriptions and disease labels from CSV

2.  **Embedding Generation**\
    Uses `all-MiniLM-L6-v2` to convert symptom text into vectors

3.  **Vector Storage**\
    Stores embeddings in a ChromaDB collection

4.  **Querying**\
    User symptoms are embedded and compared using cosine similarity

5.  **LLM Explanation**\
    Top matching diseases are passed to LLaMA 3 for explanation

------------------------------------------------------------------------

## Usage

Run the script:

``` bash
python main.py
```

Example symptom query inside the script:

``` python
query = "I have fever, headache, and muscle pain"
```

