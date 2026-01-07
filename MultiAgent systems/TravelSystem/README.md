# AI Travel Planner 

An AI-powered travel planner that helps you:

- Find the best months to visit a destination based on weather data  
- Recommend hotels matching your preferences  
- Generate a full day-by-day itinerary using a Large Language Model (LLaMA)  

This project demonstrates a **multi-agent system** approach, where different agents handle specific tasks and cooperate to provide a complete travel plan.


<img width="877" height="890" alt="Καταγραφή" src="https://github.com/user-attachments/assets/d64f0280-dc51-4747-a3fd-83a136faeb2e" />

---

## Multi-Agent Architecture

| Agent | Responsibility |
|-------|----------------|
| `WeatherAnalysisAgent` | Predicts the best months to visit using historical weather data |
| `HotelRecommenderAgent` | Finds hotels that match user preferences using semantic embeddings |
| `ItineraryPlannerAgent` | Generates a complete travel itinerary using LLaMA |

---

##  Requirements

- Python 3.13 (or 3.10+)
- pip
- Streamlit
- scikit-learn
- sentence-transformers
- llama_cpp

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download LLaMA 2 7B Chat Model (GGUF)

The itinerary generation uses **LLaMA 2 7B Chat (quantized GGUF)**. This model must be downloaded manually or via Hugging Face.

### Step 1: Create a Hugging Face account
1. Go to [Hugging Face](https://huggingface.co/) and create an account.
2. Accept Meta's license for LLaMA 2.

### Step 2: Get your Hugging Face token
1. Go to [Settings â†’ Access Tokens](https://huggingface.co/settings/tokens)  
2. Click **New token** â†’ set role to **Read**  
3. Copy the token (youâ€™ll need it for the CLI)

### Step 3: Install the Hugging Face CLI
```bash
pip install huggingface-hub
```

### Step 4: Log in via CLI
```bash
huggingface-cli login
```
Paste your token when prompted.

### Step 5: Download the GGUF model
```bash
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models
```

After downloading, your project structure should be:

```
your_project/
models/
  llama-2-7b-chat.Q4_K_M.gguf
  systems/
  Multimodal_Travel_Planning.py
  requirements.txt
  README.md
```

---

## Running the App

```bash
streamlit run Multimodal_Travel_Planning.py
```

1. Enter your destination  
2. Describe your ideal hotel  
3. Choose trip duration  
4. Click **Generate Travel Plan**  

The app will show:

- Best months to visit  
- Top hotel recommendations  
- Full day-by-day itinerary  

---

## Notes

- The system is modular â€” you can replace or add agents (e.g., flights, attractions) easily  
- LLaMA 7B requires **~8â€“10 GB RAM**; smaller quantized models are recommended for CPU  
- API keys or sensitive info should **never** be committed  

---

## License

This project is for educational and experimental purposes.

---

¨

