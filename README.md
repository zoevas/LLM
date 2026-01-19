# LLM-based Small Projects

This repository contains small projects and experiments based on **Large Language Models (LLMs)**, focusing on practical use cases, prototyping, and learning.

## Requirements

- **Python 3.13** (Python 3.10+ should also work unless otherwise specified)
- `pip` (Python package manager)
- (Optional but recommended) `virtualenv`

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Projects

Run the main script or individual project files:

```
# Navigate to the project folder
cd Symptom_Diseases_Vector_Database
python SymptomToDisease.py

# Or run LangChain version
cd MultiAgent_Systems
python SymptomToDiseaseLangchain.py

> Adjust the filename depending on which project version you want to run.

```


```

(Adjust the filename depending on the project you want to run.)

---

## Project Structure

```
RAG/
Symptom_Diseases_Vector_Database/
 data/
  Symptom2Disease.csv
  SymptomToDisease.py
  SymptomToDiseaseLangchain.py
  SymptomToDiseaseWithAgents.py
  README.md
 MultiAgent systems/
  TravelSystem/
    Multimodal_Travel_Planning_Supervisor_pattern.py
	README.md
README.md
requirements.txt
```

## Notes

- Ensure `requirements.txt` is kept up to date when adding new dependencies.
- If Python 3.13 is not available on your system, Python **3.10+** is recommended for compatibility with most LLM libraries.

## License

This project is for educational and experimental purposes. Add a license file if you plan to distribute or reuse the code.

---
