import streamlit as st
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ======================================================
# Base Agent
# ======================================================

class Agent:
    def __init__(self, name: str):
        self.name = name
  # TO DO ADD INPUT GUARDRAILS
  #  def validate_inputs(self, input_data: dict):
  #      for key in self.requires:
  #          if key not in input_data or input_data[key] is None:
  #              raise ValueError(f"{self.name} missing required input: {key}")

    def run(self, input_data: dict) -> dict:
        raise NotImplementedError


# ======================================================
# Weather Agent
# ======================================================

class WeatherAgent(Agent):
    def __init__(self):
        super().__init__("WeatherAgent")
        self.model = RandomForestRegressor(n_estimators=100)

        # Train with mock data
        historical_data = [
            {
                "month": m,
                "lat": 41.9028,
                "lon": 12.4964,
                "score": np.random.rand()
            }
            for m in range(1, 13)
        ]

        X = [[d["month"], d["lat"], d["lon"]] for d in historical_data]
        y = [d["score"] for d in historical_data]
        self.model.fit(X, y)

    def run(self, input_data):
        lat, lon = input_data["location"]

        predictions = [
            {
                "month": m,
                "score": float(self.model.predict([[m, lat, lon]])[0])
            }
            for m in range(1, 13)
        ]

        best_months = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        return {"best_months": best_months}


# ======================================================
# Hotel Agent
# ======================================================

class HotelAgent(Agent):
    def __init__(self):
        super().__init__("HotelAgent")

    def run(self, input_data):
        ranked_hotels = search_hotels(input_data["preferences"])

        if not ranked_hotels:
            raise ValueError("Hotel MCP tool returned no results")

        return {"hotels": ranked_hotels}

def search_hotels( preferences: str, top_k: int = 3):
    """Simulates an external hotel search tool (MCP module)."""
    hotels_database = [
        {"name": "Grand Hotel", "description": "Luxury hotel in city center with spa.", "price": 300},
        {"name": "Boutique Resort", "description": "Cozy boutique hotel with top amenities.", "price": 250},
        {"name": "City View Hotel", "description": "Modern hotel with stunning city views.", "price": 200}
    ]

    # Semantic similarity (embedding)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    hotel_embeddings = encoder.encode([h["description"] for h in hotels_database])
    pref_embedding = encoder.encode([preferences])
    scores = np.dot(hotel_embeddings, pref_embedding.T).flatten()

    top_idx = scores.argsort()[-top_k:][::-1]
    ranked = [{**hotels_database[i], "score": float(scores[i])} for i in top_idx]

    return ranked


# ======================================================
# Itinerary Agent (LLM)
# ======================================================

class ItineraryAgent(Agent):
    def __init__(self, model_path):
        super().__init__("ItineraryAgent")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            temperature=0.7
        )

    def run(self, input_data):
        prompt = f"""
            You are an expert travel planner.

            STRICT RULES:
            - Output plain text only
            - Do NOT write code
            Task:
                You are an expert travel planner.

                Create a {duration}-day travel itinerary for {destination}
                during the best month: {input_data['best_month']}.
                Recommended hotel: {input_data['hotel']['name']}.

                Return a clear day-by-day plan and include morning, midday, and afternoon plan.
            """

        response = self.llm(
                prompt,
                max_tokens=600,
                stop=["</s>"]
        )

        return {"itinerary": response["choices"][0]["text"].strip()}


# ======================================================
# Supervisor Agent
# ======================================================

class SupervisorAgent:
    def __init__(self):
        self.weather = WeatherAgent()
        self.hotel = HotelAgent()
        self.itinerary = ItineraryAgent(
            model_path="../models/llama-2-7b-chat.Q4_K_M.gguf"
        )

    def execute(self, user_input):
        context = {}

        # 1️⃣ Weather
        weather_out = self.weather.run({
            "location": user_input["location"]
        })
        context.update(weather_out)

        best_month = context["best_months"][0]["month"]

        # 2️⃣ Hotel
        hotel_out = self.hotel.run({
            "preferences": user_input["preferences"]
        })
        context.update(hotel_out)

        selected_hotel = context["hotels"][0]

        # 3️⃣ Itinerary
        itinerary_out = self.itinerary.run({
            "destination": user_input["destination"],
            "best_month": best_month,
            "hotel": selected_hotel,
            "duration": user_input["duration"]
        })
        context.update(itinerary_out)

        return context


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(page_title="AI Travel Planner ✈️")

st.title("AI Travel Planner ✈️")
st.write("Supervisor-based multi-agent travel planning")

destination = st.text_input("Destination", "Rome")
preferences = st.text_area(
    "Describe your ideal hotel",
    "Luxury hotel in city center with spa"
)
duration = st.slider("Trip duration (days)", 1, 14, 5)

if st.button("Generate Travel Plan ✨"):
    supervisor = SupervisorAgent()

    result = supervisor.execute({
        "destination": destination,
        "location": (41.9028, 12.4964),
        "preferences": preferences,
        "duration": duration
    })

    st.subheader("📅 Best Months to Visit")
    for m in result["best_months"]:
        st.write(f"Month {m['month']} — Score {m['score']:.2f}")

    st.subheader("🏨 Recommended Hotel")
    st.write(result["hotels"][0])

    st.subheader("📜 Itinerary")
    st.write(result["itinerary"])
