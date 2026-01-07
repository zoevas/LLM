import streamlit as st
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer


st.title("AI Travel Planner ✈️")
st.write("Find the best time to travel and discover the perfect hotel!")

destination = st.text_input("Enter your destination (e.g., Rome):", "Rome")
preferences = st.text_area("Describe your ideal hotel:", "Luxury hotel in city center with spa.")
duration = st.slider("Trip duration (days):", 1, 14, 5)



# -------------------------------
# AI Models for Travel Planning
# -------------------------------
class WeatherAnalysisAgent:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, historical_data):
        X = np.array([[d['month'], d['latitude'], d['longitude']] for d in historical_data])
        y = np.array([d['weather_score'] for d in historical_data])
        self.model.fit(X, y)

    def predict_best_time(self, location):
        predictions = [
            {'month': month,
             'score': float(self.model.predict([[month, location['latitude'], location['longitude']]]).item())}
            for month in range(1, 13)
        ]
        return sorted(predictions, key=lambda x: x['score'], reverse=True)[:3]

historical_weather_data = [
    {'month': i, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': np.random.rand()} for i in range(1, 13)
]

# is a hotel recommendation system that utilizes semantic similarity to
# match hotels with user preferences
class HotelRecommenderAgent:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.hotels_db = []
        self.hotels_embeddings = None

    def add_hotels(self, hotels):
        self.hotels_db = hotels
        descriptions = [h['description'] for h in hotels]
        self.hotels_embeddings = self.encoder.encode(descriptions)

    def find_hotels(self, preferences, top_k=3):
        pref_embedding = self.encoder.encode([preferences])
        similarities = np.dot(self.hotels_embeddings, pref_embedding.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [{**self.hotels_db[i], 'score': float(similarities[i])} for i in top_indices]

weather_agent = WeatherAnalysisAgent()
weather_agent.train(historical_weather_data)



hotels_database = [
    {'name': 'Grand Hotel', 'description': 'Luxury hotel in city center with spa.', 'price': 300},
    {'name': 'Boutique Resort', 'description': 'Cozy boutique hotel with top amenities.', 'price': 250},
    {'name': 'City View Hotel', 'description': 'Modern hotel with stunning city views.', 'price': 200}
]
hotel_agent = HotelRecommenderAgent()
hotel_agent.add_hotels(hotels_database)


from llama_cpp import Llama

class ItineraryPlannerAgent:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            temperature=0.7
        )

    def create_itinerary(self, destination, best_month, hotel, duration):
        prompt = f"""
            You are an expert travel planner.

            Create a {duration}-day travel itinerary for {destination}
            during the best month: {best_month}.
            Recommended hotel: {hotel['name']}.

            Return a clear day-by-day plan.
        """

        response = self.llm(
            prompt,
            max_tokens=300,
            stop=["</s>"]
        )

        return response["choices"][0]["text"].strip()


if st.button("Generate Travel Plan ✨"):
    best_months = weather_agent.predict_best_time({'latitude': 41.9028, 'longitude': 12.4964})
    best_month = best_months[0]['month']

    st.subheader("Best Months to Visit")
    for m in best_months:
        st.write(f"Month {m['month']}: Score {m['score']:.2f}")

    recommended_hotels = hotel_agent.find_hotels(preferences)

    agent = ItineraryPlannerAgent("../models/llama-2-7b-chat.Q4_K_M.gguf")

    itinerary = agent.create_itinerary(destination, best_month, recommended_hotels[0], duration)


    print(itinerary)

    st.subheader("📜 Generated Itinerary")
    st.write(itinerary)