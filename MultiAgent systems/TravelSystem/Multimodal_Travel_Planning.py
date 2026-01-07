import streamlit as st
import numpy as np


from sklearn.ensemble import RandomForestRegressor

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

weather_agent = WeatherAnalysisAgent()
weather_agent.train(historical_weather_data)


class ItineraryPlannerAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def create_itinerary(self, destination, best_month, hotel, duration):
        client = openai.OpenAI(api_key=self.api_key)

        prompt = f"""
        Create a {duration}-day travel itinerary for {destination} in the best month: {best_month}.
        Recommended Hotel: {hotel['name']}.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert travel planner."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

from llama_cpp import Llama

class ItineraryPlannerAgent2:
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

    st.subheader("📆 Best Months to Visit")
    for m in best_months:
        st.write(f"Month {m['month']}: Score {m['score']:.2f}")

    agent = ItineraryPlannerAgent2("models/llama-2-7b-chat.gguf")

    itinerary = agent.create_itinerary(
        destination="Paris",
        best_month="May",
        hotel={"name": "Hotel Le Meurice"},
        duration=5
    )

    print(itinerary)