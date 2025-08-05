import streamlit as st
import joblib
import pandas as pd

# Load trained model from part 1 of the exam
model = joblib.load("model.pkl")

# Load data for driver, constructor, and race names
drivers = pd.read_csv("../data/f1Data/drivers.csv")
constructors = pd.read_csv("../data/f1Data/constructors.csv")
races = pd.read_csv("../data/f1Data/races.csv")

# Mapping: driver name → driverId
driver_map = dict(zip(drivers['forename'] + " " + drivers['surname'], drivers['driverId']))

# Mapping: constructor name → constructorId
constructor_map = dict(zip(constructors['name'], constructors['constructorId']))

# Sort races by year descending, then round ascending
races = races.sort_values(by=["year", "round"], ascending=[False, True])

# Create race name (e.g., "2023 - Belgian Grand Prix")
races['race_name'] = races['year'].astype(str) + " - " + races['name']
race_map = dict(zip(races['race_name'], zip(races['year'], races['round'])))

# ---- Streamlit App ----

st.title("🏎️ F1 Podium Predictor")

st.markdown("""
This app uses a machine learning model trained on Formula 1 data to predict whether a driver will **finish on the podium** in a given race.

Just fill in the form below and click **Predict** to get the result.
""")

# Input: Grid Position
grid = st.number_input("Starting Grid Position", min_value=1, max_value=22, value=5)

# Input: Driver
driver_name = st.selectbox("Driver", list(driver_map.keys()))
driver_id = driver_map[driver_name]

# Input: Constructor
constructor_name = st.selectbox("Constructor (Team)", list(constructor_map.keys()))
constructor_id = constructor_map[constructor_name]

# Input: Grand Prix
selected_gp = st.selectbox("Grand Prix", list(race_map.keys()))
year, round_num = race_map[selected_gp]

# Prediction Button
if st.button("Predict Podium Finish"):
    features = [[grid, driver_id, constructor_id, year, round_num]]
    result = model.predict(features)[0]

    if result == 1:
        st.success(f"🥇 {driver_name} is predicted to finish on the podium!")
    else:
        st.info(f"🏁 {driver_name} is predicted **not** to finish on the podium.")

st.markdown("""

### Note on Predictions

This app uses historical race data to **simulate** podium predictions based on selected inputs (e.g., grid position, driver, team, race).

Since we are using past data, the model is predicting events that have already occurred. While this may seem redundant, the goal is to show how well the model **learns patterns** from data and how it would perform on **unseen scenarios**.

In a real-world setup, the same model could be used **before a race** to estimate the chance of a podium finish based on qualifying results and driver/team history.
            
""")