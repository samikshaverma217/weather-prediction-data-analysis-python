import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

st.set_page_config(page_title="Weather Prediction", layout="wide")
st.title("Weather Prediction and Data Analysis using Python")

# --- LOGIN ---
password = st.text_input("Enter Password", type="password")
if password != "1234":
    st.warning("Please enter password to continue")
    st.stop()
st.success("Login Successful")

# --- SAMPLE DATA - Hisar / Haryana Weather ---
csv_data = """Date,Temperature,Humidity,Rainfall,Wind_Speed,Pressure
2026-01-01,12,65,0.0,8,1012
2026-01-02,13,62,0.0,9,1011
2026-01-03,14,60,0.0,7,1010
2026-01-04,15,58,0.0,10,1009
2026-01-05,16,55,0.2,11,1008
2026-01-06,18,52,0.0,9,1007
2026-01-07,20,50,0.0,8,1006
2026-01-08,21,48,0.0,10,1005
2026-01-09,22,45,0.0,12,1004
2026-01-10,24,42,0.0,11,1003
2026-01-11,25,40,0.0,9,1002
2026-01-12,23,55,2.5,14,1001
2026-01-13,20,68,8.0,15,1000
2026-01-14,19,70,5.2,13,1001
2026-01-15,21,60,1.0,10,1003
2026-01-16,24,52,0.0,9,1005
2026-01-17,26,48,0.0,8,1007
2026-01-18,28,45,0.0,10,1008
2026-01-19,30,40,0.0,12,1009
2026-01-20,31,38,0.0,11,1010
2026-01-21,29,50,3.0,13,1008
2026-01-22,27,58,6.5,14,1006
2026-01-23,25,62,1.5,12,1007
2026-01-24,26,55,0.0,10,1009
2026-01-25,28,48,0.0,9,1010
2026-01-26,32,35,0.0,8,1011
2026-01-27,33,32,0.0,9,1012
2026-01-28,34,30,0.0,10,1013
2026-01-29,35,28,0.0,11,1014
2026-01-30,33,35,0.0,12,1012
"""

# --- UPLOAD ---
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload Weather CSV File", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File Uploaded Successfully")
else:
    df = pd.read_csv(io.StringIO(csv_data))

df["Date"] = pd.to_datetime(df["Date"])

# --- SEARCH BY DATE ---
st.subheader("Search Weather by Date")
search_date = st.date_input("Select Date")
if st.button("Search"):
    result = df[df["Date"] == pd.to_datetime(search_date)]
    if len(result) > 0:
        st.dataframe(result)
    else:
        st.error("No data found for this date")

# --- OVERVIEW METRICS ---
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Average Temperature", f"{df['Temperature'].mean():.1f} °C")
with c2: st.metric("Max Temperature", f"{df['Temperature'].max()} °C")
with c3: st.metric("Min Temperature", f"{df['Temperature'].min()} °C")
with c4: st.metric("Total Rainfall", f"{df['Rainfall'].sum():.1f} mm")

c5, c6, c7 = st.columns(3)
with c5: st.metric("Average Humidity", f"{df['Humidity'].mean():.1f}%")
with c6: st.metric("Average Wind Speed", f"{df['Wind_Speed'].mean():.1f} km/h")
with c7: st.metric("Total Days", len(df))

# --- GRAPHS ---
st.subheader("Temperature Trend Over Time")
st.line_chart(df.set_index("Date")["Temperature"])

st.subheader("Temperature vs Humidity")
st.scatter_chart(df, x="Temperature", y="Humidity")

st.subheader("Rainfall Over Time")
st.bar_chart(df.set_index("Date")["Rainfall"])

st.subheader("Wind Speed Over Time")
st.line_chart(df.set_index("Date")["Wind_Speed"])

st.subheader("Monthly Summary")
df["Month"] = df["Date"].dt.month
st.bar_chart(df.groupby("Month")["Temperature"].mean())

# --- EXTREME DAYS ---
st.subheader("Extreme Weather Days")
col1, col2 = st.columns(2)
with col1:
    st.write("Top 5 Hottest Days")
    st.table(df.sort_values("Temperature", ascending=False).head(5)[["Date", "Temperature", "Humidity", "Rainfall"]])
with col2:
    st.write("Top 5 Rainiest Days")
    st.table(df.sort_values("Rainfall", ascending=False).head(5)[["Date", "Temperature", "Humidity", "Rainfall"]])

# --- FULL TABLE ---
st.subheader("Full Weather Data")
st.dataframe(df, use_container_width=True)

# --- MACHINE LEARNING PREDICTION ---
st.subheader("Machine Learning Prediction - Tomorrow's Temperature")

# Features: Humidity, Rainfall, Wind_Speed, Pressure -> Target: Temperature
X = df[["Humidity", "Rainfall", "Wind_Speed", "Pressure"]]
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

st.write(f"Model Accuracy (R2 Score): {score:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f} °C")

st.subheader("Predict Temperature - Enter Today's Weather")
hum = st.number_input("Humidity (%)", 0, 100, 50)
rain = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)
wind = st.number_input("Wind Speed (km/h)", 0, 50, 10)
pressure = st.number_input("Pressure (hPa)", 950, 1050, 1010)

if st.button("Predict Temperature"):
    pred = model.predict([[hum, rain, wind, pressure]])[0]
    st.success(f"Predicted Temperature: {pred:.1f} °C")
    if pred > 35:
        st.warning("Very Hot Day - Stay Hydrated")
    elif pred < 15:
        st.warning("Cold Day - Wear Warm Clothes")
    elif rain > 5:
        st.warning("Rain Expected - Carry Umbrella")
    else:
        st.info("Pleasant Weather")

# --- DOWNLOAD ---
st.download_button("Download Analysis as CSV", df.to_csv(index=False), "weather_analysis.csv", "text/csv")
