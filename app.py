import streamlit as st
import pandas as pd
import joblib

# Load data and models
data = pd.read_csv("Housing_RL (1).csv")
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App layout
st.set_page_config(layout="wide")
st.title("🏠 Housing Price Finder")

# Section 1: Input Budget
st.header("Enter Your Budget")
budget = st.number_input("Maximum budget (in ₹)", min_value=100000, step=50000)

if st.button("Find Houses"):
    filtered_data = data[data["Price"] <= budget]
    
    if not filtered_data.empty:
        st.success(f"Found {len(filtered_data)} houses within your budget!")
        st.dataframe(filtered_data)

        # Optional: Predictions (if required)
        # X = scaler.transform(filtered_data.drop(columns=['Price']))
        # preds = model.predict(X)
        # filtered_data['Predicted Price'] = preds
        # st.dataframe(filtered_data)

    else:
        st.warning("No houses found within your budget. Try increasing the budget.")


