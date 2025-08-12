# import streamlit as st
# import pandas as pd
# import joblib

# # Load data and models
# data = pd.read_csv("Housing_RL (1).csv")
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# # App layout
# st.set_page_config(layout="wide")
# st.title("🏠 Housing Price Finder")

# # Section 1: Input Budget
# st.header("Enter Your Budget")
# budget = st.number_input("Maximum budget (in ₹)", min_value=100000, step=50000)

# if st.button("Find Houses"):
#     filtered_data = data[data["Price"] <= budget]
    
#     if not filtered_data.empty:
#         st.success(f"Found {len(filtered_data)} houses within your budget!")
#         st.dataframe(filtered_data)

#         # Optional: Predictions (if required)
#         # X = scaler.transform(filtered_data.drop(columns=['Price']))
#         # preds = model.predict(X)
#         # filtered_data['Predicted Price'] = preds
#         # st.dataframe(filtered_data)

#     else:
#         st.warning("No houses found within your budget. Try increasing the budget.")




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

# Section 1: Buyer Requirements
st.header("Enter Your Requirements")

# Budget input
budget = st.number_input("Maximum budget (in ₹)", min_value=100000, step=50000)

# Parking requirement
parking = st.number_input("Minimum Parking Spaces Required", min_value=0, step=1)

# Rooms requirement
rooms = st.number_input("Minimum Number of Rooms Required", min_value=1, step=1)

# House type selection (adjust options based on your dataset)
house_types = data["House_Type"].unique() if "House_Type" in data.columns else []
house_type = st.selectbox("Select Type of House", options=house_types)

# Filter button
if st.button("Find Houses"):
    filtered_data = data[
        (data["Price"] <= budget) &
        (data["Parking"] >= parking) &
        (data["Rooms"] >= rooms) &
        (data["House_Type"] == house_type)
    ]
    
    if not filtered_data.empty:
        st.success(f"Found {len(filtered_data)} houses matching your requirements!")
        st.dataframe(filtered_data)

        # Optional ML predictions
        # X = scaler.transform(filtered_data.drop(columns=['Price']))
        # preds = model.predict(X)
        # filtered_data['Predicted Price'] = preds
        # st.dataframe(filtered_data)

    else:
        st.warning("No houses found matching your criteria. Try adjusting your requirements.")



