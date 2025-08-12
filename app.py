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

# Normalize column names (case-insensitive matching)
data.columns = [col.strip().lower() for col in data.columns]

# Display available columns for debugging
st.write("Available columns in dataset:", list(data.columns))

# Expected column names (adjust if different in your dataset)
price_col = next((col for col in data.columns if "price" in col), None)
parking_col = next((col for col in data.columns if "parking" in col), None)
rooms_col = next((col for col in data.columns if "room" in col), None)
type_col = next((col for col in data.columns if "type" in col), None)

# App layout
st.set_page_config(layout="wide")
st.title("🏠 Housing Price Finder")

# Section 1: Buyer Requirements
st.header("Enter Your Requirements")

# Budget input
budget = st.number_input("Maximum budget (in ₹)", min_value=100000, step=50000)

# Parking requirement
if parking_col:
    parking = st.number_input("Minimum Parking Spaces Required", min_value=0, step=1)
else:
    st.warning("⚠ No 'Parking' column found in dataset.")
    parking = None

# Rooms requirement
if rooms_col:
    rooms = st.number_input("Minimum Number of Rooms Required", min_value=1, step=1)
else:
    st.warning("⚠ No 'Rooms' column found in dataset.")
    rooms = None

# House type selection
if type_col:
    house_types = data[type_col].unique()
    house_type = st.selectbox("Select Type of House", options=house_types)
else:
    st.warning("⚠ No 'House Type' column found in dataset.")
    house_type = None

# Filter button
if st.button("Find Houses"):
    filtered_data = data.copy()

    # Apply filters based on available columns
    if price_col:
        filtered_data = filtered_data[filtered_data[price_col] <= budget]
    if parking_col and parking is not None:
        filtered_data = filtered_data[filtered_data[parking_col] >= parking]
    if rooms_col and rooms is not None:
        filtered_data = filtered_data[filtered_data[rooms_col] >= rooms]
    if type_col and house_type is not None:
        filtered_data = filtered_data[filtered_data[type_col] == house_type]

    if not filtered_data.empty:
        st.success(f"Found {len(filtered_data)} houses matching your requirements!")
        st.dataframe(filtered_data)
    else:
        st.warning("No houses found matching your criteria. Try adjusting your requirements.")






