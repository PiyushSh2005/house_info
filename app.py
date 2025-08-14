# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import joblib

# # # # # # Load data and models
# # # # # data = pd.read_csv("Housing_RL (1).csv")
# # # # # model = joblib.load("model.pkl")
# # # # # scaler = joblib.load("scaler.pkl")

# # # # # # App layout
# # # # # st.set_page_config(layout="wide")
# # # # # st.title("🏠 Housing Price Finder")

# # # # # # Section 1: Input Budget
# # # # # st.header("Enter Your Budget")
# # # # # budget = st.number_input("Maximum budget (in ₹)", min_value=100000, step=50000)

# # # # # if st.button("Find Houses"):
# # # # #     filtered_data = data[data["Price"] <= budget]
    
# # # # #     if not filtered_data.empty:
# # # # #         st.success(f"Found {len(filtered_data)} houses within your budget!")
# # # # #         st.dataframe(filtered_data)

# # # # #         # Optional: Predictions (if required)
# # # # #         # X = scaler.transform(filtered_data.drop(columns=['Price']))
# # # # #         # preds = model.predict(X)
# # # # #         # filtered_data['Predicted Price'] = preds
# # # # #         # st.dataframe(filtered_data)

# # # # #     else:
# # # # #         st.warning("No houses found within your budget. Try increasing the budget.")




# # # # import streamlit as st
# # # # import pandas as pd
# # # # import joblib
# # # # import numpy as np

# # # # # Page config (call this before other Streamlit commands)
# # # # st.set_page_config(page_title="Housing Price Finder", layout="wide")

# # # # st.title("🏠 Housing Price Finder")

# # # # # -----------------------
# # # # # Load dataset & models
# # # # # -----------------------
# # # # try:
# # # #     data = pd.read_csv("Housing_RL (1).csv")
# # # # except Exception as e:
# # # #     st.error(f"Could not load dataset 'Housing_RL (1).csv': {e}")
# # # #     st.stop()

# # # # # attempt to load model and scaler but don't fail app if missing
# # # # model = None
# # # # scaler = None
# # # # try:
# # # #     model = joblib.load("model.pkl")
# # # # except Exception:
# # # #     model = None

# # # # try:
# # # #     scaler = joblib.load("scaler.pkl")
# # # # except Exception:
# # # #     scaler = None

# # # # # -----------------------
# # # # # Normalize column names
# # # # # -----------------------
# # # # # strip whitespace, lowercase, replace spaces with underscores
# # # # data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

# # # # # Optional debug: show columns & sample
# # # # if st.checkbox("Show dataset columns & sample rows"):
# # # #     st.write("Columns:", list(data.columns))
# # # #     st.dataframe(data.head())

# # # # # -----------------------
# # # # # Detect likely column names
# # # # # -----------------------
# # # # price_col = next((c for c in data.columns if "price" in c), None)
# # # # car_col = next((c for c in data.columns if "car" in c), None)            # <- uses car_col
# # # # rooms_col = next((c for c in data.columns if "room" in c), None)
# # # # # search for several possibilities for house/type column
# # # # type_col = next((c for c in data.columns if any(k in c for k in ("type", "house_type", "property_type"))), None)

# # # # # Convert likely numeric columns to numeric (coerce errors to NaN)
# # # # for col in (price_col, car_col, rooms_col):
# # # #     if col:
# # # #         data[col] = pd.to_numeric(data[col], errors="coerce")

# # # # # -----------------------
# # # # # Inputs
# # # # # -----------------------
# # # # st.header("Enter Your Requirements")

# # # # # Budget input defaults to a reasonable value if price column exists
# # # # if price_col and not data[price_col].isna().all():
# # # #     min_price = int(data[price_col].min(skipna=True))
# # # #     max_price = int(data[price_col].max(skipna=True))
# # # #     default_budget = min(max_price, max(min_price, 1000000))
# # # #     budget = st.number_input("Maximum budget (₹)", min_value=0, value=default_budget, step=50000)
# # # # else:
# # # #     budget = st.number_input("Maximum budget (₹)", min_value=0, value=1000000, step=50000)
# # # #     st.info("No price column auto-detected; price filter will still attempt to run if a 'price' column appears.")

# # # # # Car spaces (if column found)
# # # # if car_col:
# # # #     car_spaces = st.number_input("Minimum car spaces required", min_value=0, value=0, step=1)
# # # # else:
# # # #     car_spaces = None
# # # #     st.info("No 'car' column auto-detected; car-space filter will be skipped.")

# # # # # Rooms (if column found)
# # # # if rooms_col:
# # # #     rooms = st.number_input("Minimum number of rooms required", min_value=0, value=1, step=1)
# # # # else:
# # # #     rooms = None
# # # #     st.info("No 'rooms' column auto-detected; rooms filter will be skipped.")

# # # # # House type (if available)
# # # # if type_col:
# # # #     type_options = list(data[type_col].dropna().unique())
# # # #     type_options_sorted = sorted(type_options)
# # # #     type_options_sorted.insert(0, "Any")
# # # #     house_type = st.selectbox("House type", options=type_options_sorted)
# # # # else:
# # # #     house_type = "Any"
# # # #     st.info("No 'type' column auto-detected; type filter will be skipped.")

# # # # # -----------------------
# # # # # Filtering & results
# # # # # -----------------------
# # # # if st.button("Find Houses"):
# # # #     filtered = data.copy()

# # # #     # Apply filters only when corresponding columns were detected
# # # #     if price_col:
# # # #         filtered = filtered[filtered[price_col] <= budget]

# # # #     if car_col and car_spaces is not None:
# # # #         filtered = filtered[filtered[car_col] >= car_spaces]

# # # #     if rooms_col and rooms is not None:
# # # #         filtered = filtered[filtered[rooms_col] >= rooms]

# # # #     if type_col and house_type != "Any":
# # # #         # exact match; adjust if you want case-insensitive or partial matching
# # # #         filtered = filtered[filtered[type_col] == house_type]

# # # #     if filtered.empty:
# # # #         st.warning("No houses found matching your criteria. Try adjusting your requirements.")
# # # #     else:
# # # #         st.success(f"Found {len(filtered)} houses matching your requirements!")
# # # #         st.dataframe(filtered.reset_index(drop=True))

# # # #         # Optional: run model predictions if model & scaler are available
# # # #         if model is not None and scaler is not None:
# # # #             try:
# # # #                 # Build a numeric feature matrix from filtered data.
# # # #                 # This is a best-effort: we select numeric columns and drop the target price if present.
# # # #                 X = filtered.select_dtypes(include=[np.number]).copy()
# # # #                 if price_col and price_col in X.columns:
# # # #                     X = X.drop(columns=[price_col], errors="ignore")
# # # #                 if X.shape[1] == 0:
# # # #                     st.info("No numeric features available for prediction; prediction skipped.")
# # # #                 else:
# # # #                     X_scaled = scaler.transform(X)
# # # #                     preds = model.predict(X_scaled)
# # # #                     filtered_with_preds = filtered.copy().reset_index(drop=True)
# # # #                     filtered_with_preds["predicted_price"] = preds
# # # #                     st.markdown("### Model predictions (best-effort)")
# # # #                     st.dataframe(filtered_with_preds)
# # # #             except Exception as e:
# # # #                 st.info(f"Model prediction skipped due to error: {e}")


# # # import streamlit as st
# # # import pandas as pd
# # # import joblib
# # # import numpy as np

# # # # -----------------------
# # # # Page config
# # # # -----------------------
# # # st.set_page_config(page_title="Housing Price Finder", layout="wide")
# # # st.title("🏠 Housing Price Finder")

# # # # -----------------------
# # # # Load dataset & models
# # # # -----------------------
# # # try:
# # #     data = pd.read_csv("Housing_RL (1).csv")  # Keep original columns for model
# # # except Exception as e:
# # #     st.error(f"Could not load dataset: {e}")
# # #     st.stop()

# # # # Load model and scaler (optional)
# # # try:
# # #     model = joblib.load("model.pkl")
# # # except Exception:
# # #     model = None

# # # try:
# # #     scaler = joblib.load("scaler.pkl")
# # # except Exception:
# # #     scaler = None

# # # # -----------------------
# # # # Create lowercase copy for filtering
# # # # -----------------------
# # # data_lower = data.copy()
# # # data_lower.columns = [c.strip().lower().replace(" ", "_") for c in data_lower.columns]

# # # # Optional debug: show columns
# # # if st.checkbox("Show dataset columns & sample"):
# # #     st.write("Original columns:", list(data.columns))
# # #     st.write("Lowercase columns:", list(data_lower.columns))
# # #     st.dataframe(data_lower.head())

# # # # -----------------------
# # # # Detect columns (lowercase version)
# # # # -----------------------
# # # price_col_l = next((c for c in data_lower.columns if "price" in c), None)
# # # car_col_l = next((c for c in data_lower.columns if "car" in c), None)
# # # rooms_col_l = next((c for c in data_lower.columns if "room" in c), None)
# # # type_col_l = next((c for c in data_lower.columns if any(k in c for k in ("type", "house_type", "property_type"))), None)

# # # # -----------------------
# # # # Inputs
# # # # -----------------------
# # # st.header("Enter Your Requirements")

# # # # Budget input
# # # budget = st.number_input("Maximum budget (₹)", min_value=0, value=1000000, step=50000)

# # # # Car spaces
# # # if car_col_l:
# # #     car_spaces = st.number_input("Minimum car spaces required", min_value=0, value=0, step=1)
# # # else:
# # #     car_spaces = None
# # #     st.info("No 'car' column detected; car-space filter skipped.")

# # # # Rooms
# # # if rooms_col_l:
# # #     rooms = st.number_input("Minimum number of rooms required", min_value=0, value=1, step=1)
# # # else:
# # #     rooms = None
# # #     st.info("No 'rooms' column detected; rooms filter skipped.")

# # # # House type
# # # if type_col_l:
# # #     house_types = sorted(data_lower[type_col_l].dropna().unique())
# # #     house_types.insert(0, "Any")
# # #     house_type = st.selectbox("House type", options=house_types)
# # # else:
# # #     house_type = "Any"
# # #     st.info("No 'house type' column detected; type filter skipped.")

# # # # -----------------------
# # # # Filtering
# # # # -----------------------
# # # if st.button("Find Houses"):
# # #     filtered_lower = data_lower.copy()

# # #     if price_col_l:
# # #         filtered_lower = filtered_lower[filtered_lower[price_col_l] <= budget]
# # #     if car_col_l and car_spaces is not None:
# # #         filtered_lower = filtered_lower[filtered_lower[car_col_l] >= car_spaces]
# # #     if rooms_col_l and rooms is not None:
# # #         filtered_lower = filtered_lower[filtered_lower[rooms_col_l] >= rooms]
# # #     if type_col_l and house_type != "Any":
# # #         filtered_lower = filtered_lower[filtered_lower[type_col_l] == house_type]

# # #     if filtered_lower.empty:
# # #         st.warning("No houses found matching your criteria.")
# # #     else:
# # #         st.success(f"Found {len(filtered_lower)} houses matching your requirements!")

# # #         # Map back to original DataFrame rows for display & prediction
# # #         filtered_original = data.loc[filtered_lower.index]

# # #         st.markdown("### Filtered Houses")
# # #         st.dataframe(filtered_original.reset_index(drop=True))

# # #         # -----------------------
# # #         # Model Predictions
# # #         # -----------------------
# # #         if model is not None and scaler is not None:
# # #             try:
# # #                 # Select same feature columns as training
# # #                 # We assume model trained on all numeric columns except target price
# # #                 X = filtered_original.select_dtypes(include=[np.number]).copy()

# # #                 if price_col_l:
# # #                     # Get original price column name from lowercase mapping
# # #                     original_price_col = data.columns[list(data_lower.columns).index(price_col_l)]
# # #                     X = X.drop(columns=[original_price_col], errors="ignore")

# # #                 # Scale & predict
# # #                 X_scaled = scaler.transform(X)
# # #                 preds = model.predict(X_scaled)

# # #                 # Add predictions to display
# # #                 results = filtered_original.copy()
# # #                 results["Predicted Price"] = preds
# # #                 st.markdown("### Model Predictions")
# # #                 st.dataframe(results.reset_index(drop=True))
# # #             except Exception as e:
# # #                 st.info(f"Model prediction skipped due to error: {e}")



# # import streamlit as st
# # import pandas as pd
# # import joblib
# # import numpy as np

# # # -----------------------
# # # Page config
# # # -----------------------
# # st.set_page_config(page_title="Housing Price Finder", layout="wide")
# # st.title("🏠 Housing Price Finder")

# # # -----------------------
# # # Load dataset & models
# # # -----------------------
# # try:
# #     data = pd.read_csv("Housing_RL (1).csv")  # Keep original columns for model prediction
# # except Exception as e:
# #     st.error(f"Could not load dataset: {e}")
# #     st.stop()

# # # Load model and scaler
# # try:
# #     model = joblib.load("model.pkl")
# # except Exception:
# #     model = None

# # try:
# #     scaler = joblib.load("scaler.pkl")
# # except Exception:
# #     scaler = None

# # # -----------------------
# # # Lowercase copy for filtering
# # # -----------------------
# # data_lower = data.copy()
# # data_lower.columns = [c.strip().lower().replace(" ", "_") for c in data_lower.columns]

# # # Debug: optional
# # if st.checkbox("Show dataset columns & sample"):
# #     st.write("Original columns:", list(data.columns))
# #     st.write("Lowercase columns:", list(data_lower.columns))
# #     st.dataframe(data_lower.head())

# # # -----------------------
# # # Detect columns for filtering
# # # -----------------------
# # price_col_l = next((c for c in data_lower.columns if "price" in c), None)
# # car_col_l = next((c for c in data_lower.columns if "car" in c), None)
# # rooms_col_l = next((c for c in data_lower.columns if "room" in c), None)
# # type_col_l = next((c for c in data_lower.columns if any(k in c for k in ("type", "house_type", "property_type"))), None)

# # # -----------------------
# # # User Inputs
# # # -----------------------
# # st.header("Enter Your Requirements")

# # budget = st.number_input("Maximum budget (₹)", min_value=0, value=1000000, step=50000)

# # if car_col_l:
# #     car_spaces = st.number_input("Minimum car spaces required", min_value=0, value=0, step=1)
# # else:
# #     car_spaces = None
# #     st.info("No 'car' column detected; skipping filter.")

# # if rooms_col_l:
# #     rooms = st.number_input("Minimum number of rooms required", min_value=0, value=1, step=1)
# # else:
# #     rooms = None
# #     st.info("No 'rooms' column detected; skipping filter.")

# # if type_col_l:
# #     house_types = sorted(data_lower[type_col_l].dropna().unique())
# #     house_types.insert(0, "Any")
# #     house_type = st.selectbox("House type", options=house_types)
# # else:
# #     house_type = "Any"
# #     st.info("No 'house type' column detected; skipping filter.")

# # # -----------------------
# # # Filtering
# # # -----------------------
# # if st.button("Find Houses"):
# #     filtered_lower = data_lower.copy()

# #     if price_col_l:
# #         filtered_lower = filtered_lower[filtered_lower[price_col_l] <= budget]
# #     if car_col_l and car_spaces is not None:
# #         filtered_lower = filtered_lower[filtered_lower[car_col_l] >= car_spaces]
# #     if rooms_col_l and rooms is not None:
# #         filtered_lower = filtered_lower[filtered_lower[rooms_col_l] >= rooms]
# #     if type_col_l and house_type != "Any":
# #         filtered_lower = filtered_lower[filtered_lower[type_col_l] == house_type]

# #     if filtered_lower.empty:
# #         st.warning("No houses found matching your criteria.")
# #     else:
# #         st.success(f"Found {len(filtered_lower)} houses matching your requirements!")

# #         # Map back to original DataFrame for display and prediction
# #         filtered_original = data.loc[filtered_lower.index]

# #         st.markdown("### Filtered Houses")
# #         st.dataframe(filtered_original.reset_index(drop=True))

# #         # -----------------------
# #         # # Model Predictions
# #         # # -----------------------
# #         # if model is not None and scaler is not None:
# #         #     try:
# #         #         # Align columns with model's training features
# #         #         expected_cols = list(model.feature_names_in_)
# #         #         X = filtered_original.reindex(columns=expected_cols, fill_value=0)

# #         #         # Scale features if scaler is available
# #         #         X_scaled = scaler.transform(X)
# #         #         preds = model.predict(X_scaled)

# #         #         results = filtered_original.copy().reset_index(drop=True)
# #         #         results["Predicted Price"] = preds
# #         #         st.markdown("### Model Predictions")
# #         #         st.dataframe(results)
# #         #     except Exception as e:
# #         #         st.info(f"Model prediction skipped due to error: {e}")

# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import base64
# import matplotlib.pyplot as plt

# # -----------------------
# # Function to set background
# # -----------------------
# def set_bg(image_file):
#     with open(image_file, "rb") as f:
#         data = f.read()
#     encoded = base64.b64encode(data).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Set background
# set_bg("downloads.jpg")

# # -----------------------
# # Page config
# # -----------------------
# st.set_page_config(page_title="Housing Price Finder", layout="wide")
# st.title("🏠 Housing Price Finder")

# # -----------------------
# # Load dataset & models
# # -----------------------
# try:
#     data = pd.read_csv("Housing_RL (1).csv")
# except Exception as e:
#     st.error(f"Could not load dataset: {e}")
#     st.stop()

# try:
#     model = joblib.load("model.pkl")
# except Exception:
#     model = None

# try:
#     scaler = joblib.load("scaler.pkl")
# except Exception:
#     scaler = None

# # -----------------------
# # Lowercase copy for filtering
# # -----------------------
# data_lower = data.copy()
# data_lower.columns = [c.strip().lower().replace(" ", "_") for c in data_lower.columns]

# # Function to map lowercase column name to original
# def get_original_colname(lower_name, original_df):
#     for col in original_df.columns:
#         if col.strip().lower().replace(" ", "_") == lower_name:
#             return col
#     return None

# # Detect columns
# price_col_l = next((c for c in data_lower.columns if "price" in c), None)
# car_col_l = next((c for c in data_lower.columns if "car" in c), None)
# rooms_col_l = next((c for c in data_lower.columns if "room" in c), None)
# type_col_l = next((c for c in data_lower.columns if any(k in c for k in ("type", "house_type", "property_type"))), None)

# # Get original names
# price_col = get_original_colname(price_col_l, data)
# car_col = get_original_colname(car_col_l, data)
# rooms_col = get_original_colname(rooms_col_l, data)
# type_col = get_original_colname(type_col_l, data)

# # -----------------------
# # Layout: Left and Right
# # -----------------------
# left_col, right_col = st.columns([1, 1])

# with left_col:
#     st.header("Enter Your Requirements")
#     budget = st.number_input("Maximum budget (₹)", min_value=0, value=1000000, step=50000)

#     if car_col:
#         car_spaces = st.number_input("Minimum car spaces required", min_value=0, value=0, step=1)
#     else:
#         car_spaces = None
#         st.info("No 'car' column detected; skipping filter.")

#     if rooms_col:
#         rooms = st.number_input("Minimum number of rooms required", min_value=0, value=1, step=1)
#     else:
#         rooms = None
#         st.info("No 'rooms' column detected; skipping filter.")

#     if type_col:
#         house_types = sorted(data[type_col].dropna().unique())
#         house_types.insert(0, "Any")
#         house_type = st.selectbox("House type", options=house_types)
#     else:
#         house_type = "Any"
#         st.info("No 'house type' column detected; skipping filter.")

#     # -----------------------
#     # Filtering
#     # -----------------------
#     if st.button("Find Houses"):
#         filtered_df = data.copy()

#         if price_col:
#             filtered_df = filtered_df[filtered_df[price_col] <= budget]
#         if car_col and car_spaces is not None:
#             filtered_df = filtered_df[filtered_df[car_col] >= car_spaces]
#         if rooms_col and rooms is not None:
#             filtered_df = filtered_df[filtered_df[rooms_col] >= rooms]
#         if type_col and house_type != "Any":
#             filtered_df = filtered_df[filtered_df[type_col] == house_type]

#         if filtered_df.empty:
#             st.warning("No houses found matching your criteria.")
#             st.session_state.filtered_data = None
#         else:
#             st.success(f"Found {len(filtered_df)} houses matching your requirements!")
#             st.markdown("### Filtered Houses")
#             st.dataframe(filtered_df.reset_index(drop=True))
#             st.session_state.filtered_data = filtered_df
#     else:
#         st.session_state.filtered_data = None

# with right_col:
#     st.header("📊 Graphs & Analysis")
#     if st.session_state.get("filtered_data") is not None:
#         filtered_df = st.session_state.filtered_data

#         # Example 1: Price distribution
#         if price_col:
#             fig, ax = plt.subplots()
#             ax.hist(filtered_df[price_col], bins=10, edgecolor='black')
#             ax.set_title("Price Distribution")
#             ax.set_xlabel("Price")
#             ax.set_ylabel("Count")
#             st.pyplot(fig)

#         # Example 2: Rooms vs Price
#         if rooms_col and price_col:
#             fig2, ax2 = plt.subplots()
#             ax2.scatter(filtered_df[rooms_col], filtered_df[price_col])
#             ax2.set_title("Rooms vs Price")
#             ax2.set_xlabel("Rooms")
#             ax2.set_ylabel("Price")
#             st.pyplot(fig2)
#     else:
#         st.info("Run a search to see graphs here.")
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import matplotlib.pyplot as plt

# -----------------------
# Function to set background
# -----------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_bg("downloaded.jpg")

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Housing Price Finder", layout="wide")
st.title("🏠 Housing Price Finder")

# -----------------------
# Load dataset & models
# -----------------------
try:
    data = pd.read_csv("Housing_RL (1).csv")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

try:
    model = joblib.load("model.pkl")
except Exception:
    model = None

try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    scaler = None

# -----------------------
# Lowercase copy for filtering
# -----------------------
data_lower = data.copy()
data_lower.columns = [c.strip().lower().replace(" ", "_") for c in data_lower.columns]

# Function to map lowercase column name to original
def get_original_colname(lower_name, original_df):
    for col in original_df.columns:
        if col.strip().lower().replace(" ", "_") == lower_name:
            return col
    return None

# Detect columns
price_col_l = next((c for c in data_lower.columns if "price" in c), None)
car_col_l = next((c for c in data_lower.columns if "car" in c), None)
rooms_col_l = next((c for c in data_lower.columns if "room" in c), None)
type_col_l = next((c for c in data_lower.columns if any(k in c for k in ("type", "house_type", "property_type"))), None)

# Get original names
price_col = get_original_colname(price_col_l, data)
car_col = get_original_colname(car_col_l, data)
rooms_col = get_original_colname(rooms_col_l, data)
type_col = get_original_colname(type_col_l, data)

# -----------------------
# Layout: Left and Right
# -----------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Enter Your Requirements")
    budget = st.number_input("Maximum budget (₹)", min_value=0, value=1000000, step=50000)

    if car_col:
        car_spaces = st.number_input("Minimum car spaces required", min_value=0, value=0, step=1)
    else:
        car_spaces = None
        st.info("No 'car' column detected; skipping filter.")

    if rooms_col:
        rooms = st.number_input("Minimum number of rooms required", min_value=0, value=1, step=1)
    else:
        rooms = None
        st.info("No 'rooms' column detected; skipping filter.")

    if type_col:
        house_types = sorted(data[type_col].dropna().unique())
        house_types.insert(0, "Any")
        house_type = st.selectbox("House type", options=house_types)
    else:
        house_type = "Any"
        st.info("No 'house type' column detected; skipping filter.")

    # -----------------------
    # Filtering
    # -----------------------
    if st.button("Find Houses"):
        filtered_df = data.copy()

        if price_col:
            filtered_df = filtered_df[filtered_df[price_col] <= budget]
        if car_col and car_spaces is not None:
            filtered_df = filtered_df[filtered_df[car_col] >= car_spaces]
        if rooms_col and rooms is not None:
            filtered_df = filtered_df[filtered_df[rooms_col] >= rooms]
        if type_col and house_type != "Any":
            filtered_df = filtered_df[filtered_df[type_col] == house_type]

        if filtered_df.empty:
            st.warning("No houses found matching your criteria.")
            st.session_state.filtered_data = None
        else:
            st.success(f"Found {len(filtered_df)} houses matching your requirements!")
            st.markdown("### Filtered Houses")
            st.dataframe(filtered_df.reset_index(drop=True))
            st.session_state.filtered_data = filtered_df

            # -----------------------
            # Download button for report
            # -----------------------
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report",
                data=csv,
                file_name="housing_report.csv",
                mime="text/csv"
            )
    else:
        st.session_state.filtered_data = None

with right_col:
    st.header("📊 Graphs & Analysis")
    if st.session_state.get("filtered_data") is not None:
        filtered_df = st.session_state.filtered_data

        # Example 1: Price distribution
        if price_col:
            fig, ax = plt.subplots()
            ax.hist(filtered_df[price_col], bins=10, edgecolor='black')
            ax.set_title("Price Distribution")
            ax.set_xlabel("Price")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Example 2: Rooms vs Price
        if rooms_col and price_col:
            fig2, ax2 = plt.subplots()
            ax2.scatter(filtered_df[rooms_col], filtered_df[price_col])
            ax2.set_title("Rooms vs Price")
            ax2.set_xlabel("Rooms")
            ax2.set_ylabel("Price")
            st.pyplot(fig2)
    else:
        st.info("Run a search to see graphs here.")
























