import streamlit as st
import pandas as pd
import joblib

# Load Saved Model Files
model = joblib.load("LR_insurance.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# Page Configuration
st.set_page_config(
    page_title="🏥 Insurance Cost Predictor",
    page_icon="💰",
    layout="wide"
)

# Sidebar
st.sidebar.title("🏥 Insurance Cost Prediction")
st.sidebar.write(" Optional Assignment 21")
st.sidebar.write(" Linear Regression Model")
st.sidebar.write(" Developed using Streamlit")

# Main Title
st.title("🏥 Insurance Cost Prediction System")
st.write("📋 Enter the details below to predict your insurance expenses.")

st.divider()

# Input Columns
col1, col2 = st.columns(2)

# Numerical Inputs
with col1:

    st.subheader("📊 Personal Information")

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "👨‍👩‍👧 Children",
        min_value=0,
        max_value=10,
        value=0
    )

# Categorical Inputs
with col2:

    st.subheader("📝 Additional Information")

    sex = st.selectbox(
        "👤 Gender",
        ["male", "female"]
    )

    smoker = st.selectbox(
        "🚬 Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "📍 Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

# Prediction Button
predict = st.button(
    "🔮 Predict Insurance Cost",
    use_container_width=True
)

# Prediction
if predict:

    # Create Input DataFrame
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Convert Categorical Data into Numerical Data
    input_df = pd.get_dummies(input_df)

    # Match Training Columns
    input_df = input_df.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Scale Numerical Columns
    numerical_cols = [
        "age",
        "bmi",
        "children"
    ]

    input_df[numerical_cols] = scaler.transform(
        input_df[numerical_cols]
    )

    # Predict
    prediction = model.predict(input_df)

    # Display Result
    st.success(
        f"🎉 Predicted Insurance Cost: ₹{prediction[0]:,.2f}"
    )

    st.balloons()

# Footer
st.divider()

st.caption(
    " Developed by Harish Marne | "
    " AIML Optional Assignment 21"
)