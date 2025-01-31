import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError  # Import the MSE loss function
from data_preprocessing import preprocess_data

# Load the trained models and preprocessor
@st.cache_resource
def load_models_and_preprocessor():
    # Load the machine learning model and preprocessor
    ml_model = joblib.load('best_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Load the deep learning model with custom objects
    dl_model = tf.keras.models.load_model(
        'deep_learning_model.h5',
        custom_objects={'mse': MeanSquaredError()}  # Pass the MSE loss function
    )

    return ml_model, dl_model, preprocessor

def main():
    st.title("Math Score Prediction App")
    st.write("This app predicts the math score based on student features using both machine learning and deep learning models.")

    # Input fields for features
    st.sidebar.header("Input Features")
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.sidebar.selectbox(
        "Parental Level of Education",
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    )
    lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

    # Create a DataFrame from the input
    input_data = pd.DataFrame({
        'gender': [gender],
        'race_ethnicity': [race_ethnicity],
        'parental_level_of_education': [parental_level_of_education],
        'lunch': [lunch],
        'test_preparation_course': [test_preparation_course]
    })

    # Display the input data
    st.write("### Input Data")
    st.write(input_data)

    # Load the models and preprocessor
    ml_model, dl_model, preprocessor = load_models_and_preprocessor()

    # Preprocess the input data using the saved preprocessor
    X_processed = preprocessor.transform(input_data)

    # Make predictions
    if st.button("Predict Math Score"):
        # Machine Learning Prediction
        ml_prediction = ml_model.predict(X_processed)
        st.write(f"### Predicted Math Score (Machine Learning): **{ml_prediction[0]:.2f}**")

        # Deep Learning Prediction
        dl_prediction = dl_model.predict(X_processed)
        st.write(f"### Predicted Math Score (Deep Learning): **{dl_prediction[0][0]:.2f}**")

if __name__ == "__main__":
    main()