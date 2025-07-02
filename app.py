import streamlit as st
import pickle
import numpy as np

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️")

# --- Model and Vectorizer Loading ---
# Using a function with caching to load resources efficiently
@st.cache_resource
def load_model_and_vectorizer():
    """Loads the saved model and vectorizer from disk."""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        # Return None if files are not found
        return None, None

# Load the resources
model, vectorizer = load_model_and_vectorizer()

# --- Main App UI ---
st.title("🕵️ Fake Review Detector")
st.write(
    "Enter a hotel review below to determine if it is likely to be **genuine** or **deceptive**. "
    "The model was trained on a dataset of hotel reviews."
)

# Check if the model and vectorizer loaded successfully
if model is None or vectorizer is None:
    st.error("Model files not found. Please run the `model_training.py` script first to generate them.")
else:
    # Text area for user input
    user_input = st.text_area(
        "Enter the review text here:",
        height=150,
        placeholder="e.g., The hotel was fantastic! The staff were friendly and the room was clean..."
    )

    # Analyze button
    if st.button("Analyze Review"):
        if user_input.strip():
            # 1. Vectorize the user input
            input_vect = vectorizer.transform([user_input])

            # 2. Make a prediction
            prediction = model.predict(input_vect)
            prediction_proba = model.predict_proba(input_vect)

            # 3. Display the result
            st.subheader("Analysis Result")
            
            # Remember: deceptive = 0, truthful = 1
            if prediction[0] == 1:
                probability = prediction_proba[0][1]
                st.success(f"This review seems **Genuine / Truthful**.")
                st.write(f"**Confidence:** {probability:.2%}")
            else:
                probability = prediction_proba[0][0]
                st.warning(f"This review seems **Deceptive / Fake**.")
                st.write(f"**Confidence:** {probability:.2%}")
                
            # Optional: Show prediction probabilities in an expander
            with st.expander("See prediction details"):
                st.write("Prediction Probabilities:")
                st.write(f"- Genuine: {prediction_proba[0][1]:.2%}")
                st.write(f"- Deceptive: {prediction_proba[0][0]:.2%}")

        else:
            st.warning("Please enter a review to analyze.")