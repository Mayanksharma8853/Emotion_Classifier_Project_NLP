import streamlit as st
import pickle

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Emotion Classifier",
    layout="centered"
)

# =========================
# Custom CSS (Professional Background)
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f6f9;
    }

    h1, h2, h3, p, label {
        color: #1f2937 !important;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        color: black;
        border-radius: 10px;
        border: 1px solid #d1d5db;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load Saved Objects
# =========================
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# =========================
# Emotion Mapping
# =========================
emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

emotion_color = {
    "sadness": "#3b82f6",
    "joy": "#16a34a",
    "love": "#dc2626",
    "anger": "#ea580c",
    "fear": "#4b5563",
    "surprise": "#0ea5e9"
}

# =========================
# App Header
# =========================
st.markdown("<h1 style='text-align: center;'>Emotion Classification App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Detect emotions from text using NLP</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# User Input
# =========================
user_text = st.text_area(
    "Enter your text:",
    height=150,
    placeholder="I am feeling very happy today!"
)

# =========================
# Prediction
# =========================
if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        user_vector = vectorizer.transform([user_text])
        prediction = model.predict(user_vector)

        pred_label = int(prediction[0])
        emotion = emotion_map.get(pred_label, "Unknown")
        color = emotion_color.get(emotion, "#000000")

        st.markdown(
            f"""
            <div style="
                background-color:#ffffff;
                padding:15px;
                border-radius:10px;
                text-align:center;
                border-left:6px solid {color};
                font-size:22px;
                font-weight:bold;
                color:#111827;
            ">
                Predicted Emotion: {emotion.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: #6b7280;'>Built with Streamlit, TF-IDF and Logistic Regression</p>",
    unsafe_allow_html=True
)
