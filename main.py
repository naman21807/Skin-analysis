import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
API_key=os.getenv("API_KEY")

client = genai.Client(api_key=API_key)

def get_response(type):
    prompt = f"You are a professional dermatologist tell description characterstics immediate action plan if needed, recommended routines and  why it works and any pro tip for {type} don't type too much neither too less, just hit the sweet spot "
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    # print(response.text)
    return response.text


# Load model
model = load_model("skin_model.h5")

# Class labels (CHANGE THIS based on your dataset)
classes = ["dry", "normal", "oily"]

# Title
st.title("💆 Skin Type Predictor")

st.write("Upload an image to predict your skin type")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # MUST match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"Predicted Skin Type: **{predicted_class.upper()}**")
    st.write(f"Confidence: {confidence:.2f}")

    # Recommendations (simple logic)
    if predicted_class == "dry":
        # st.info("💧 Use heavy moisturizer, avoid harsh soaps, stay hydrated.")
        # st.info(get_response("Tell about dry skin type and how to improve"))
        st.write(get_response("dry skin"))
    elif predicted_class == "oily":
        # st.info(get_response("Tell about oily skin type and how to improve"))
        st.write(get_response("oily skin"))
    elif predicted_class == "normal":
        # st.info(get_response("Tell about normal skin type and how to maintain it and improve further"))
        st.write(get_response("normal skin"))