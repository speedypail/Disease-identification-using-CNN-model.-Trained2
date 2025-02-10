import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
model = tf.keras.models.load_model('trained_model.keras')
class_names = ['CCI_Caterpillars', 'CCI_Leaflets', 'WCLWD_DryingofLeaflets', 'WCLWD_Healthy', 'WCLWD_Yellowing']
st.title("Disease Prediction App")
st.write("Upload an image and let the model predict the disease category.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to RGB format for consistency
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(img_array, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for the model
    image_resized = image.resize((300, 300))  
    input_arr = img_to_array(image_resized)
    input_arr = np.array([input_arr])  
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = class_names[result_index]


    # Display the result
    st.write(f"Prediction: **{model_prediction}**")
