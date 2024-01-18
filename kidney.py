import streamlit as st
from PIL import Image, ImageOps
import os
import numpy as np
from keras.models import load_model

st.title('Image Recognition')

img_file = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])

def load_img(img):
    img = Image.open(img)
    return img

if img_file is not None:
    file_details = {}
    file_details['name'] = img_file.name
    file_details['size'] = img_file.size
    file_details['type'] = img_file.type
    st.write(file_details)
    st.image(load_img(img_file), width=255)

    with open(os.path.join('uploads', 'src.jpg'), 'wb') as f:
        f.write(img_file.getbuffer())

    st.success('Image Saved')

    # Load the trained model for kidney prediction
    kidney_model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    # Load the image for kidney prediction
    kidney_image = Image.open('uploads/src.jpg').convert("RGB")
    size = (224, 224)
    kidney_image = ImageOps.fit(kidney_image, size, Image.Resampling.LANCZOS)
    kidney_image_array = np.asarray(kidney_image)
    normalized_kidney_image_array = (kidney_image_array.astype(np.float32) / 127.5) - 1
    kidney_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    kidney_data[0] = normalized_kidney_image_array

    # Predict kidney image
    kidney_prediction = kidney_model.predict(kidney_data)
    kidney_index = np.argmax(kidney_prediction)
    kidney_class_name = class_names[kidney_index]
    kidney_confidence_score = kidney_prediction[0][kidney_index]

    # Display kidney prediction and confidence score
    st.success('Predicted Class (Kidney): ' + kidney_class_name[2:])
    st.warning('Confidence Score: ' + str(kidney_confidence_score))
