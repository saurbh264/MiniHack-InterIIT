import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from model import model_func

st.title("Image Search with Question")

uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

question = st.text_input("Enter your question")

n = st.number_input("How many top images to show?", min_value=1, max_value=10, value=5)

if st.button("Show Results"):

    if uploaded_image is not None and question:
        img = Image.open(uploaded_image)

        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        result_images = model_func(img, question, n)

        st.write(f"Top {n} images based on your query:")

        for idx, url in enumerate(result_images, 1):
            st.write(f"Image {idx}:")
            st.image(requests.get(url).content, use_column_width=True)

    else:
        st.warning("Please upload an image and enter a question.")

