import streamlit as st

from PIL import Image

file_up = st.file_uploader("Upload an image", type="jpg")
image = Image.open(file_up)
st.image(image, caption='Uploaded Image.', use_column_width=True)