import model_utils as mu
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

NAMES = [
    "Queratosis actínica",
    "Carcinoma de células basales",
    "Lesiones benignas similares a queratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevo melanocítico",
    "Lesiones vasculares"
]

checkpoint = st.file_uploader("Sube un checkpoint", type="ckpt")
uploaded = st.file_uploader("Sube una imagen", type="jpg")

if checkpoint is not None:
    model = mu.load_model(checkpoint)

if uploaded is not None:
    image = Image.open(uploaded)

clicked = st.button("Evaluar")

if clicked:
    probs, image_resized, overlay_img = mu.eval_image(model, image)
    
    probs_display = [str(np.round(x*100, 1)) + "%" for x in probs]
    idx = np.argmax(probs)
    name = NAMES[idx]
    df = pd.DataFrame(zip(NAMES, probs_display), columns=["Nombre", "Probabilidad"])

    st.markdown("### Predicción")
    st.markdown(f"{name}")

    st.markdown("### Probabilidades")
    st.table(df)

    st.markdown("### Imágenes")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Imagen")
        st.image(image_resized)
    with col2:
        st.markdown("Mapa de interés")
        st.image(overlay_img)