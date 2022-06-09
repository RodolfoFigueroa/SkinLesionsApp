import model_utils as mu
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

NAMES = [
    "Queratosis actínica",
    "Carcinoma de células basales",
    "Lesión benigna similar a queratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevo melanocítico",
    "Lesión vascular",
]

NAMES_BINARY = [
    "No melanoma",
    "Melanoma",
]

LABEL_MAP = {
    "akiec": 0,
    "bcc": 1, 
    "bkl": 2, 
    "df": 3, 
    "mel": 4, 
    "nv": 5, 
    "vasc": 6
}

LABEL_MAP_BINARY = {
    "akiec": 0,
    "bcc": 0, 
    "bkl": 0, 
    "df": 0, 
    "mel": 1, 
    "nv": 0, 
    "vasc": 0
}
         
checkpoint = st.file_uploader("Sube un checkpoint", type="ckpt")
uploaded = st.file_uploader("Sube una imagen", type="jpg")

check1, check2 = st.columns(2)
with check1:
    binary = st.checkbox("Binario")
with check2:
    evaluation = st.checkbox("Evaluación")

metadata = pd.read_csv("./HAM10000_metadata")
if binary:
    metadata["target"] = metadata["dx"].map(LABEL_MAP_BINARY)
else:
    metadata["target"] = metadata["dx"].map(LABEL_MAP)
    
clicked = st.button("Evaluar")

if clicked:
    model = mu.load_model(checkpoint, binary)
    image = Image.open(uploaded)
    
    probs, image_resized, overlay_img = mu.eval_image(model, image)
    
    if len(probs) == 2:
        names = NAMES_BINARY
    else:
        names = NAMES
    
    probs_display = [str(np.round(x*100, 1)) + "%" for x in probs]
    idx = np.argmax(probs)
    name = names[idx]
    
    idx_sort = np.argsort(probs)[::-1]
    probs_sorted = [probs_display[i] for i in idx_sort]
    names_sorted = [names[i] for i in idx_sort]
    df = pd.DataFrame(zip(names_sorted, probs_sorted), columns=["Nombre", "Probabilidad"])

    st.markdown("### Predicción")
    st.markdown(f"{name}")
    if evaluation:
        st.markdown("Correcto: ")

    st.markdown("### Probabilidades")
    st.table(df)

    st.markdown("### Imágenes")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Imagen original")
        st.image(image_resized)
    with col2:
        st.markdown("Mapa de interés")
        st.image(overlay_img)
