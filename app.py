import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Cargar el modelo guardado
model = load_model('C:/SEBASTIAN/BOOTCAMP IA/PROYECTO_FINAL_RED_NEURONAL_CNN/notebooks/best_model.keras')

# Diccionario de clases
class_dict = {0: 'Orgánico', 1: 'Reciclable'}

# Configuración de la interfaz
st.title("Clasificador de Imágenes: Residuos Orgánicos y Reciclables")
st.write("Sube una imagen para clasificarla.")

# Cargar y predecir la imagen
uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    st.write("Clasificando...")
    
    # Preprocesar la imagen
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    prediction = model.predict(img_array)
    predicted_class = class_dict[int(prediction[0][0] > 0.5)]
    
    st.write(f"La imagen pertenece a la categoría: **{predicted_class}**")

