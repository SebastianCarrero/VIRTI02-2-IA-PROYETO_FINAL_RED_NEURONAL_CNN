import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import gdown
import os

import zipfile

# Ruta donde se guardará el modelo descomprimido
MODEL_PATH = 'best_model.keras'
MODEL_ZIP_PATH = 'best_model.zip'

# Descargar el archivo si no existe
if not os.path.exists(MODEL_ZIP_PATH):
    file_id = '1GUmE7-jt437-oub8QhChCGM0G3exwtRD'  # ID del archivo de Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'  # Enlace de descarga directa
    gdown.download(url, MODEL_ZIP_PATH, quiet=False)

# Si el archivo está comprimido, descomprimirlo
if MODEL_ZIP_PATH.endswith('.zip'):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')  # Extrae el archivo en el directorio actual

# Verifica si el archivo existe después de la descompresión
if os.path.exists(MODEL_PATH):
    print(f"El archivo {MODEL_PATH} está disponible para cargar.")
else:
    print(f"Error: El archivo {MODEL_PATH} no se pudo descomprimir correctamente.")

# Cargar el modelo
model = load_model(MODEL_PATH)

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
    image = image.resize((224, 224))  # Ajusta al tamaño esperado por tu modelo
    img_array = img_to_array(image) / 255.0  # Normaliza la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Expande dimensiones para hacerla compatible con el modelo
    
    # Predicción
    prediction = model.predict(img_array)
    predicted_class = class_dict[int(prediction[0][0] > 0.5)]  # Predicción binaria

    st.write(f"La imagen pertenece a la categoría: **{predicted_class}**")

