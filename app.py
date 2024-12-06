import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image


# Función para descargar el modelo desde Dropbox
def download_model(url, destination):
    """Descarga el modelo desde Dropbox"""
    try:
        st.write("Intentando descargar el modelo desde Dropbox...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza error si la solicitud falla

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validar si el archivo existe después de la descarga
        if os.path.exists(destination):
            st.success(f"Modelo descargado correctamente en: {destination}")
        else:
            st.error("El archivo no se descargó correctamente.")
            raise FileNotFoundError("El archivo no se descargó correctamente.")
    except Exception as e:
        st.error(f"Error al descargar el modelo: {e}")
        raise e


# URL Dropbox ajustada para descarga directa
MODEL_URL = "https://www.dropboxusercontent.com/scl/fi/g1b2zjf0o8wlq9z4vgbbs/best_model.keras?dl=1"

# Crear ruta en el directorio de trabajo temporal en Streamlit Cloud
MODEL_DIR = os.path.join(os.getcwd(), "models")  # Crear una carpeta temporal para guardar el modelo
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")  # Ruta completa para el modelo

# Crear la carpeta si no existe
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Descargar el modelo si no existe en la ruta esperada
if not os.path.exists(MODEL_PATH):
    with st.spinner("Descargando modelo desde Dropbox..."):
        try:
            download_model(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error("No se pudo descargar el modelo. Detalles del error:")
            st.error(e)

# Intentar cargar el modelo
try:
    with st.spinner("Cargando el modelo..."):
        model = load_model(MODEL_PATH)
        st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    model = None  # Previene errores si el modelo no está disponible

# Diccionario de clases
class_dict = {0: 'Orgánico', 1: 'Reciclable'}

# Configurar la interfaz
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

    # Validar si el modelo está cargado
    if model:
        try:
            # Predicción
            prediction = model.predict(img_array)
            predicted_class = class_dict[int(prediction[0][0] > 0.5)]
            st.write(f"La imagen pertenece a la categoría: **{predicted_class}**")
        except Exception as e:
            st.error(f"Error al predecir la imagen: {e}")
    else:
        st.error("El modelo no se cargó correctamente.")
