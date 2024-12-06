import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import os
import numpy as np
from PIL import Image


# Configurar la URL de Dropbox
url = "https://www.dropbox.com/scl/fi/g1b2zjf0o8wlq9z4vgbbs/best_model.keras?dl=1"
MODEL_PATH = "best_model.keras"


# Función para descargar el modelo desde Dropbox si aún no existe
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Descargando el modelo desde Dropbox...")
        response = requests.get(url)

        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as file:
                file.write(response.content)
            st.write("Modelo descargado correctamente.")
        else:
            st.error(
                f"Error al descargar el modelo desde Dropbox. Código de estado: {response.status_code}"
            )
            raise FileNotFoundError(
                f"Error al descargar el modelo desde Dropbox. Código de estado: {response.status_code}"
            )


# Descargar el modelo si no existe
download_model()

# Cargar el modelo
st.write("Cargando el modelo...")
model = load_model(MODEL_PATH)
st.write("Modelo cargado exitosamente.")


# Diccionario de clases
class_dict = {0: "Orgánico", 1: "Reciclable"}

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

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = class_dict[int(prediction[0][0] > 0.5)]

    # Mostrar los resultados
    st.write(f"La imagen pertenece a la categoría: **{predicted_class}**")
