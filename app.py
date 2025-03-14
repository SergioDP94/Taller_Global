import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load('random_forest_viviendas.pkl')

# Configuración de la página
st.set_page_config(page_title="Predicción de Precio de Viviendas", page_icon="🏠", layout="centered")
st.image("imagen.png")
st.title("Predicción de Precio de Viviendas 🏡")
st.write("Ingrese las características de la vivienda para obtener una estimación del precio.")

# Entradas del usuario
tamaño = st.number_input("Tamaño (m²):", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
habitaciones = st.number_input("Habitaciones:", min_value=1, max_value=10, value=3, step=1)
antigüedad = st.number_input("Antigüedad (años):", min_value=0, max_value=100, value=10, step=1)
proximidad_centro = st.number_input("Proximidad al Centro (km):", min_value=0.1, max_value=50.0, value=5.0, step=0.1)

# Botón para predecir
if st.button("Predecir Precio"):
    features = np.array([[tamaño, habitaciones, antigüedad, proximidad_centro]])
    precio_predicho = modelo.predict(features)[0]
    st.success(f"Precio estimado: ${precio_predicho:,.2f}")


# Footer
st.markdown("---")
st.write("Desarrollado con Python y potenciado por Streamlit 🚀")
st.markdown("### 🎊 ᴛᴀʟʟᴇʀ ɢʀᴀᴛᴜɪᴛᴏ DE PYTHON")
st.markdown("👉 [𝗥𝗲𝘀𝗲𝗿𝘃𝗮 𝘁𝘂 𝗮𝘀𝗶𝘀𝘁𝗲𝗻𝗰𝗶𝗮](https://bit.ly/TALLMACHINEPYTHONSTREAMLIT)")