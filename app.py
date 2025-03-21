import streamlit as st
from PIL import Image 

st.set_page_config(page_title='Reconocimiento Facial', page_icon=':👤:', layout='wide')

#Sidebar to add the image
with st.sidebar:
    image = Image.open('ister.png')
    st.image(image, width=200)

st.header('Sistema de Asistencia usando Reconocimiento Facial')

#Loading the model and connecting to the DB
with st.spinner('Cargando Modelo y conectandose a la BD...'):
    import face_rec

st.success('Sistema Listo')
