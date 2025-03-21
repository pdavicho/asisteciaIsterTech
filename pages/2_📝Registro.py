import streamlit as st
from app import face_rec
import cv2
import numpy as np

st.subheader('📝 - Registrar Usuario')

#Init Registration Form
registration_form = face_rec.RegistrationForm()

#Form inputs
person_name = st.text_input(label='Nombre', placeholder='Nombre y Apellido')
role = st.selectbox(label='Seleccionar Cargo', 
                   options=('Docente', 'Administrativo', 'Servicios'))

st.divider()
st.markdown('Iniciar cámara y registrar al menos 200 ejemplos')

#Create columns for buttons
col1, col2 = st.columns(2)

#Create buttons in columns
with col1:
    start_button = st.button("Iniciar Cámara")
with col2:
    stop_button = st.button("Detener")

if start_button:
    #Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #Create placeholder for video
    frame_placeholder = st.empty()

    while not stop_button:
        ret, frame = cam.read()
        if ret:
            #Get embeddings
            reg_img, embedding = registration_form.get_embeddings(frame)
            
            if embedding is not None:
                with open('face_embedding.txt', mode='ab') as f:
                    np.savetxt(f, embedding)

            #Convert BGR to RGB for Streamlit
            reg_img = cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB)
            #Display the frame
            frame_placeholder.image(reg_img)

    #Release camera when stopped
    cam.release()
    frame_placeholder.empty()

#Save data section
st.markdown('Presionar Guardar, para almacenar los datos.')

if st.button('Guardar'):
    return_val = registration_form.save_data_in_redis_db(person_name, role)
    if return_val == True:
        st.success(f'{person_name} registrado exitosamente')
    elif return_val == 'name_false':
        st.error('Ingrese el nombre: no dejar vacio o con espacios.')
    elif return_val == 'file_false':
        st.error('Por favor refresque la página y ejecute nuevamente.')

st.divider()
st.markdown('🔙 [Volver al Inicio](/app.py)')