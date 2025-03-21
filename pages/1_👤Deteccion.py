import streamlit as st
from app import face_rec
from face_rec import faceapp
import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import time

st.subheader('👤 - Detección')

#Retrieve the data from Redis Database
with st.spinner('Esperando BD...'):
    df = face_rec.retrive_data('academy:register')
    if df is None or df.empty:
        st.error("No se pudieron cargar los datos de la base de datos")
        st.stop()

#Display available data
#st.write("### Personas Registradas")
#st.dataframe(df[['Name', 'Role']])

#Get unique names for the selectbox
unique_names = df['Name'].unique().tolist()

#Create selectbox for selecting a person
selected_name = st.selectbox("Nombre:", options=unique_names, index=None)

if selected_name:
    #Display selected person details
    st.subheader(f"Persona Seleccionada: {selected_name}")
    person_data = df[df['Name'] == selected_name].iloc[0]
    st.write(f"Cargo: {person_data['Role']}")
  
#Button to start the camera
start_camera = st.button("Iniciar Cámara", disabled=selected_name is None)

if start_camera and selected_name:
    #Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #Create placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    #Initialize time variables
    waitTime = 3  # Time in sec for data save
    autoStopTime = 5  # Time in sec for auto stop after match
    startTime = time.time()  # Initial time for auto stop
    setTime = time.time()    # Initial time for data save
    savedTime = 0           # Time of last save
    matchTime = None       # Time when match was found
    realtimepred = face_rec.RealTimePred()

    #Button to stop the camera
    stop_button = st.button("Detener")

    while not stop_button:
        ret, frame = cam.read()
        if ret:
            current_time = time.time()
            elapsed_time = current_time - startTime
            difftime = current_time - setTime
            
            # Process frame for face recognition
            results = faceapp.get(frame)
            display_frame = frame.copy()
            
            found_match = False
            
            # Process each detected face
            for res in results:
                x1, y1, x2, y2 = res['bbox'].astype(int)
                embeddings = res['embedding']
                
                #Compare with selected person only
                person_name, person_role = face_rec.ml_search_algorithm(
                    pd.DataFrame([person_data]),
                    'facial_features',
                    test_vector=embeddings,
                    name_role=['Name', 'Role']
                )
                
                #Set color and text based on match
                if person_name == selected_name:
                    color = (0, 255, 0)  #Green
                    display_name = selected_name
                    found_match = True
                    # Set match time if not already set
                    if matchTime is None:
                        matchTime = current_time
                else:
                    color = (0, 0, 255)  #Red
                    display_name = "Desconocido"
                
                #Draw rectangle and text
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(display_frame, display_name, (x1, y1-10),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                cv2.putText(display_frame, str(datetime.now()), (x1, y2+20),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # Check if we should stop after successful match
            if matchTime is not None:
                time_since_match = current_time - matchTime
                remaining_time = max(0, autoStopTime - int(time_since_match))
                
                # Add countdown to frame
                cv2.putText(display_frame,
                           f"Cerrando en: {remaining_time}s",
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (0, 255, 0),
                           2)
                
                # Stop if time has elapsed
                if time_since_match >= autoStopTime:
                    st.success(f'Verificación completada para: {selected_name}')
                    break
            
            # Check if it's time to save data and there's a match
            if difftime >= waitTime and found_match:
                # Save to Redis with person details
                save_success = realtimepred.saveLogs_redis(
                    person_name=selected_name,
                    role=person_data['Role']
                )
                
                if save_success:
                    setTime = time.time()
                    savedTime = time.time()
                    st.success('Datos registrados en la BD correctamente.')
                else:
                    st.error('Error al guardar los datos en la BD')

            # Show saving message for 3 seconds only when there's a match and save was successful
            if time.time() - savedTime <= 3 and found_match:
                cv2.putText(display_frame,
                           "Datos guardados!",
                           (int(display_frame.shape[1]/4), int(display_frame.shape[0]/2)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 255, 0),
                           2)

            # Time counter to frame (only show countdown when there's a match)
            time_text = f"Tiempo activo: {int(elapsed_time)}s"
            
            cv2.putText(display_frame,
                       time_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)

            # Display the frame
            frame_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True
            )
            
            #Update status
            if found_match:
                status_placeholder.success(f"✅ Verificación exitosa: {selected_name}")
            else:
                status_placeholder.warning("❌ Persona no reconocida")

    #Camera cleanup
    cam.release()
    frame_placeholder.empty()
    if matchTime is not None:
        status_placeholder.success("✅ Verificación completada y guardada")
    else:
        status_placeholder.info("Cámara detenida")

st.markdown("---")
st.write("Nota: El sistema compara las características faciales de la persona seleccionada con la imagen en vivo.")
st.write("Cuadro Verde = Persona verificada | Cuadro Rojo = Persona desconocida")
