import streamlit as st
import pandas as pd
from datetime import datetime
from app import face_rec

st.subheader('📑 - Reporte')

#Retrive logs and show in Report
#Extract data from Redis List
name = 'attendance:logs'

def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end) #Extract all data from the redis database
    decode_logs = [log.decode('utf-8') for log in logs_list]

    return decode_logs

def process_logs(logs):
    report = {}
    
    for log in logs:
        #Split the log by '@' to get name, profile, and timestamp
        name, _, timestamp = log.split('@')
        timestamp = datetime.strptime(timestamp.strip(), '%Y-%m-%d %H:%M:%S.%f')
        date_str = timestamp.date()
        
        #Create a composite key using name and date
        name_date_key = (name, date_str)
        
        #Initialize the report for name and date if it does not exist
        if name_date_key not in report:
            report[name_date_key] = {'entry': None, 'exit': None}
        
        #Assign the first occurrence as input and the last as output
        if report[name_date_key]['entry'] is None:
            report[name_date_key]['entry'] = timestamp
        else:
            report[name_date_key]['exit'] = timestamp
    
    #Convert the report to a DataFrame
    data = []
    for (name, date_str), times in report.items():
        entry_time = times['entry']
        exit_time = times['exit']
        worked_hours = (entry_time - exit_time).total_seconds() / 3600 if exit_time else 0
        data.append([name, date_str, entry_time, exit_time, worked_hours])
    
    df = pd.DataFrame(data, columns=['Nombre', 'Fecha', 'Marca1', 'Marca2', 'Horas Trabajadas'])
    return df


#Tabs to show the info
tab1, tab2 = st.tabs(['Personal Registrado', 'Logs'])

with tab1:
    if st.button('Actualizar Datos'):
        #Retrive the data from Redis Database
        with st.spinner('Retriving Data from redis DB...'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            #Rename the columns
            redis_face_db = redis_face_db.rename(columns={'Name': 'Nombre', 'Role': 'Cargo'})
            st.dataframe(redis_face_db[['Nombre', 'Cargo']])

with tab2:
    if st.button('Actualizar Logs'):
        datos = load_logs(name=name)
        df_reporte = process_logs(datos)
        #st.write(load_logs(name=name))
        st.write(df_reporte)

st.divider()
st.markdown('🔙 [Volver al Inicio](/app.py)')
