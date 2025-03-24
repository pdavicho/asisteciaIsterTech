# 📸 Sistema de Reconocimiento Facial

Sistema de reconocimiento facial basado en **InsightFace** y **Redis Cloud**. 

## 🚀 Características
- Carga el modelo **InsightFace** con `buffalo_sc` para la extracción de características faciales.
- Utiliza **Redis Cloud** como base de datos para almacenar los registros de asistencia.
- Interfaz intuitiva con tres módulos principales:
  1. **📷 Detección:** Busca el nombre de la persona registrada y compara su imagen en vivo con la base de datos.
     - ✅ **Cuadro Verde:** Persona verificada.
     - ❌ **Cuadro Rojo:** Persona desconocida.
  2. **📝 Registro:** Permite registrar nuevos usuarios ingresando **nombre** y **rol**. Se recomienda capturar alrededor de **200 imágenes** para mejorar la precisión.
  3. **📊 Reporte:** Muestra los usuarios registrados y las asistencias marcadas.

## 🛠 Instalación

Se recomienda usar un entorno virtual de Python (versiones **3.8 a 3.10**):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🎯 Uso

1. Ejecuta el sistema con:
   ```bash
   streamlit run app.py
   ```
2. Accede a la interfaz web y selecciona una de las opciones disponibles.
3. Registra nuevos usuarios y verifica su identidad en la sección de detección.

## 📌 Requisitos
Asegúrate de tener instalado:
- Python **3.8 - 3.10**
- Dependencias en `requirements.txt`
- Conexión a **Redis Cloud**


**🔗 Creado por PabloDavid**

