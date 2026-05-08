import pandas as pd
from datetime import datetime

# Función para extraer el año de fecha_alta
def extraer_anio(fecha_str):
    if pd.isna(fecha_str):
        return None
    try:
        # Intentar parsear como MM/DD/YYYY
        fecha = pd.to_datetime(fecha_str, format='%m/%d/%Y', errors='coerce')
        if pd.isna(fecha):
            # Intentar parsear como MMM-YY
            fecha = pd.to_datetime(fecha_str, format='%b-%y', errors='coerce')
        if pd.isna(fecha):
            # Otro formato, intentar general
            fecha = pd.to_datetime(fecha_str, errors='coerce')
        return fecha.year if not pd.isna(fecha) else None
    except:
        return None

# Leer el CSV
df = pd.read_csv('ejercicio4-inegi-MaximoRodriguez.csv')

# Convertir fecha_alta a años
df['fecha_alta'] = df['fecha_alta'].apply(extraer_anio)

# Redondear latitud a 2 decimales
df['latitud'] = df['latitud'].round(2)

# Guardar el CSV procesado
df.to_csv('ejercicio4-inegi-MaximoRodriguez_procesado.csv', index=False)

print("Procesamiento completado. Archivo guardado como 'ejercicio4-inegi-MaximoRodriguez_procesado.csv'")