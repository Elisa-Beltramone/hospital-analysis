# 🏥 Hospital Control Room

Aplicación de **Streamlit** para el análisis y predicción de la demanda hospitalaria en tiempo real.  
Permite visualizar indicadores operativos, analizar datos históricos y predecir la carga hospitalaria a 7 días.

---

## Demo

https://hospital-infraestructura.streamlit.app/

---

## Funcionalidades

### Panel Operativo
- Distribución de **Obra Social**
- Análisis de **género de pacientes**
- Diagnóstico más frecuentes
- Heatmap de actividad por **hora y día**
- Relación entre **especialidad médica y diagnóstico**

---

### Predicción de demanda
- Forecast de demanda hospitalaria a **7 días (168 horas)**
- Selección de fecha y hora de inicio
- Visualización de:
  - Predicción horaria
  - Demanda diaria agregada

### KPIs incluidos
- Próxima hora de demanda
- Pico de los próximos 7 días
- Promedio de demanda

### Sistema de alertas
- 🟢 Normal
- 🟡 Demanda elevada
- 🔴 Riesgo de saturación

---

### Evaluación del modelo
- Comparación entre predicción y valores reales
- Métricas:
  - MAE (Error Absoluto Medio)
  - RMSE (Raíz del Error Cuadrático Medio)
- Visualización de errores

---

## Modelo de Machine Learning

Modelo basado en **XGBoost**, entrenado con variables temporales y exógenas.

### Features utilizadas

**Variables temporales:**
- hora_sin, hora_cos  
- dow_sin, dow_cos  
- mes_sin, mes_cos  

**Lags históricos:**
- lag_1  
- lag_24  
- lag_168  

**Estadísticos móviles:**
- roll_24  

**Variables externas:**
- feriado  
- vispera_feriado  
- temperatura  
- humedad  
- lluvia  

---

## 📁 Estructura del proyecto
├── app.py # Aplicación Streamlit
├── xgb_demanda.pkl # Modelo entrenado
├── df_model.csv # Dataset procesado
├── df_unificado.csv # Dataset original
├── requirements.txt # Dependencias
└── README.md # Documentación

## 🔧 Mejoras futuras
Modelos por especialidad médica
Backtesting temporal avanzado
Alertas en tiempo real
Optimización de features externas
