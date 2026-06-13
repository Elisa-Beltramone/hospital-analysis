🏥 Hospital Control Room - Sistema de Predicción de Demanda

Este proyecto es una aplicación desarrollada en Streamlit para el análisis y la predicción de la demanda hospitalaria en tiempo real. Permite visualizar indicadores operativos, analizar datos históricos y generar predicciones de carga hospitalaria a 7 días.

🚀 Demo

Proximamente

📊 Funcionalidades principales
📌 Panel Operativo
Distribución de Obra Social
Análisis de género de pacientes
Ranking de diagnósticos más frecuentes
Heatmap de actividad por hora y día
Relación entre especialidad médica y diagnóstico
🤖 Módulo de Predicción
Predicción de demanda hospitalaria a 7 días (168 horas)
Selección de fecha y hora de inicio
Visualización de:
Forecast horario
Demanda diaria esperada
KPIs:
Próxima hora
Pico de demanda
Promedio de demanda
Sistema de alertas:
🟢 Normal
🟡 Demanda elevada
🔴 Riesgo de saturación
📉 Evaluación del modelo
Comparación entre valores reales y predichos
Métricas:
MAE (Error Absoluto Medio)
RMSE (Raíz del Error Cuadrático Medio)
Visualización de error y desviaciones
🧠 Modelo de Machine Learning

El sistema utiliza un modelo de XGBoost entrenado con variables temporales y exógenas:

Variables utilizadas:
Codificación temporal:
hora_sin, hora_cos
dow_sin, dow_cos
mes_sin, mes_cos
Lags históricos:
lag_1, lag_24, lag_168
Estadísticos móviles:
roll_24
Variables externas:
feriados
vísperas de feriado
temperatura
humedad
lluvia
📁 Estructura del proyecto
├── app.py                  # Aplicación Streamlit principal
├── xgb_demanda.pkl        # Modelo entrenado
├── df_model.csv           # Dataset procesado para modelado
├── df_unificado.csv       # Dataset crudo hospitalario
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación
⚙️ Instalación local
1. Clonar el repositorio
git clone https://github.com/tu_usuario/tu_repo.git
cd tu_repo
2. Crear entorno virtual
python -m venv venv
3. Activar entorno
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
4. Instalar dependencias
pip install -r requirements.txt
5. Ejecutar la app
streamlit run app.py
☁️ Despliegue

Este proyecto puede desplegarse fácilmente en:

Streamlit Cloud
Render
AWS / Azure / Google Cloud
📈 Resultados del modelo

El modelo obtiene actualmente:

MAE: ~1.02
RMSE: ~1.44

Nota: El modelo es adecuado para análisis de tendencia, pero aún puede mejorarse para predicción operativa de alta precisión.

🔧 Mejoras futuras
Incorporación de eventos epidemiológicos
Modelos por especialidad médica
Backtesting avanzado
Alertas en tiempo real
Sistema de autenticación para hospitales