import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Hospital Control Room",
    layout="wide"
)

# =========================
# MODEL
# =========================
model = joblib.load("xgb_demanda.pkl")

# =========================
# DATA
# =========================
@st.cache_data
def load_model():
    df = pd.read_csv("df_model.csv", parse_dates=["fecha_hora"])
    return df

@st.cache_data
def load_raw():
    df = pd.read_csv("df_unificado.csv", parse_dates=["fecha_hora"])
    return df

df_model = load_model()
df_raw = load_raw()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Filtros")

especialidad = st.sidebar.selectbox(
    "Especialidad",
    ["Todas"] + sorted(df_raw["MEDICO"].dropna().unique())
)

# =========================
# FILTER RAW
# =========================
df_raw_filt = df_raw.copy()

if especialidad != "Todas":
    df_raw_filt = df_raw_filt[df_raw_filt["MEDICO"] == especialidad]

# asegurar features temporales
df_raw_filt["hora"] = df_raw_filt["fecha_hora"].dt.hour
df_raw_filt["dia"] = df_raw_filt["fecha_hora"].dt.dayofweek

# =========================
# ALERT MODEL
# =========================
df_model = df_model.sort_values("fecha_hora").copy()

df_model["rolling_mean"] = df_model["demanda"].rolling(24).mean()
df_model["rolling_std"] = df_model["demanda"].rolling(24).std()

df_model["zscore"] = (
    df_model["demanda"] - df_model["rolling_mean"]
) / df_model["rolling_std"]

df_model["alerta"] = df_model["zscore"] > 2

df_plot = df_model.copy()

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs([
    "📊 Operación",
    "🤖 Predicción",
])

# =========================
# TAB 1 - OPERACIÓN
# =========================
with tab1:

    # -------------------------
    # OBRA SOCIAL
    # -------------------------
    st.subheader("🏥 Obra Social")

    obra = df_raw_filt["OBRA SOCIAL"].value_counts(normalize=True).mul(100).reset_index()
    obra.columns = ["Obra Social", "Porcentaje"]

    fig = px.bar(
        obra.sort_values("Porcentaje"),
        x="Porcentaje",
        y="Obra Social",
        orientation="h",
        text="Porcentaje"
    )
    fig.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # GÉNERO
    # -------------------------
    st.subheader("👤 Género")

    gen = df_raw_filt["GENERO"].value_counts(normalize=True).mul(100).reset_index()
    gen.columns = ["Genero", "Porcentaje"]

    fig = px.pie(gen, names="Genero", values="Porcentaje", hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # DIAGNÓSTICOS
    # -------------------------
    st.subheader("🧠 Diagnósticos")

    diag = df_raw_filt["DIAGNOSTICO"].value_counts().reset_index()
    diag.columns = ["Diagnostico", "Cantidad"]

    fig = px.bar(diag, x="Cantidad", y="Diagnostico", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # HEATMAP HORA vs DÍA
    # -------------------------
    st.subheader("🔥 Heatmap Hora vs Día")

    map_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo"
    }

    df_raw_filt["dia_nombre"] = df_raw_filt["dia_semana"].map(map_dias)

    heat = (
        df_raw_filt
        .groupby(["hora", "dia_nombre"])
        .size()
        .reset_index(name="count")
    )

    heat_pivot = heat.pivot(
        index="hora",
        columns="dia_nombre",
        values="count"
    ).fillna(0)

    orden_dias = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

    heat_pivot = heat_pivot.reindex(columns=orden_dias)

    fig = px.imshow(
        heat_pivot,
        aspect="auto",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)


    # -------------------------
    # CROSS TAB
    # -------------------------
    st.subheader("📊 Especialidad vs Diagnóstico")

    ct = pd.crosstab(
        df_raw_filt["MEDICO"],
        df_raw_filt["DIAGNOSTICO"],
        normalize="index"
    )

    st.dataframe(ct, use_container_width=True)

# =========================
# FEATURE ENGINEERING FUTURE
# =========================
# =========================
# FEATURES (orden EXACTO del modelo)
# =========================
features = [
    "hora_sin","hora_cos",
    "dow_sin","dow_cos",
    "mes_sin","mes_cos",
    "lag_1","lag_24","lag_168",
    "roll_24",
    "feriado","vispera_feriado",
    "temperatura","humedad","lluvia"
]

# =========================
# FUTURE FEATURE BUILDER (PRODUCCIÓN)
# =========================
def build_future_features(df_hist, steps=24):

    df_hist = df_hist.sort_values("fecha_hora").copy()

    last_time = df_hist["fecha_hora"].iloc[-1]

    # últimos valores reales (NO se modifican)
    last_vals = df_hist.iloc[-1]
    last_24 = df_hist["demanda"].iloc[-24:] if len(df_hist) >= 24 else df_hist["demanda"]
    last_168 = df_hist["demanda"].iloc[-168:] if len(df_hist) >= 168 else df_hist["demanda"]

    future_rows = []

    for i in range(1, steps + 1):

        t = last_time + pd.Timedelta(hours=i)

        hour = t.hour
        dow = t.dayofweek
        month = t.month

        row = {
            # temporal encoding
            "hora_sin": np.sin(2*np.pi*hour/24),
            "hora_cos": np.cos(2*np.pi*hour/24),
            "dow_sin": np.sin(2*np.pi*dow/7),
            "dow_cos": np.cos(2*np.pi*dow/7),
            "mes_sin": np.sin(2*np.pi*month/12),
            "mes_cos": np.cos(2*np.pi*month/12),

            # lags SOLO desde histórico real
            "lag_1": df_hist["demanda"].iloc[-1],
            "lag_24": df_hist["demanda"].iloc[-24] if len(df_hist) >= 24 else df_hist["demanda"].mean(),
            "lag_168": df_hist["demanda"].iloc[-168] if len(df_hist) >= 168 else df_hist["demanda"].mean(),

            # rolling SOLO histórico
            "roll_24": last_24.mean(),

            # exógenas (último valor conocido = baseline hospitalario)
            "feriado": last_vals.get("feriado", 0),
            "vispera_feriado": last_vals.get("vispera_feriado", 0),
            "temperatura": last_vals.get("temperatura", df_hist["temperatura"].mean() if "temperatura" in df_hist else 0),
            "humedad": last_vals.get("humedad", df_hist["humedad"].mean() if "humedad" in df_hist else 0),
            "lluvia": last_vals.get("lluvia", 0),
        }

        future_rows.append(row)

    return pd.DataFrame(future_rows)
# =========================
# TAB 2 - PREDICCIÓN
# =========================
with tab2:

    st.subheader("🤖 Predicción hospitalaria (24h)")

    X_future = build_future_features(df_model, steps=24)

    # seguridad: orden de features EXACTO
    X_future = X_future[features]

    preds = model.predict(X_future)

    future_dates = pd.date_range(
        start=df_model["fecha_hora"].iloc[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="H"
    )

    pred_df = pd.DataFrame({
        "fecha_hora": future_dates,
        "prediccion": preds
    })

    fig = px.line(
        pred_df,
        x="fecha_hora",
        y="prediccion",
        title="Forecast demanda hospitalaria"
    )

    st.plotly_chart(fig, use_container_width=True)

    # KPIs clínicos
    col1, col2, col3 = st.columns(3)

    col1.metric("Próxima hora", round(preds[0], 1))
    col2.metric("Pico 24h", round(preds.max(), 1))
    col3.metric("Promedio 24h", round(preds.mean(), 1))

    # alertas operativas
    umbral = df_model["demanda"].mean() + 2 * df_model["demanda"].std()

    if preds[0] > umbral:
        st.error("🔴 Riesgo de saturación en la próxima hora")
    elif preds[0] > df_model["demanda"].mean() + df_model["demanda"].std():
        st.warning("🟡 Demanda elevada esperada")
    else:
        st.success("🟢 Capacidad normal")