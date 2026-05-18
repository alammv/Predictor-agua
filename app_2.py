import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Predictor de Tarifas de Agua",
    page_icon="💧",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .resultado-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    .modelo-tag {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 Predictor de Tarifas de Agua")
st.markdown("Ingresa los datos para estimar la tarifa de agua (S/. por m³)")
st.markdown("📓 [Ver cuaderno de código en Google Colab](https://drive.google.com/file/d/1PPhlFjL6E-1IZGyb_xjrjM6fGqHApC5B/view?usp=sharing)", unsafe_allow_html=True)
st.markdown("👤 **Nombre:** Alan Raul Jimenez &nbsp;|&nbsp; **Código ISIL:** 00000")
st.divider()

@st.cache_resource
def cargar_modelos():
    rf = joblib.load("modelos/random_forest_model.pkl")
    lr = joblib.load("modelos/linear_regression_model.pkl")
    return rf, lr

try:
    rf_model, lr_model = cargar_modelos()
    modelos_ok = True
except Exception as e:
    st.error(f"No se pudieron cargar los modelos: {e}")
    modelos_ok = False

st.subheader("📋 Datos de entrada")

col1, col2 = st.columns(2)

with col1:
    periodo_anio_mes = st.number_input("Período (ej: 202401)", min_value=202401, max_value=202612, value=202401, step=1)
    cargo_fijo = st.number_input("Cargo fijo del prestador (S/.)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    aplica_igv = st.selectbox("¿Aplica IGV?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    tiene_factor = st.selectbox("¿Tiene factor de ajuste?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    factor_ajuste = st.slider("Valor del factor de ajuste", min_value=0.5, max_value=1.0, value=0.92, step=0.01)
    consumo_promedio = st.number_input("Consumo promedio localidad (m³)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)

with col2:
    clase = st.selectbox("Clase de cliente", options=[0, 1], format_func=lambda x: "Residencial" if x == 0 else "No residencial")
    categoria = st.selectbox("Categoría", options=[0, 1, 2, 3], format_func=lambda x: ["Doméstico", "Comercial", "Industrial", "Estatal"][x])
    volumen_asignado = st.number_input("Volumen asignado por categoría", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
    rango_ini = st.number_input("Rango inicial de consumo (m³)", min_value=0, max_value=1000, value=0, step=5)
    rango_fin = st.number_input("Rango final de consumo (m³)", min_value=1.0, max_value=1000.0, value=20.0, step=5.0)
    tiene_scf = st.selectbox("¿Tiene SCF?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

modelo_elegido = st.radio("Modelo a usar", options=["Random Forest", "Regresión Lineal"], horizontal=True)

if st.button("🔍 Predecir tarifa", use_container_width=True, type="primary"):
    if modelos_ok:
        entrada = np.array([[
            periodo_anio_mes,
            cargo_fijo,
            aplica_igv,
            tiene_factor,
            factor_ajuste,
            consumo_promedio,
            clase,
            categoria,
            volumen_asignado,
            rango_ini,
            rango_fin,
            tiene_scf
        ]])

        modelo = rf_model if modelo_elegido == "Random Forest" else lr_model
        prediccion = modelo.predict(entrada)[0]

        st.markdown(f"""
        <div class="resultado-box">
            Tarifa estimada: S/. {prediccion:.4f} por m³
            <div class="modelo-tag">Modelo usado: {modelo_elegido}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Los modelos no están disponibles.")
