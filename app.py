import streamlit as st
import joblib
import numpy as np

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Tarifas de Agua",
    page_icon="💧",
    layout="centered"
)

# ── Estilos ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .titulo { font-size: 2rem; font-weight: 700; color: #1a73e8; }
    .subtitulo { color: #555; margin-bottom: 1.5rem; }
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

# ── Encabezado ───────────────────────────────────────────────────────────────
st.markdown('<div class="titulo">💧 Predictor de Tarifas de Agua</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Ingresa los datos para estimar la tarifa de agua (S/. por m³)</div>', unsafe_allow_html=True)

# Enlace al notebook
st.markdown("📓 [Ver cuaderno de código en Google Colab](#)", unsafe_allow_html=True)
st.markdown("👤 **Nombre:** Tu Nombre Aquí &nbsp;|&nbsp; **Código ISIL:** 000000")
st.divider()

# ── Cargar modelos ───────────────────────────────────────────────────────────
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

# ── Formulario ───────────────────────────────────────────────────────────────
st.subheader("📋 Datos de entrada")

col1, col2 = st.columns(2)

with col1:
    cargo_fijo = st.number_input(
        "Cargo fijo del prestador (S/.)",
        min_value=0.0, max_value=10.0, value=3.5, step=0.1
    )
    consumo_promedio = st.number_input(
        "Consumo promedio de la localidad (m³)",
        min_value=1.0, max_value=30.0, value=12.0, step=0.5
    )
    rango_ini = st.number_input(
        "Rango inicial de consumo (m³)",
        min_value=0, max_value=1000, value=0, step=5
    )

with col2:
    rango_fin = st.number_input(
        "Rango final de consumo (m³)",
        min_value=1.0, max_value=1000.0, value=20.0, step=5.0
    )
    factor_ajuste = st.slider(
        "Factor de ajuste de localidad",
        min_value=0.5, max_value=1.0, value=0.92, step=0.01
    )
    aplica_igv = st.selectbox(
        "¿Aplica IGV?",
        options=[0, 1],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

clase = st.selectbox(
    "Clase de cliente",
    options=[0, 1],
    format_func=lambda x: "Residencial" if x == 0 else "No residencial"
)

modelo_elegido = st.radio(
    "Modelo a usar",
    options=["Random Forest", "Regresión Lineal"],
    horizontal=True
)

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("🔍 Predecir tarifa", use_container_width=True, type="primary"):
    if modelos_ok:
        entrada = np.array([[
            cargo_fijo,
            aplica_igv,
            factor_ajuste,
            consumo_promedio,
            clase,
            rango_ini,
            rango_fin
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
