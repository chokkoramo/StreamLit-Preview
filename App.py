import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Predictor en vivo: Normal Equation vs Gradient Descent")
st.title("📈 Predictor en vivo: Ecuación Normal vs Gradiente Descendente")
st.caption("Regresión paramétrica vs. Machine Learning, aplicadas a tu propio dataset.")


# ---------- 1. Funciones de los dos métodos ----------

def regresion_normal_equation(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w[0], w[1:]


def regresion_gradient_descent(X, y, lr=0.1, n_iter=500):
    n = X.shape[0]
    X_b = np.c_[np.ones((n, 1)), X]
    w = np.zeros(X_b.shape[1])
    for _ in range(n_iter):
        error = X_b @ w - y
        grad = (2 / n) * (X_b.T @ error)
        w -= lr * grad
    return w[0], w[1:]


# ---------- 2. Obtener el dataset (generado o cargado) ----------

st.header("1. Dataset")
fuente = st.radio("¿De dónde sale la data?", ["Generar dataset sintético", "Cargar CSV"])

if fuente == "Generar dataset sintético":
    col1, col2, col3 = st.columns(3)
    n_samples = col1.slider("n° de muestras", 50, 2000, 300, step=50)
    n_features = col2.slider("n° de features", 1, 10, 3)
    noise = col3.slider("ruido", 0, 100, 15)

    if st.button("Generar dataset"):
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
        cols = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        st.session_state["df"] = df
        st.session_state.pop("modelo", None)  # invalida el modelo anterior: hay dataset nuevo
else:
    archivo = st.file_uploader("Sube tu CSV", type="csv")
    if archivo is not None:
        nuevo_df = pd.read_csv(archivo)
        if not nuevo_df.equals(st.session_state.get("df")):
            st.session_state["df"] = nuevo_df
            st.session_state.pop("modelo", None)  # invalida el modelo anterior: hay dataset nuevo

df = st.session_state.get("df")

if df is None:
    st.info("Genera un dataset o carga un CSV para continuar.")
    st.stop()

st.write(df.head())

# ---------- 3. Elegir target y features ----------

st.header("2. Selecciona columnas")
columnas = list(df.columns)
target_col = st.selectbox("Columna target (lo que quieres predecir)", columnas, index=len(columnas) - 1)
feature_cols = st.multiselect(
    "Columnas features (variables predictoras)",
    [c for c in columnas if c != target_col],
    default=[c for c in columnas if c != target_col],
)

if not feature_cols:
    st.warning("Selecciona al menos una feature.")
    st.stop()

# ---------- 4. Entrenar ----------

st.header("3. Entrenar modelos")
if "modelo" not in st.session_state:
    st.caption("⚠️ Todavía no has entrenado con este dataset. Presiona 'Entrenar'.")
if st.button("Entrenar"):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # La Ecuación Normal se entrena sin escalar
    b_ne, w_ne = regresion_normal_equation(X_train, y_train)
    pred_ne = b_ne + X_test @ w_ne

    # El Gradiente Descendente converge mejor con features escaladas
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    b_gd, w_gd = regresion_gradient_descent(X_train_sc, y_train, lr=0.1, n_iter=1000)
    pred_gd = b_gd + X_test_sc @ w_gd

    st.session_state["modelo"] = {
        "b_ne": b_ne, "w_ne": w_ne,
        "b_gd": b_gd, "w_gd": w_gd,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }

    st.subheader("Métricas en datos de prueba")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ecuación Normal**")
        st.metric("MSE", f"{mean_squared_error(y_test, pred_ne):.3f}")
        st.metric("R²", f"{r2_score(y_test, pred_ne):.3f}")
    with col2:
        st.markdown("**Gradiente Descendente**")
        st.metric("MSE", f"{mean_squared_error(y_test, pred_gd):.3f}")
        st.metric("R²", f"{r2_score(y_test, pred_gd):.3f}")

# ---------- 5. Predecir con valores nuevos ----------

if "modelo" in st.session_state:
    st.header("4. Probar una predicción nueva")
    modelo = st.session_state["modelo"]

    with st.form("form_prediccion"):
        entradas = {}
        for col in modelo["feature_cols"]:
            entradas[col] = st.number_input(col, value=float(df[col].mean()))
        enviado = st.form_submit_button("Predecir")

    if enviado:
        x_nuevo = np.array([[entradas[c] for c in modelo["feature_cols"]]])

        pred_ne = modelo["b_ne"] + x_nuevo @ modelo["w_ne"]
        x_nuevo_sc = modelo["scaler"].transform(x_nuevo)
        pred_gd = modelo["b_gd"] + x_nuevo_sc @ modelo["w_gd"]

        col1, col2 = st.columns(2)
        col1.metric("Predicción — Ecuación Normal", f"{pred_ne[0]:.3f}")
        col2.metric("Predicción — Gradiente Descendente", f"{pred_gd[0]:.3f}")