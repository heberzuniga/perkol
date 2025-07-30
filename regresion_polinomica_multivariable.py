
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Título de la app
st.title("Predicción Salarial con Regresión Polinómica Multivariable")
st.write("""
Esta aplicación utiliza un modelo de regresión polinómica multivariable para predecir el salario en función de varias características del empleado:
- Nivel
- Años de experiencia
- Nivel educativo
- Departamento
""")

# Cargar los datos
@st.cache_data
def load_data():
    return pd.read_csv("Position_Salaries.csv")

df = load_data()

# Mostrar una vista previa
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Codificar variables categóricas
df_encoded = pd.get_dummies(df, columns=["Educacion", "Departamento"], drop_first=True)

# Separar variables independientes (X) y dependiente (y)
X = df_encoded.drop(["Salario", "Position"], axis=1)  # Eliminamos 'Position' porque es redundante con 'Level'
y = df_encoded["Salario"]

# Selección del grado del polinomio
grado = int(st.slider("Selecciona el grado del polinomio", min_value=1, max_value=3, value=2))

# Aplicar transformación polinómica
poly = PolynomialFeatures(degree=grado, include_bias=False)
X_poly = poly.fit_transform(X)

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_poly, y)

# Predicción y evaluación
y_pred = modelo.predict(X_poly)
r2 = r2_score(y, y_pred)
st.write(f"Coeficiente de determinación R²: {r2:.3f}")
# Gráfico de comparación: Salario real vs. Salario predicho
st.subheader("Comparación: Salario Real vs. Predicho")
fig, ax = plt.subplots()
ax.scatter(y, y_pred, alpha=0.6, color="green", edgecolors="black")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("Salario real")
ax.set_ylabel("Salario predicho")
ax.set_title("Gráfico de dispersión")
st.pyplot(fig)


# Formulario para ingresar nuevos datos
st.subheader("Hacer una predicción personalizada")
with st.form("form_prediccion"):
    level_input = st.slider("Nivel (1-15)", min_value=1, max_value=15, value=5)
    exp_input = st.slider("Años de experiencia", min_value=0, max_value=40, value=5)
    edu_input = st.selectbox("Nivel educativo", ["Licenciatura", "Maestría", "Doctorado"])
    dept_input = st.selectbox("Departamento", ["Finanzas", "Marketing", "IT", "Operaciones", "Recursos Humanos"])
    submit = st.form_submit_button("Predecir salario")

if submit:
    # Crear dataframe con un solo registro para predecir
    entrada = {
        "Level": level_input,
        "Experiencia": exp_input,
        "Educacion_Maestría": 1 if edu_input == "Maestría" else 0,
        "Educacion_Doctorado": 1 if edu_input == "Doctorado" else 0,
        "Departamento_IT": 1 if dept_input == "IT" else 0,
        "Departamento_Marketing": 1 if dept_input == "Marketing" else 0,
        "Departamento_Operaciones": 1 if dept_input == "Operaciones" else 0,
        "Departamento_Recursos Humanos": 1 if dept_input == "Recursos Humanos" else 0
    }

    # Asegurar que todas las columnas estén presentes
    for col in X.columns:
        if col not in entrada:
            entrada[col] = 0  # Asignar 0 si no está presente

    df_entrada = pd.DataFrame([entrada])
    X_nuevo_poly = poly.transform(df_entrada)

    pred_salario = modelo.predict(X_nuevo_poly)[0]
    st.success(f"El salario estimado es: ${pred_salario:,.2f}")
