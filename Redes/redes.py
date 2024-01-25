"""
24/01/2024
Jeyfrey J. Calero R.
Aplicación de Redes Neuronales con scikit-learn streamlit, pandas, seaborn y matplolib 
"""


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


#Red Neuronal de prediction

st.header("Aplicación de Redes Neuronales") 
st.sidebar.header("Parámetros de usuarios")

def User_Input_Features():
    sepal_length = st.sidebar.slider("Longitud Sepalo", 4.3, 7.9, 5.84) 
    sepal_width = st.sidebar.slider("Ancho Sepalo", 2.0, 4.4, 3.5)
    petal_length = st.sidebar.slider("Longitud Pétalo", 1.0, 6.9, 3.7)
    petal_width = st.sidebar.slider("Ancho Pétalo", 0.1, 2.5, 1.2)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
#Se crea la tabla con la libreria PANDAS y el módulo DATAFRAME
    features = pd.DataFrame(data, index=[0])
    return features

df = User_Input_Features()

st.subheader("Datos de entrada")
st.write(df)

Iris = datasets.load_iris()
trainingx = Iris.data
trainingy = Iris.target

#Se crea una instancia
clf = RandomForestClassifier()
clf.fit(trainingx, trainingy)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader("Etiquetas de clase por indice")
st.write(Iris.target_names)
st.subheader("Predicción")
st.write(Iris.target_names[prediction])
st.subheader("Probabilidad de predicción")
st.write(prediction_proba)

#Creamos el gráfico de dispersión
Iris = datasets.load_iris()
Iris_df = pd.DataFrame(data = Iris.data, columns=Iris.feature_names)
Iris_df["target"] = Iris.target

st.subheader("Gráfico dispersión Iris")
dispersion_fig = sns.pairplot(Iris_df, hue = "target")
dispersion_fig.fig.savefig("Dispersión iris.png")
st.pyplot(dispersion_fig.fig)

#Creamos un Histograma
st.subheader("Histograma")
for feature in Iris.feature_names:
    plt.figure(figsize = (8, 5))
    sns.histplot(Iris_df[feature], kde = True)
    plt.title(f"Distribución de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frecuencia")
    plt.savefig(f"{feature}.png")
    st.image(f"{feature}.png")
    plt.close()
    
    