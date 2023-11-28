import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Cargar el modelo guardado
model = tf.keras.models.load_model("titanic.h5")

# Crear un objeto StandardScaler para escalar los datos de entrada
scaler = StandardScaler()

# Crear la aplicación Streamlit
def main():    
    st.title("Predicción de supervivencia en el Titanic")        
    # Crear entradas para las características del pasajero    
    pclass = st.selectbox("Clase", options=[1, 2, 3])    
    sex = st.selectbox("Sexo", options=["Masculino", "Femenino"])    
    age = st.number_input("Edad", min_value=0, max_value=100, value=25)    
    fare = st.number_input("Tarifa", min_value=0.0, max_value=600.0, value=10.0)    
    embarked = st.selectbox("Puerto de embarque", options=["Cherbourg", "Queenstown", "Southampton"])      
    
    # Codificar variables categóricas   
    sex = 0 if sex == "Masculino" else 1    
    embarked = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]    

    # Crear una matriz NumPy con los datos de entrada    
    input_data = np.array([[pclass, sex, age, fare, embarked]])        

    # Escalar los datos de entrada    
    input_data = scaler.fit_transform(input_data)        

    # Realizar la predicción usando el modelo cargado    
    if st.button("Predecir supervivencia"):        
        prediction = model.predict(input_data)        
        probability = prediction[0][0]   

        st.write(f"Probabilidad de supervivencia: {probability:.2f}")

        if probability > 0.8:            
            st.write("El pasajero probablemente sobrevivirá.")        
        else:            
            st.write("El pasajero probablemente no sobrevivirá.")
if __name__ == "__main__":    
    main()