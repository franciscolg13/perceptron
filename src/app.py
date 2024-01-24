import streamlit as st
from neuron import Neuron  # Import the Neuron class from the neuron module
import math

st.title('Aplicacion streamlit para un perceptrón')

st.image("https://images.theconversation.com/files/339172/original/file-20200602-133875-1u1teus.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=754&fit=clip", width=300)
st.markdown("### Simulador")


entradas = st.slider("Elige el numero de entradas", 1, 10)
st.write("Has elegido ", entradas, 'entradas')

st.markdown("### Pesos")
weights_row = st.columns(entradas)
st.markdown("### Entradas")
inputs_row = st.columns(entradas)

weights = []
inputs = []

for i in range(1, entradas + 1):
    # Display weights in the first row
    with weights_row[i - 1]:
        weight = st.number_input(f'Introduce un peso w{i}', step=0.01, key=f"weight_{i}")
        weights.append(weight)

for i in range(1, entradas + 1):

    # Display inputs in the second row
    with inputs_row[i - 1]:
        input_value = st.number_input(f'Introduzca el valor de la entrada x{i}:', key=f"input_{i}")
        inputs.append(input_value)

col1, col2 = st.columns(2)


with col1:
   st.markdown("### Sesgo")
   bias_input = st.number_input("Introduce el valor de Sesgo")
with col2:
   st.markdown("### Función de activación")
   funcion = st.selectbox(
    'Elige una función de activación',
    ('ReLU', 'Sigmoid', 'Tanh'))
   

if st.button("Calcular la salida", type="primary"):
    # Calcular la salida usando las entradas proporcionadas
    try:
        # Obtener el nombre de la función de activación correspondiente
        funciones_activacion = {
            'ReLU': '_relu',
            'Sigmoid': '_sigmoid',
            'Tanh': '_tanh'
        }
        nombre_funcion = funciones_activacion[funcion]
        neuron = Neuron(weights, bias=bias_input, func=nombre_funcion)
        output = neuron.predict(inputs)
        st.success(f"La salida del perceptrón es: {output}")
    except ValueError as e:
        st.error(str(e))