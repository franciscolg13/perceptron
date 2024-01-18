import streamlit as st
import math

class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func

    def predict(self, input_data):
        # Misma cantidad de entradas como de pesos.
        if len(input_data) != len(self.weights):
            raise ValueError("Input data and weights must have the same length.")


        # Calcula la suma ponderada de los inputs
        weighted_sum = sum(w * x for w, x in zip(self.weights, input_data))

        # Añade Bias
        weighted_sum += self.bias

        # Aplica la funcion de activacion.
        if self.func == "ReLU":
            return max(0, weighted_sum)  # Funcion de activacion ReLu
        elif self.func == "Sigmoid": # Funcion sigmoide
            return 1 / (1 + math.exp(-weighted_sum))
        elif self.func == "Tanh": # Funcion tanh
            return math.tanh(weighted_sum)
        else:
            raise ValueError(f"Unsupported activation function: {self.func}")
        
    def changeBias(self, new_bias):
        self.bias = new_bias

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
        neuron = Neuron(weights, bias=bias_input, func=funcion)
        output = neuron.predict(inputs)
        st.success(f"La salida del perceptrón es: {output}")
    except ValueError as e:
        st.error(str(e))