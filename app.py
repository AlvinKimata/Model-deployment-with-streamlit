import numpy as np
import pandas as pd
import streamlit as st
from prediction import predict


st.title('Classifying Iris flowers.')
st.markdown('Toy model to play to classify iris flowers into \
    setosa, versicolor, virginica')

#Include feature sliders for the 4 plant features.
st.header('Plant features')
col1, col2 = st.columns(2)

with col1:
    st.text('Sepal features')
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text('Sepal features')
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Peta; width (cm)', 0.1, 2.5, 0.5)


if st.button('Predict type of Iris flower.'):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    if result[0] == 0:
        st.text("The predicted flower is: Setosa.")

    elif result[0] == 1:
        st.text("The predicted flower is: Virginica.")

    else:
        st.text("The predicted flower is: Versicolor.")


