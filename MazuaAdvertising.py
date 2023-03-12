import streamlit as st
import pandas as pd
import pickle

st.write("""
# MazuaAdvertising Prediction App

This app predicts the **Sales** by media platform!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 100.0, 300.0, 150.5)
    Radio = st.sidebar.slider('Radio', 50.0, 150.0, 70.5)
    Newspaper = st.sidebar.slider('Newspaper', 50.0, 150.0, 70.5)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("MazuaAdvertising.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction[0])
