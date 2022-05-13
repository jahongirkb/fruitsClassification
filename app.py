import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title("Mevalarni klassifikatsiya qiluvchi model")

file = st.file_uploader("Rasmni yuklash", type=['jpg', 'png', 'gif'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('fruits_model.pkl')

    pred, pred_id, probs = model.predict(img)
    prob = probs[pred_id]*100
    if prob>98:
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.info("Bunday rasm modelda mavjud emas!")
