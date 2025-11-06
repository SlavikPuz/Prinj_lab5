from core import survival_percent, young_old_survival_by_class
import streamlit as st
import pandas as pd
from PIL import Image

st.title("Анализ выживших пассажиров Титаника по возрастным группам")

image = Image.open("titanic.jpeg")
st.image(image, use_container_width=True)

@st.cache_data
def load_data():
    df = pd.read_csv("titanic_train.csv")
    return df

data = load_data()

st.success(f"Всего записей: {len(data)}")

classes = sorted(data["Pclass"].dropna().unique())
selected_class = st.selectbox("Выберите класс билета:", classes)

young_percent, old_percent = young_old_survival_by_class(data, selected_class)

st.subheader("Результаты анализа")
col1, col2 = st.columns(2)
with col1:
    st.metric("Процент выживших молодых (<30)", f"{young_percent:.2f}%")
with col2:
    st.metric("Процент выживших старых (>60)", f"{old_percent:.2f}%")

st.subheader("Отфильтрованные данные по выбранному классу")
st.dataframe(data.loc[data["Pclass"] == selected_class, ["Name", "Age", "Pclass", "Survived"]].dropna())
