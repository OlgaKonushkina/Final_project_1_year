import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Загружаем модель (кэшируем) ===
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

xgb_model = load_model()

# === Заголовок ===
st.title("🚀 ML Прогнозирование")
st.write("Введите 20 признаков, и модель предскажет результат.")

# === Поля ввода (20 признаков) ===
# Ввод признаков
feature1 = st.number_input("Рейтинг школы 2:")
feature2 = st.number_input("Рейтинг школы 1:")
feature3 = st.number_input("Рейтинг школы 3:")
feature4 = st.number_input("Количество спален:")
feature5 = st.number_input("Односемейный, отдельный дом (0 - нет, 1 - да)):")
feature6 = st.number_input("Штат, расположен на юге (0 - нет, 1 - да):")
feature7 = st.number_input("Штат, расположен на востоке (0 - нет, 1 - да):")
feature8 = st.number_input("Наличие отопления (0 - нет, 1 - да)):")
feature9 = st.number_input("Наличие кондиционера (0 - нет, 1 - да)):")
feature10 = st.number_input("Квартира (0 - нет, 1 - да)):")
feature11 = st.number_input("Штат, расположен на западе (0 - нет, 1 - да):")
feature12 = st.number_input("2 этажа (0 - нет, 1 - да):")
feature13 = st.number_input("Наличие паркинга (0 - нет, 1 - да)):")
feature14 = st.number_input("1 этаж (0 - нет, 1 - да):")
feature15 = st.number_input("Земля (0 - нет, 1 - да)):")
feature16 = st.number_input("Наличие бассейна (0 - нет, 1 - да)):")
feature17 = st.number_input("Наличие камина (0 - нет, 1 - да)):")
feature18 = st.number_input("Лишен права выкупа (0 - нет, 1 - да)):")
feature19 = st.number_input("Штат, расположен на севере (0 - нет, 1 - да):")
feature20 = st.number_input("3 этажа (0 - нет, 1 - да):")


# Кнопка для отправки данных
if st.button("Предсказать"):    
    # Пример предсказания 
    predicted_price = xgb_model.predict([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20]])
    st.write(f"Предсказанная стоимость недвижимости: {predicted_price[0]}")
