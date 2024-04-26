import cv2
import numpy as np
import streamlit as st

def preprocess_image(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_rectangles(image, contours, min_contour_area, max_contour_area):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result

st.title("Обнаружение контуров рукописных символов")

uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    threshold = st.slider("Порог яркости", 0, 255, 128)
    min_contour_area = st.slider("Минимальная площадь контура", 0, 200, 100)
    max_contour_area = st.slider("Максимальная площадь контура", 20000, 40000, 30000)

    preprocessed_image = preprocess_image(image, threshold)

    contours = detect_contours(preprocessed_image)

    result_image = draw_rectangles(preprocessed_image, contours, min_contour_area, max_contour_area)

    st.subheader("Контуры")
    st.image(result_image)
