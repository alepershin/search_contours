import cv2
import numpy as np
import streamlit as st

from keras.models import load_model

# Загрузка моделей
model_28_28 = load_model('model_28_28.h5')
model_56_28 = load_model('model_56_28.h5')
model_112_28 = load_model('model_112_28.h5')

def classify_contour(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    ratio = w / h

    # Масштабирование изображения до размеров, подходящих для модели
    if ratio < 2:
        resized_image = cv2.resize(cropped_image, (28, 28), interpolation=cv2.INTER_AREA)
        model = model_28_28
    elif ratio < 4:
        resized_image = cv2.resize(cropped_image, (56, 28), interpolation=cv2.INTER_AREA)
        model = model_56_28
    else:
        resized_image = cv2.resize(cropped_image, (112, 28), interpolation=cv2.INTER_AREA)
        model = model_112_28

    # Нормализация изображения
    resized_image = resized_image / 255.0
    resized_image = resized_image.reshape(1, resized_image.shape[0], resized_image.shape[1], 1)

    # Предсказание
    prediction = model.predict(resized_image)
    # Верните идентификатор класса, категорию или символ в зависимости от того, как обучена модель
    # Тем временем предположим, что она возвращает идентификатор класса
    class_id = np.argmax(prediction)
    return class_id

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
            class_id = classify_contour(contour, image)
            # Обеспечим, чтобы текст не выходил за верхнюю границу изображения
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            # Выводим текст красным цветом для лучшей видимости
            cv2.putText(result, str(class_id), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
