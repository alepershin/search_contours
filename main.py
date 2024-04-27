import cv2
import numpy as np
import streamlit as st

from keras.models import load_model

# Загрузка моделей
model_28_28 = load_model('model_28_28.h5')
model_56_28 = load_model('model_56_28.h5')
model_112_28 = load_model('model_112_28.h5')

# Константа символов для модели 28x28
s1 = '0123456789ABCDEFGH{xK<mN>PбRSTUVwxyzabdefghnqrt+'

symbols_28x28 = {i: char for i, char in enumerate(s1)} # Создаем словарь на основе s1
symbols_28x28.update({
    48: "Ответ", 49: "sqrt", 50: "(", 51: ")", 52: "[",
    # Если есть другие специфические символы за пределами s1, добавьте их здесь
})

# Словарь для модели 56x28
symbols_56x28 = {
    0: "-", 1: "sqrt", 2: "Ответ", 3: "нет", 4: "решения", 5: "Проверка", 6: "x",
    # Если есть дополнительные символы, добавьте их здесь
}

# Словарь для модели 112x28
symbols_112x28 = {
    0: "-", 1: "sqrt", 2: "Самостоятельная", 3: "получим", 4: "Умножим", 5: "уравнения", 6: "члены",
    # Если есть дополнительные символы, добавьте их здесь
}

models = {
    'model_28_28': (model_28_28, symbols_28x28),
    'model_56_28': (model_56_28, symbols_56x28),
    'model_112_28': (model_112_28, symbols_112x28),
}

def classify_contour(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    ratio = w / h

    # Определяем размеры изображения и выбираем модель
    if ratio < 2:
        resized_dim, model_info = (28, 28), models['model_28_28']
    elif ratio < 4:
        resized_dim, model_info = (56, 28), models['model_56_28']
    else:
        resized_dim, model_info = (112, 28), models['model_112_28']
    model, symbols = model_info

    # Масштабирование и нормализация изображения
    resized_image = cv2.resize(cropped_image, resized_dim, interpolation=cv2.INTER_AREA) / 255.0
    resized_image = resized_image.reshape(1, *resized_dim, 1)

    # Предсказание
    prediction = model.predict(resized_image)
    class_id = np.argmax(prediction)

    # Получение соответствующего символа
    ch = symbols.get(class_id, "") # Если class_id нет в словаре, вернется пустая строка

    return ch

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
            ch = classify_contour(contour, image)
            # Обеспечим, чтобы текст не выходил за верхнюю границу изображения
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            # Выводим текст красным цветом для лучшей видимости
            cv2.putText(result, ch, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
