import cv2
import numpy as np
import streamlit as st

from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# Загружаем шрифт
font_path = 'FreeSans.ttf'
font = ImageFont.truetype(font_path, 24)

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

def predict_and_store_contours(image, contours):
    predictions = []  # Список для хранения информации о контурах и предсказаниях
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Получим координаты контура
        area = cv2.contourArea(contour)

        if min_contour_area < area < max_contour_area:
            cropped_image = image[y:y + h, x:x + w]

            symbol = classify_contour(contour, image)

            # Сохраняем информацию в словарь
            contour_info = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "symbol": symbol
            }
            predictions.append(contour_info)

    return predictions


def preprocess_image(image, threshold, target_width):
    # Сначала определим коэффициент масштабирования
    # Это соотношение между желаемой шириной и текущей шириной изображения
    scale_ratio = target_width / image.shape[1]

    # Изменим размер изображения, чтобы ширина соответствовала target_width
    # Высота изменится пропорционально, чтобы сохранить соотношение сторон изображения
    new_size = (target_width, int(image.shape[0] * scale_ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Затем преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Применяем пороговое преобразование для получения бинарного изображения
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    return binary

def detect_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_predictions_on_image(image, predictions, font_path='FreeSans.ttf'):
    # Загрузка шрифта, который поддерживает отображение кириллицы
    font_size = 48  # Можно регулировать размер шрифта в зависимости от изображения
    font = ImageFont.truetype(font_path, font_size)

    # Конвертирование изображения OpenCV в формат PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Перебираем элементы массива predictions и отображаем текст
    for item in predictions:
        x, y, w, h = item['x'], item['y'], item['w'], item['h']
        symbol = item['symbol']

        # Добавьте смещение или подстройте координаты, где будет нарисован текст
        text_position = (x, y - font_size if y - font_size > font_size else y + h)

        # Вывод текста на изображение с использованием шрифта
        draw.text(text_position, symbol, font=font, fill=(255,255,0,0))

    # Конвертируем обратно в формат OpenCV и возвращаем изображение
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def draw_rectangles(image, predictions):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in predictions:
        x, y, w, h = item['x'], item['y'], item['w'], item['h']
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result

def replace_minus_with_equals(contour_predictions):
    # Создадим копию списка, чтобы не изменять исходный
    new_predictions = contour_predictions.copy()

    # Сортируем список по вертикальной (y) координате контуров
    new_predictions.sort(key=lambda item: (item['y'], item['x']))

    i = 0  # Инициализируем счётчик
    while i < len(new_predictions) - 1:
        current_item = new_predictions[i]
        next_item = new_predictions[i + 1]

        # Проверка: являются ли символы минусами
        if current_item['symbol'] == next_item['symbol'] == "-":
            # Проверка: находятся ли минусы на схожем горизонтальном уровне
            y_diff = abs(current_item['y'] - next_item['y'])
            if y_diff < some_threshold:  # some_threshold - пороговое значение, подлежащее настройке
                # Заменяем первый "-" на "="
                current_item['symbol'] = "="
                # Удаляем второй "-" из списка
                del new_predictions[i + 1]
                # Пропускаем дополнительную итерацию, так как второй минус удален
                continue
        i += 1  # Переходим к следующему элементу в списке

    return new_predictions

st.title("Проверка письменных работ по математике")

uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    threshold = st.slider("Порог яркости", 0, 255, 128)
    min_contour_area = st.slider("Минимальная площадь контура", 0, 200, 100)
    max_contour_area = st.slider("Максимальная площадь контура", 20000, 40000, 30000)
    show_results_on_image = st.checkbox("Показать результат распознавания на картинке")
    some_threshold = st.slider("Пороговое значение для определения знака равенства", 0, 20, 10)

    # Пользователь может задать ширину изображения
    target_width = st.slider("Ширина изображения", 400, image.shape[1], 1400)
    preprocessed_image = preprocess_image(image, threshold, target_width)

    contours = detect_contours(preprocessed_image)

    contour_predictions = predict_and_store_contours(preprocessed_image, contours)

    new_predictions = replace_minus_with_equals(contour_predictions)

    result_image = draw_rectangles(preprocessed_image, new_predictions)

    # Используем функцию для отрисовки результатов распознавания на изображении
    if show_results_on_image:
        result_image = draw_predictions_on_image(result_image, contour_predictions, 'FreeSans.ttf')

    st.subheader("Контуры")
    st.image(result_image)