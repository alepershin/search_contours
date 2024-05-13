import cv2
import numpy as np
import streamlit as st
import re

from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# Загружаем шрифт
font_path = 'FreeSans.ttf'
font = ImageFont.truetype(font_path, 18)

# Загрузка моделей
model_28_28 = load_model('model_28_28.h5')
model_56_28 = load_model('model_56_28.h5')
model_112_28 = load_model('model_112_28.h5')

# Константа символов для модели 28x28
s1 = '0123456789ABCDEFGH{xK<mN>PбRSTUVwxyzabdefghnqrt+'

symbols_28x28 = {i: char for i, char in enumerate(s1)} # Создаем словарь на основе s1
symbols_28x28.update({
    48: "Ответ", 49: "sqrt", 50: "(", 51: ")", 52: "[", 53: "все",
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

def classify_contour(contour, image, confidence_threshold):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    ratio = w / h

    # Определяем размеры изображения и выбираем модель
    if ratio < 2:
        frame_size = (28, 28)
        symbols = symbols_28x28
        model = model_28_28
    elif ratio < 4:
        frame_size = (56, 28)
        symbols = symbols_56x28
        model = model_56_28
    else:
        frame_size = (112, 28)
        symbols = symbols_112x28
        model = model_112_28

    # Масштабируем изображение, сохраняя пропорции
    scale = min(frame_size[0] / w, frame_size[1] / h)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_AREA)

    # Создаем новое изображение с "рамкой", заполненной нулями (черным цветом)
    new_image = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)

    # Вычисляем позицию для центрирования масштабированного изображения
    top = (frame_size[1] - new_size[1]) // 2
    left = (frame_size[0] - new_size[0]) // 2

    # Вставляем масштабированное изображение в центр "рамки"
    new_image[top:top + new_size[1], left:left + new_size[0]] = resized

    # Подготавливаем изображение для предсказания
    new_image = new_image / 255.0   # Нормализация
    new_image = new_image.reshape(1, frame_size[1], frame_size[0], 1)   # Добавляем размерность batch и каналов

    # Классифицируем изображение
    prediction = model.predict(new_image)
    class_id = np.argmax(prediction)

    # Получаем вероятность
    confidence = np.max(prediction)
    if confidence > confidence_threshold:
        # Получение соответствующего символа, если вероятность выше заданного порога
        ch = symbols.get(class_id, "") # Если class_id нет в словаре, вернется пустая строка
    else:
        ch = ""

    return ch

def predict_and_store_contours(image, contours, confidence_threshold):
    predictions = []  # Список для хранения информации о контурах и предсказаниях
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Получим координаты контура

        symbol = classify_contour(contour, image, confidence_threshold)
        if (contains_cyrillic(symbol) and symbol != "Ответ") or symbol == "" or symbol == "[":
            continue

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

def contains_cyrillic(text):
    for ch in [chr(i) for i in range(0x0400, 0x04FF)]:
        if ch in text:
            return True
    return False

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
    # Сортируем список сначала по горизонтали (x), затем по вертикали (y)
    contour_predictions.sort(key=lambda item: (item['x'], item['y']))

    i = 0  # Инициализируем счётчик
    while i < len(contour_predictions) - 1:
        current_item = contour_predictions[i]
        next_item = contour_predictions[i + 1]

        # Проверяем, являются ли символы минусами
        if current_item['symbol'] == next_item['symbol'] == "-":
            # Расчет расстояния между верхним краем первого минуса и нижним краем второго
            vertical_distance = abs(next_item['y'] - (current_item['y'] + current_item['h']))

            # Используем ширину самого большого минуса как пороговое значение для сравнения
            max_width = max(current_item['w'], next_item['w'])

            # Если вертикальное расстояние меньше ширины самого большого минуса,
            # считаем минусы расположенными достаточно близко, чтобы их можно было заменить на "="
            if vertical_distance < max_width:
                # Замена первого "-" на "="
                current_item['symbol'] = "="
                current_item['x'] = min(next_item['x'], current_item['x'])
                current_item['w'] = max_width
                current_item['h'] = next_item['y'] - current_item['y'] + next_item['h']

                # Удаляем второй "-" из списка
                del contour_predictions[i + 1]
                # Не увеличиваем счетчик, так как второй минус был удален
                continue
        i += 1  # Переходим к следующему элементу в списке

    return contour_predictions

def filter_contours_by_size(contours, min_size):
    filtered_contours = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)  # Получаем линейные размеры контура
        # Проверяем, соответствует ли контур заданным размерам
        if w >= min_size or h >= min_size:
            filtered_contours.append(contour)
    return filtered_contours

def group_by_lines(contour_predictions):
    # Сортировка контуров по координате y
    contour_predictions.sort(key=lambda item: (item['y'], item['x']))

    # Задаем начальные условия для первой строки
    lines = [[]]
    current_line_y = contour_predictions[0]['y'] + contour_predictions[0]['h']

    for item in contour_predictions:
        if item['y'] > current_line_y:
            # Начало новой строки
            lines.append([])
        # Добавление текущего символа в текущую строку
        lines[-1].append(item)
        if item['y'] + item['h'] > current_line_y:
            current_line_y = item['y'] + item['h']

    for i in range(len(lines) - 1):
        for cnt in lines[i]:
            if cnt['symbol'] == '-':
                ch = find_nearest_contour_above(cnt, lines[i])
                if ch != None and ch['symbol'] != '-' and find_nearest_contour_below(cnt, lines[i]) == None:
                    for j in lines[i + 1]:
                        lines[i].append(j)
                    lines[i + 1] = []
                    i += 1
                    break

    # Создаем новый список только с непустыми элементами
    lines = [line for line in lines if line != []]

    return lines

def find_nearest_contour_below(current_contour, contour_predictions):
    # Извлечение координат текущего контура
    current_x, current_y, current_w, current_h = current_contour['x'], current_contour['y'], current_contour['w'], current_contour['h']
    current_bottom = current_y + current_h
    nearest_contour = None
    min_distance = float('inf')  # Начальное значение минимального расстояния устанавливаем очень большим

    for contour in contour_predictions:
        # Извлечение координат потенциального контура, который находится ниже
        x, y, w, h = contour['x'], contour['y'], contour['w'], contour['h']
        # Проверяем, что потенциальный контур находится ниже текущего
        if y > current_bottom:
            # Находим центры контуров по оси X
            current_center_x = current_x + current_w / 2
            contour_center_x = x + w / 2

            # Если центры контуров по оси X близки, проверяем расстояние по вертикали
            if abs(current_center_x - contour_center_x) < max(current_w, w) / 2:
                distance = y - current_bottom  # Расстояние от нижней границы текущего контура до верхней границы потенциального
                # Если это расстояние меньше, чем текущее минимальное, обновляем ближайший контур
                if distance < min_distance:
                    min_distance = distance
                    nearest_contour = contour

    return nearest_contour

def find_nearest_contour_above(current_contour, contour_predictions):
    # Извлечение координат текущего контура
    current_x, current_y, current_w, _ = current_contour['x'], current_contour['y'], current_contour['w'], current_contour['h']
    nearest_contour = None
    min_distance = float('inf')  # начальное значение минимального расстояния

    for contour in contour_predictions:
        # Извлечение координат потенциального контура, который находится выше
        x, y, w, h = contour['x'], contour['y'], contour['w'], contour['h']
        contour_bottom = y + h
        # Проверка, что потенциальный контур находится выше текущего
        if contour_bottom < current_y:
            # Находим центры контуров по оси X
            current_center_x = current_x + current_w / 2
            contour_center_x = x + w / 2

            # Если центры контуров по оси X близки, проверяем расстояние по вертикали
            if abs(current_center_x - contour_center_x) < max(current_w, w) / 2:
                distance = current_y - contour_bottom  # Вертикальное расстояние от текущего до нижней границы потенциального
                # Если это расстояние меньше текущего минимального, обновляем ближайший контур
                if distance < min_distance:
                    min_distance = distance
                    nearest_contour = contour

    return nearest_contour

def find_nearest_contour_left(current_contour, contour_predictions):
    current_x, current_y, current_w, current_h = current_contour['x'], current_contour['y'], current_contour['w'], current_contour['h']
    current_left_edge = current_x

    nearest_contour = None
    min_distance = float('inf')

    for contour in contour_predictions:
        if contour is not current_contour:
            x, y, w, h = contour['x'], contour['y'], contour['w'], contour['h']
            contour_right_edge = x + w
            # Проверка, что потенциальный контур находится слева от текущего
            if contour_right_edge < current_left_edge:
                # Вычисляем горизонтальное расстояние от текущего контура до правого края потенциального
                distance = current_left_edge - contour_right_edge
                # Проверка вертикального перекрытия на границе
                vertical_overlap = current_y < (y + h) and (current_y + current_h) > y
                if distance < min_distance and vertical_overlap:
                    min_distance = distance
                    nearest_contour = contour

    return nearest_contour

def find_nearest_contour_right(current_contour, contour_predictions):
    current_x, current_y, current_w, current_h = current_contour['x'], current_contour['y'], current_contour['w'], current_contour['h']
    current_right_edge = current_x + current_w

    nearest_contour = None
    min_distance = float('inf')

    for contour in contour_predictions:
        if contour is not current_contour:
            x, y, w, h = contour['x'], contour['y'], contour['w'], contour['h']
            # Проверка, что потенциальный контур находится справа от текущего
            if x > current_right_edge:
                # Вычисляем горизонтальное расстояние от правого края текущего контура до потенциального
                distance = x - current_right_edge
                # Проверка вертикального перекрытия
                vertical_overlap = current_y < (y + h) and (current_y + current_h) > y
                if distance < min_distance and vertical_overlap:
                    min_distance = distance
                    nearest_contour = contour

    return nearest_contour

def block_recognition(line):
    s = ''
    # Сортировка контуров по координате x
    line.sort(key=lambda item: (item['x'], item['y']))
    i = 0
    while i < len(line):
        ch = line[i]['symbol']
        if ch == '-':
            numerator = ''
            y_numerator, h_numerator = 0, 0
            denominator = ''
            y_denominator, h_denominator = 0, 0
            j = i
            while i < len(line) - 1:
                if find_nearest_contour_below(line[i + 1], line) == line[j]:
                    if line[i + 1]['y'] + line[i + 1]['h'] < y_numerator + h_numerator / 2:
                        numerator += "**"
                    numerator += line[i + 1]['symbol']
                    y_numerator = line[i + 1]['y']
                    h_numerator = line[i + 1]['h']
                    i += 1
                elif find_nearest_contour_above(line[i + 1], line) == line[j]:
                    if line[i + 1]['y'] + line[i + 1]['h'] < y_denominator + h_denominator / 2:
                        denominator += "**"
                    denominator += line[i + 1]['symbol']
                    y_denominator = line[i + 1]['y']
                    h_denominator = line[i + 1]['h']
                    i += 1
                else:
                    break
            if numerator != '' and denominator != '':
                ch = '(' + numerator + ')/(' + denominator + ')'
            elif numerator == '-' or denominator == '-':
                ch = "="
        elif i > 0:
            if line[i]['y'] + line[i]['h'] < line[i - 1]['y'] + line[i - 1]['h'] / 2:
                ch = "**" + ch
        s += ch
        i += 1
    return s

st.title("Проверка письменных работ по математике")

uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    threshold = st.slider("Порог яркости", 0, 255, 128)
    min_size = st.slider("Минимальный размер контура", 0, 20, 7)
    show_results_on_image = st.checkbox("Показать результат распознавания на картинке", value=False)

    confidence_threshold = st.slider("Точность распознавания", 0.0, 1.0, 0.4)

    # Пользователь может задать ширину изображения
    target_width = st.slider("Ширина изображения", 400, image.shape[1], 1400)
    preprocessed_image = preprocess_image(image, threshold, target_width)

    contours = detect_contours(preprocessed_image)
    filtered_contours = filter_contours_by_size(contours, min_size)

    contour_predictions = predict_and_store_contours(preprocessed_image, filtered_contours, confidence_threshold)

    lines_of_contours = group_by_lines(contour_predictions)

    result_image = draw_rectangles(preprocessed_image, contour_predictions)

    for line in lines_of_contours:
        xmin = target_width
        ymin = result_image.shape[1]
        xmax = 0
        ymax = 0
        for cnt in line:
            if cnt['x'] < xmin:
                xmin = cnt['x']
            if cnt['y'] < ymin:
                ymin = cnt['y']
            if cnt['x'] + cnt['w'] > xmax:
                xmax = cnt['x'] + cnt['w']
            if cnt['y'] + cnt['h'] > ymax:
                ymax = cnt['y'] + cnt['h']

        s = block_recognition(line)
        if s.find('=') != -1:
            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(result_image, s, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Используем функцию для отрисовки результатов распознавания на изображении
    if show_results_on_image:
        result_image = draw_predictions_on_image(result_image, contour_predictions, 'FreeSans.ttf')

    st.subheader("Контуры")
    st.image(result_image)
