from flask import Flask, request, render_template, jsonify
import joblib
import re

# Загружаем обученную модель
model = joblib.load('procurement_classification_model.pkl')

# Создаем Flask приложение
app = Flask(__name__)

# Определяем функцию для проверки корректности ввода
def validate_input(description, price, duration):
    if not isinstance(description, str) or len(description.strip()) == 0:
        return False, "Некорректное описание. Оно должно быть непустой строкой."
    if not re.match(r'^[0-9]+(\.[0-9]+)?$', price):
        return False, "Некорректная цена. Она должна быть положительным числом."
    if not re.match(r'^[0-9]+$', duration):
        return False, "Некорректная длительность. Она должна быть положительным целым числом."
    return True, "Корректный ввод"

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    price = request.form['price']
    duration = request.form['duration']

    # Проверяем корректность ввода
    is_valid, message = validate_input(description, price, duration)
    if not is_valid:
        return render_template('index.html', prediction=message)

    # Подготавливаем данные для предсказания
    input_data = [description, float(price), int(duration)]

    # Предсказываем класс
    predicted_class = model.predict([input_data])[0]

    # Отображаем результат
    return render_template('index.html', prediction=f'Предсказанный класс закупки: {predicted_class}')

if __name__ == "__main__":
    app.run(debug=True)
