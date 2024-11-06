from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
import re

app = Flask(__name__)

# Инициализация компонентов Natasha для лемматизации
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Функция для лемматизации текста
def lemmatize_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemmatized_text = ' '.join([token.lemma for token in doc.tokens])
    return lemmatized_text

# Кастомный трансформер для лемматизации
class LemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Применяем функцию лемматизации к каждому элементу
        return X.apply(lemmatize_text)

# Загрузка модели с учетом кастомного трансформера
model = joblib.load('model/best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        duration_days = request.form['duration']
        price = request.form['price']
        description = request.form['description']

        error_message = None  # Переменная для хранения сообщений об ошибках

        # Проверка наличия данных
        if not duration_days or not price or not description:
            error_message = 'Пожалуйста, заполните все поля.'
            return render_template('index.html', error_message=error_message)

        # Замена запятых на точки в числовых полях
        duration_days = duration_days.replace(',', '.')
        price = price.replace(',', '.')

        # Проверка корректности и преобразование данных
        try:
            duration_days = float(duration_days)
            if duration_days <= 0:
                error_message = 'Длительность работ должна быть положительным числом больше нуля.'
                return render_template('index.html', error_message=error_message)
        except ValueError:
            error_message = 'Длительность работ должна быть числом.'
            return render_template('index.html', error_message=error_message)

        try:
            price = float(price)
            if price <= 0:
                error_message = 'Цена контракта должна быть положительным числом больше нуля.'
                return render_template('index.html', error_message=error_message)
        except ValueError:
            error_message = 'Цена контракта должна быть числом.'
            return render_template('index.html', error_message=error_message)

        # Проверка длины описания
        if len(description.strip()) == 0:
            error_message = 'Описание объекта закупки не должно быть пустым.'
            return render_template('index.html', error_message=error_message)
        elif len(description) > 1000:
            error_message = 'Описание объекта закупки слишком длинное (максимум 1000 символов).'
            return render_template('index.html', error_message=error_message)

        # Конвертация дней в секунды
        duration_seconds = duration_days * 24 * 60 * 60  # Преобразование дней в секунды

        # Создание DataFrame с входными данными
        input_data = pd.DataFrame({
            'Длительность работ': [duration_seconds],
            'Цена контракта': [price],
            'Объект закупки': [description]
        })

        # Получение предсказания от модели
        prediction = model.predict(input_data)
        predicted_class = prediction[0]

        # Определение группы по предсказанному классу
        if ((predicted_class.startswith('41') and not predicted_class.startswith('41.1')) or
            predicted_class.startswith('42') or
            predicted_class.startswith('43')):
            group_message = "Строительно-монтажные работы"
        elif predicted_class == '41.1' or predicted_class == '71.1':
            group_message = "Проектно-изыскательские работы"
        elif predicted_class == '43.2':
            group_message = "Подключение коммуникаций"
        else:
            group_message = "Прочее"

        # Передача предсказания и группы в шаблон
        return render_template('index.html',
                               prediction_text=f'Предсказанный код ОКПД-2: {predicted_class}',
                               group_message=group_message)
    except Exception as e:
        return render_template('index.html', error_message=f'Ошибка: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
