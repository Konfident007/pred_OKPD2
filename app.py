from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

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
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        duration = request.form['duration']
        price = request.form['price']
        description = request.form['description']

        # Проверка и преобразование данных
        duration = float(duration)
        price = float(price)

        # Создание DataFrame с входными данными
        input_data = pd.DataFrame({
            'Длительность работ': [duration],
            'Цена контракта': [price],
            'Объект закупки': [description]
        })

        # Получение предсказания от модели
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Предсказанный код ОКПД-2: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Ошибка: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
