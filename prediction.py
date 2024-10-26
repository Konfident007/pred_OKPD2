import joblib

# Загружаем обученную модель
model = joblib.load('procurement_classification_model.pkl')

# Функция для предсказания класса закупки
def predict_class(input_data):
    return model.predict([input_data])[0]
