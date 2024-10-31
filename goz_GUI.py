# app.py
import tkinter as tk
import requests

def get_prediction():
    try:
        # Получение значений из полей ввода
        duration = float(entry_duration.get())
        contract_price = float(entry_price.get())

        # Получение текста из виджета Text
        purchase_object = text_object.get("1.0", tk.END).strip()

        # Подготовка данных для отправки
        data = {
            'duration': duration,
            'contract_price': contract_price,
            'purchase_object': purchase_object
        }

        # Отправка запроса к API
        response = requests.post('http://localhost:8000/predict', json=data)
        response.raise_for_status()  # Проверяем наличие ошибок HTTP
        prediction = response.json().get('prediction')

        # Отображение результата
        message_result.config(text=f'Предсказание: {prediction}', fg='black')
    except Exception as e:
        # Отображение ошибки без растягивания окна
        error_message = f'Ошибка: {e}'
        message_result.config(text=error_message, fg='red')

root = tk.Tk()
root.title('Предсказание модели')

# Создание фрейма для размещения элементов
frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

# Метка и поле ввода для 'Длительность работ'
tk.Label(frame, text='Длительность работ:').grid(row=0, column=0, sticky='e', pady=5)
entry_duration = tk.Entry(frame, width=30)
entry_duration.grid(row=0, column=1, pady=5)

# Метка и поле ввода для 'Цена контракта'
tk.Label(frame, text='Цена контракта:').grid(row=1, column=0, sticky='e', pady=5)
entry_price = tk.Entry(frame, width=30)
entry_price.grid(row=1, column=1, pady=5)

# Метка и виджет Text для 'Объект закупки'
tk.Label(frame, text='Объект закупки:').grid(row=2, column=0, sticky='ne', pady=5)
text_object = tk.Text(frame, width=50, height=5)
text_object.grid(row=2, column=1, pady=5)

# Добавление вертикального скроллбара к виджету Text
scrollbar = tk.Scrollbar(frame, orient='vertical', command=text_object.yview)
scrollbar.grid(row=2, column=2, sticky='ns', pady=5)
text_object.config(yscrollcommand=scrollbar.set)

# Кнопка для получения предсказания
tk.Button(frame, text='Получить предсказание', command=get_prediction).grid(row=3, column=0, columnspan=3, pady=10)

# Используем виджет Message для отображения результата или ошибки
message_result = tk.Message(frame, text='', font=('Arial', 12), width=400, justify='left')
message_result.grid(row=4, column=0, columnspan=3, pady=10)

root.mainloop()
