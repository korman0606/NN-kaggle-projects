# Cats vs Dogs Classifier – 100% Accuracy on Kaggle

**Классификация изображений кошек и собак** с использованием Transfer Learning (ResNet50).  
Модель достигает 100% accuracy на валидационной выборке (на данном датасете).

## 📊 Результаты
![Training loss](outputs/loss_plot.png)

## 🚀 Как запустить

1. **Скачать данные** с Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)  
   Распаковать в папку `data/` так, чтобы было:
data/
├── train/
│ ├── cats/
│ └── dogs/
└── test/

2. **Установить зависимости**:
pip install -r requirements.txt

3. **Обучить модель**:
python -m src.train

4. **Сделать предсказания для теста**:
python -m src.predict
