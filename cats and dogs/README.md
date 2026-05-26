# Cats vs Dogs Classifier – 100% Accuracy on Kaggle

**Классификация изображений кошек и собак** с использованием Transfer Learning (ResNet50).  
Модель достигает 100% accuracy на тестовой выборке.

## 📊 Результаты
![Results](cats and dogs/outputs and images/results.png)

## 🚀 Как запустить

1. **Скачать данные** с Kaggle: [Dogs vs Cats]([https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/competitions/ninja-neural-nets-dogs-vs-cats))  
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
