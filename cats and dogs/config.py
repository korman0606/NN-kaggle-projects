import os

# Пути к данным (на локальной машине)
# Предполагается, что данные скачаны с Kaggle и лежат в data
DATA_DIR = "data"
TRAIN_CATS_DIR = os.path.join(DATA_DIR, "/train/cats")
TRAIN_DOGS_DIR = os.path.join(DATA_DIR, "/train/dogs")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 8
IMG_SIZE = (224, 224)
LEARNING_RATE_NEW = 1e-3
LEARNING_RATE_OLD = 1e-5
WEIGHT_DECAY = 1e-5
DROPOUT = 0.5
TRAIN_SPLIT = 0.9

# Пути для сохранения
MODEL_WEIGHTS_PATH = "models/model_weights.pth"
LOSS_PLOT_PATH = "outputs/loss_plot.png"
SUBMISSION_PATH = "outputs/submission.csv"

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
