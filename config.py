BATCH_SIZE = 32
EPOCHS = 8

LR_BACKBONE = 1e-5
LR_CLASSIFIER = 1e-3

IMAGE_SIZE = 224

TRAIN_DIR = "/kaggle/input/competitions/ninja-neural-nets-dogs-vs-cats/train"
TEST_DIR = "/kaggle/input/competitions/ninja-neural-nets-dogs-vs-cats/test"

DEVICE = "cuda"

CHECKPOINT_PATH = "checkpoints/best_model.pth"
