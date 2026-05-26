import os
import torch
import pandas as pd
from config import *
from src.dataset import get_dataloaders
from src.model import build_model

def predict():
    device = torch.device(DEVICE)
    # Загружаем модель
    model = build_model(pretrained=False, dropout_rate=DROPOUT)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Тестовый загрузчик
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        data_root=DATA_DIR
    )
    
    all_preds = []
    for x in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            y_pred = torch.argmax(pred, dim=1)
            all_preds.extend(y_pred.cpu().tolist())
    
    # Имена файлов теста
    test_filenames = sorted(os.listdir(TEST_DIR))
    submission = pd.DataFrame({
        'id': test_filenames,
        'label': all_preds
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    predict()
