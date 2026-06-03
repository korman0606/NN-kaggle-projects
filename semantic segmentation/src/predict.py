import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from src.dataset import create_loaders
from src.model import UNetWithResNet50

def rle_multiclass_encode(mask):
    pixels = mask.flatten(order='F')
    if pixels.size == 0:
        return ""
    parts = []
    run_val = int(pixels[0])
    run_start = 1
    run_len = 1
    for idx in range(1, pixels.size):
        val = int(pixels[idx])
        if val == run_val:
            run_len += 1
        else:
            parts.append(f"{run_val} {run_start} {run_len}")
            run_val = val
            run_start = idx + 1
            run_len = 1
    parts.append(f"{run_val} {run_start} {run_len}")
    return " ".join(parts)

def predict_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _, _, test_dataset = create_loaders()
    model = UNetWithResNet50().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    if hasattr(test_dataset, 'data_path'):
        image_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_dataset.data_path]
    else:
        image_ids = [f"test_{i}" for i in range(len(test_dataset))]

    rows = []
    with torch.no_grad():
        for i, batch_images in enumerate(tqdm(test_loader, desc="Inference")):
            batch_images = batch_images.to(device)
            logits = model(batch_images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            for j in range(preds.shape[0]):
                rle = rle_multiclass_encode(preds[j])
                rows.append({'Id': image_ids[i * config.BATCH_SIZE + j], 'Predicted': rle})

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Сабмишн сохранён в {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    predict_submission()
