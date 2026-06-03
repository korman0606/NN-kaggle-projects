import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import config
from src.dataset import create_loaders
from src.model import UNetWithResNet50
from src.utils import compute_miou  # если нужно

def denormalize(image_tensor):
    mean = config.NORMALIZE_MEAN
    std = config.NORMALIZE_STD
    image = image_tensor.clone().detach().cpu()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image

CLASS_COLORS = np.array([
    [128, 128, 128],   # фон
    [255, 0, 0],       # здания
    [0, 255, 0],       # деревья
    [0, 0, 255],       # вода
    [255, 255, 0]      # дороги
], dtype=np.uint8)

def mask_to_color(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(CLASS_COLORS):
        color_mask[mask == cls] = color
    return color_mask

def overlay_mask(image, mask, alpha=0.5):
    color_mask = mask_to_color(mask)
    overlay = (image * 255).astype(np.uint8)
    blended = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    return blended

def show_prediction(idx=0, save_path="outputs/sample_overlay.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _, _, _, _ = create_loaders()
    model = UNetWithResNet50().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    val_iter = iter(val_loader)
    images, true_masks = next(val_iter)
    image = images[idx]
    true_mask = true_masks[idx]
    with torch.no_grad():
        pred_mask = torch.argmax(model(image.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()

    image_np = denormalize(image)
    pred_color = mask_to_color(pred_mask)
    true_color = mask_to_color(true_mask.cpu().numpy())
    overlay = overlay_mask(image_np, pred_mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(image_np); axes[0,0].set_title('Original'); axes[0,0].axis('off')
    axes[0,1].imshow(pred_color); axes[0,1].set_title('Predicted Mask'); axes[0,1].axis('off')
    axes[1,0].imshow(true_color); axes[1,0].set_title('Ground Truth Mask'); axes[1,0].axis('off')
    axes[1,1].imshow(overlay); axes[1,1].set_title('Overlay'); axes[1,1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    show_prediction(idx=25)
