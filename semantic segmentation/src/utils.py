import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_miou(model, dataloader, device, num_classes=5):
    model.eval()
    iou_per_class = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                true_mask = (masks == cls)
                intersection = (pred_mask & true_mask).sum().float()
                union = (pred_mask | true_mask).sum().float()
                if union == 0:
                    iou = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
                else:
                    iou = intersection / union
                iou_per_class.append(iou.item())
    iou_per_class = np.array(iou_per_class).reshape(-1, num_classes)
    miou = iou_per_class.mean(axis=0).mean()
    return miou

def plot_training_curves(train_losses, val_losses, val_mious, save_path="outputs/training_curves.png"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_mious)+1), val_mious, label='Val mIoU', marker='^', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
