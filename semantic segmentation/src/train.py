import torch
import torch.nn as nn
from tqdm import tqdm
import config
from src.dataset import create_loaders
from src.model import init_model, get_optimizer
from src.utils import compute_miou, plot_training_curves

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _, _, _ = create_loaders()
    model = init_model(device)
    optimizer = get_optimizer(model)

    class_weights = torch.tensor([1.0/f for f in config.CLASS_FREQ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    best_model_path = config.BEST_MODEL_PATH
    train_losses, val_losses, val_mious = [], [], []

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}', leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        miou = compute_miou(model, val_loader, device)
        val_mious.append(miou)

        print(f'Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {miou:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> Saved best model (val loss = {val_loss:.4f})')

    plot_training_curves(train_losses, val_losses, val_mious, save_path="outputs/training_curves.png")
    print("Обучение завершено. Лучшая модель сохранена.")

if __name__ == "__main__":
    train()
