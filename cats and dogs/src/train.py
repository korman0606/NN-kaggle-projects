import torch
import torch.nn as nn
from tqdm import tqdm
from config import *
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import plot_losses, calc_accuracy

def train():
    # Загрузка данных
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        img_size=IMG_SIZE,
        data_root=DATA_DIR
    )
    
    # Модель
    model = build_model(pretrained=True, dropout_rate=DROPOUT)
    device = torch.device(DEVICE)
    model = model.to(device)
    
    # Разделяем параметры для разных lr
    old_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'layer4' in name:
                old_params.append(param)
            else:
                new_params.append(param)
    
    optimizer = torch.optim.RMSprop([
        {'params': old_params, 'lr': LEARNING_RATE_OLD},
        {'params': new_params, 'lr': LEARNING_RATE_NEW, 'weight_decay': WEIGHT_DECAY}
    ])
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in train_tqdm:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            train_tqdm.set_postfix(loss=running_loss)
        train_losses.append(running_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={running_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Сохраняем веса
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"Model saved to {MODEL_WEIGHTS_PATH}")
    
    # График
    plot_losses(train_losses, val_losses, save_path=LOSS_PLOT_PATH)
    
    # Итоговая accuracy на валидации
    acc = calc_accuracy(model, val_loader, device)
    print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
