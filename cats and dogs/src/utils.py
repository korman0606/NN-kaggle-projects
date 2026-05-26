import torch
import matplotlib.pyplot as plt

def calc_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)           # one-hot [1,0] или [0,1]
            y_true = torch.argmax(y, dim=1)
            pred = model(x)
            y_pred = torch.argmax(pred, dim=1)
            correct += (y_pred == y_true).sum().item()
            total += y.size(0)
    return correct / total

def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
