import torch
from tqdm import tqdm

from utils.metrics import compute_metrics


def train_model(loader, model, criterion, optimizer=None, device='cpu', train=True):
    epoch_loss = 0
    all_preds, all_labels = [], []

    if train:
        model.train()
    else:
        model.eval()
    
    iterator = tqdm(loader, desc="Train" if train else "Val")
    
    for imgs, labels in iterator:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if train:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs)
                loss = criterion(logits, labels)
        
        epoch_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss /= len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    if not train:
        print(f"\nValidation results:")
        print(f"Loss: {epoch_loss:.4f}")
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")

    return epoch_loss, metrics