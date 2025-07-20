import torch
from tqdm import tqdm

from utils.metrics import compute_metrics

def test_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    test_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            test_loss += loss.item() * imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    test_metrics = compute_metrics(all_labels, all_preds)

    return test_loss, test_metrics
