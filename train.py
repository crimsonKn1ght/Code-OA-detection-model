import os, datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data.dataloader import get_loaders
from models.hybrid import DoubleUNetCBAM
from utils import compute_metrics

# ----- Hyperparameters -----
batch_size = 8
lr         = 1e-4
epochs     = 20
# ---------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader = get_loaders(
    './dataset/path_train', './dataset/path_val', batch_size=batch_size
)

model     = DoubleUNetCBAM(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
writer    = SummaryWriter(log_dir=f"runs/exp_{datetime.datetime.now():%Y%m%d_%H%M%S}")

for epoch in range(1, epochs+1):
    # Training
    model.train()
    train_loss = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward(); optimizer.step()

        train_loss += loss.item()*imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_metrics = compute_metrics(all_labels, all_preds)

    # Validation (similar block, model.eval())
    # ...

    # Log to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('F1/train',   train_metrics['f1'], epoch)
    # ...

writer.close()
