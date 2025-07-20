import os
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from utils.dataloader import get_loaders
from utils.train import train_model
from utils.test import test_model
from models.hybrid import DoubleUNetCBAM

from config import (BATCH_SIZE, LR, EPOCHS, NUM_CLASSES, DATASET_PATH, 
                   TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LOG_DIR, CHECKPOINT_DIR)


import warnings
warnings.filterwarnings("ignore")


def split_dataset(dataset_path, train_ratio, val_ratio, test_ratio, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset_path: Path to the dataset directory
        train_ratio: Proportion for training (e.g., 0.7)
        val_ratio: Proportion for validation (e.g., 0.15)
        test_ratio: Proportion for testing (e.g., 0.15)
        random_state: Random seed for reproducible splits
        
    Returns:
        train_files, val_files, test_files: Lists of file paths
    """
    # Get all files from dataset directory (preserving class structure)
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                all_files.append(os.path.join(root, file))
    
    if len(all_files) == 0:
        raise ValueError(f"No image files found in {dataset_path}")
    
    print(f"📁 Total files found: {len(all_files)}")
    
    # First split: separate test set
    train_val_files, test_files = train_test_split(
        all_files, 
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=None  # You can add stratification if needed
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_ratio since we're working with remaining data after test split
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=adjusted_val_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=None
    )
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_files)} files ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"   Val:   {len(val_files)} files ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"   Test:  {len(test_files)} files ({len(test_files)/len(all_files)*100:.1f}%)")
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Split the dataset
    train_files, val_files, test_files = split_dataset(
        DATASET_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Get data loaders using file lists
    train_loader, val_loader = get_loaders(
        train_files=train_files, 
        val_files=val_files, 
        batch_size=BATCH_SIZE
    )

    # Get test loader separately
    test_loader = get_loaders(
        test_files=test_files, 
        batch_size=BATCH_SIZE
    )

    model = DoubleUNetCBAM(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    writer = SummaryWriter(
        log_dir=f"{LOG_DIR}/exp_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = train_model(
            train_loader, model, criterion, optimizer, device, train=True
        )

        val_loss, val_metrics = train_model(
            val_loader, model, criterion, optimizer, device, train=False
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1/train", train_metrics["f1"], epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print("✅ Best model saved.")

    writer.close()

    print("\n🧪 Running on test set...")
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))

    test_loss, test_metrics = test_model(
        model, test_loader, criterion, device
    )

    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"   {k}: {v:.4f}")