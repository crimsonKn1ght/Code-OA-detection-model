import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


class FileListDataset(Dataset):
    """Custom dataset that loads images from a list of file paths."""
    
    def __init__(self, file_list, transform=None):
        """
        Args:
            file_list: List of file paths to images
            transform: Optional transform to be applied on images
        """
        self.file_list = file_list
        self.transform = transform
        
        # Extract class names from directory structure
        # Assumes structure like: /path/to/dataset/class_name/image.jpg
        self.class_to_idx = {}
        self.classes = []
        
        # Get unique class names from file paths
        class_names = set()
        for file_path in file_list:
            # Get parent directory name as class
            class_name = os.path.basename(os.path.dirname(file_path))
            class_names.add(class_name)
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get class label from directory name
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_loaders(train_files=None, val_files=None, test_files=None, 
                train_dir=None, val_dir=None, test_dir=None, 
                batch_size=8, num_workers=4):
    """
    Create data loaders from either file lists or directory paths.
    
    Args:
        train_files, val_files, test_files: Lists of file paths (new approach)
        train_dir, val_dir, test_dir: Directory paths (legacy approach)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader(s) depending on what's provided
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    loaders = []
    
    # NEW APPROACH: Handle file lists (for split datasets)
    if train_files is not None or val_files is not None or test_files is not None:
        
        # Handle train loader
        if train_files is not None:
            train_ds = FileListDataset(train_files, transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
            loaders.append(train_loader)
        
        # Handle val loader  
        if val_files is not None:
            val_ds = FileListDataset(val_files, transform)
            val_loader = DataLoader(val_ds, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
            loaders.append(val_loader)
        
        # Handle test loader (for single test loader case)
        if test_files is not None and train_files is None and val_files is None:
            test_ds = FileListDataset(test_files, transform)
            test_loader = DataLoader(test_ds, batch_size=batch_size, 
                                   shuffle=False, num_workers=num_workers)
            return test_loader  # Return single test loader
    
    # LEGACY APPROACH: Handle directory paths (backward compatibility)
    else:       
        # Handle train loader
        if train_dir is not None:
            train_ds = datasets.ImageFolder(train_dir, transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)
            loaders.append(train_loader)
        
        # Handle val loader
        if val_dir is not None:
            val_ds = datasets.ImageFolder(val_dir, transform)
            val_loader = DataLoader(val_ds, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
            loaders.append(val_loader)
        
        # Handle test loader (for single test loader case)
        if test_dir is not None:
            test_ds = datasets.ImageFolder(test_dir, transform)
            test_loader = DataLoader(test_ds, batch_size=batch_size, 
                                   shuffle=False, num_workers=num_workers)
            return test_loader  # Return single test loader
    
    # Return appropriate loaders
    if len(loaders) == 2:
        return loaders[0], loaders[1]  # train_loader, val_loader
    elif len(loaders) == 1:
        return loaders[0]
    else:
        raise ValueError("At least one data source (files or directories) must be provided")