from torch.utils.data import DataLoader, TensorDataset 
from torch import tensor, float32
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def torch_DataLoader(X, batch_size,
                     device='cpu', drop_last=False, shuffle=True):
   
    training_images, test_images = train_test_split(X, test_size=0.2, random_state=42)
    train_images, val_images = train_test_split(training_images, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(
        tensor(train_images, device=device, dtype=float32),
        tensor(train_images, device=device, dtype=float32)
    )
    val_dataset = TensorDataset(
        tensor(val_images, device=device, dtype=float32),
        tensor(val_images, device=device, dtype=float32)
    )
    test_dataset = TensorDataset(
        tensor(test_images, device=device, dtype=float32),
        tensor(test_images, device=device, dtype=float32)
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              drop_last=drop_last,
                              shuffle=shuffle)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            shuffle=shuffle)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             drop_last=drop_last,
                             shuffle=shuffle)

    return train_loader, val_loader, test_loader

def torch_DataLoader_SR(X, batch_size,
                        device='cpu', drop_last=False, shuffle=True, size=(7,7)):
    
    
    training_images, test_images = train_test_split(X, test_size=0.2, random_state=42)
    train_images, val_images = train_test_split(training_images, test_size=0.2, random_state=42)
    
    
    train_dataset = TensorDataset(
        F.interpolate(tensor(train_images, device=device, dtype=float32), size=size, mode='bilinear', align_corners=False),
        tensor(train_images, device=device, dtype=float32)
    )
    val_dataset = TensorDataset(
        F.interpolate(tensor(val_images, device=device, dtype=float32), size=size, mode='bilinear', align_corners=False),
        tensor(val_images, device=device, dtype=float32)
    )
    test_dataset = TensorDataset(
        F.interpolate(tensor(test_images, device=device, dtype=float32), size=size, mode='bilinear', align_corners=False),
        tensor(test_images, device=device, dtype=float32)
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              drop_last=drop_last,
                              shuffle=shuffle)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            shuffle=shuffle)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             drop_last=drop_last,
                             shuffle=shuffle)
    return train_loader, val_loader, test_loader
