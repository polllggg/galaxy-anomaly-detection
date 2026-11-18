import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')


class GalaxyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index] / 255.0
        label = self.labels[index]
        image = np.transpose(image, (2, 0, 1))
        return image, label


with h5py.File(os.path.join(DATA_DIR, 'Galaxy10 DECaLS.h5'), 'r') as f:
    images = np.array(f['images'])
    labels = np.array(f['ans'])
    train_img, test_img, train_lab, test_lab = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_lab)}, Test: {len(test_lab)}")

train_dataset = GalaxyDataset(train_img, train_lab)
test_dataset = GalaxyDataset(test_img, test_lab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images_batch, labels_batch = next(iter(train_loader))
print(f"Batch images shape: {images_batch.shape}")
print(f"Batch labels shape: {labels_batch.shape}")
