import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import torchvision.transforms as transforms
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')


class GalaxyDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index] / 255.0
        label = self.labels[index]
        image = np.transpose(image, (2, 0, 1))
        image = torch.FloatTensor(image)
        if self.augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30)
            ])
            image = transform(image)
        return image, label


with h5py.File(os.path.join(DATA_DIR, 'Galaxy10 DECaLS.h5'), 'r') as f:
    images = np.array(f['images'])
    labels = np.array(f['ans'])
    train_img, test_img, train_lab, test_lab = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

train_dataset = GalaxyDataset(train_img, train_lab, augment=True)
test_dataset = GalaxyDataset(test_img, test_lab, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, 10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
model = model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images_batch, labels_batch in train_loader:
        images_batch = images_batch.float().to(device)
        labels_batch = labels_batch.long().to(device)

        outputs = model(images_batch)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {train_acc:.1f}%")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images_batch, labels_batch in test_loader:
        images_batch = images_batch.float().to(device)
        labels_batch = labels_batch.long().to(device)
        outputs = model(images_batch)
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.1f}%")

torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'galaxy_classifier.pth'))
print("Model saved")
