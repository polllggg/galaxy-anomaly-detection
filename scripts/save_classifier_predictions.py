import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location='cpu'))
model.eval()

images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
labels = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))
anomaly_labels = np.load(os.path.join(DATA_DIR, 'anomaly_labels.npy'))

all_images = np.concatenate([images, anomaly_images])
all_labels = np.concatenate([labels, anomaly_labels])

predictions = []
with torch.no_grad():
    for img in all_images:
        img_t = torch.FloatTensor(np.transpose(img / 255.0, (2, 0, 1))).unsqueeze(0)
        output = model(img_t)
        _, pred = torch.max(output, 1)
        predictions.append(pred.item())

np.save(os.path.join(DATA_DIR, 'all_predictions.npy'), np.array(predictions))
np.save(os.path.join(DATA_DIR, 'all_true_labels.npy'), all_labels)
print(f"Saved {len(predictions)} predictions")
