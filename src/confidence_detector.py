import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location=device))
model = model.to(device)
model.eval()

normal_images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))


def get_confidence(images_array, model, device, batch_size=64):
    imgs = torch.FloatTensor(np.transpose(images_array / 255.0, (0, 3, 1, 2)))
    loader = DataLoader(TensorDataset(imgs), batch_size=batch_size)
    confidences = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            probs = torch.softmax(model(batch), dim=1)
            max_conf = probs.max(dim=1).values
            confidences.extend(max_conf.cpu().numpy())
    return np.array(confidences)


print("Getting normal confidence...")
normal_confidence = get_confidence(normal_images[:500], model, device)

print("Getting anomaly confidence...")
anomaly_confidence = get_confidence(anomaly_images[:500], model, device)

# Anomaly score = 1 - confidence (higher means more anomalous)
normal_scores = 1 - normal_confidence
anomaly_scores = 1 - anomaly_confidence

np.save(os.path.join(DATA_DIR, 'conf_normal_scores.npy'), normal_scores)
np.save(os.path.join(DATA_DIR, 'conf_anomaly_scores.npy'), anomaly_scores)

print(f"Normal mean confidence:  {normal_confidence.mean():.4f}")
print(f"Anomaly mean confidence: {anomaly_confidence.mean():.4f}")
print("Saved confidence scores!")
