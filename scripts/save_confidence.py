import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
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


def get_confidence(images_array, model, device):
    scores = []
    with torch.no_grad():
        for img in images_array:
            img = img / 255.0
            img = torch.FloatTensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
            output = model(img)
            probs = torch.softmax(output, dim=1)
            max_conf = probs.max().item()
            scores.append(1 - max_conf)  # higher = more anomalous
    return np.array(scores)


print("Normal confidence...")
normal_scores = get_confidence(normal_images, model, device)
print("Anomaly confidence...")
anomaly_scores = get_confidence(anomaly_images, model, device)

np.save(os.path.join(DATA_DIR, 'conf_normal_scores.npy'), normal_scores)
np.save(os.path.join(DATA_DIR, 'conf_anomaly_scores.npy'), anomaly_scores)
print("Saved!")
