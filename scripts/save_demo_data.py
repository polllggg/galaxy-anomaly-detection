import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location='cpu'))
model.eval()

normal_images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))

demo_images = np.concatenate([normal_images[:10], anomaly_images[:10]])
demo_labels = np.array([0] * 10 + [1] * 10)

features_list = []
model.fc = nn.Identity()
with torch.no_grad():
    for img in demo_images:
        img_t = torch.FloatTensor(np.transpose(img / 255.0, (2, 0, 1))).unsqueeze(0)
        feat = model(img_t).numpy().flatten()
        features_list.append(feat)

model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location='cpu'))
model.eval()

predictions = []
confidences = []
with torch.no_grad():
    for img in demo_images:
        img_t = torch.FloatTensor(np.transpose(img / 255.0, (2, 0, 1))).unsqueeze(0)
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        predictions.append(pred.item())
        confidences.append(conf.item())

np.save(os.path.join(DATA_DIR, 'demo_images.npy'), demo_images)
np.save(os.path.join(DATA_DIR, 'demo_labels.npy'), demo_labels)
np.save(os.path.join(DATA_DIR, 'demo_predictions.npy'), np.array(predictions))
np.save(os.path.join(DATA_DIR, 'demo_confidences.npy'), np.array(confidences))
np.save(os.path.join(DATA_DIR, 'demo_features.npy'), np.array(features_list))
print("Demo data saved!")
