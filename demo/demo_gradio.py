import os
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')

class_names = [
    'Disturbed', 'Merging', 'Round Smooth',
    'In-between Smooth', 'Cigar Shaped', 'Barred Spiral',
    'Unbarred Tight Spiral', 'Unbarred Loose Spiral',
    'Edge-on without Bulge', 'Edge-on with Bulge'
]

# Load ResNet once and share weights between classifier and feature extractor
_base = models.resnet18(weights='DEFAULT')
_base.fc = nn.Linear(512, 10)
_base.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location='cpu'))
_base.eval()

classifier = _base

feature_extractor = models.resnet18(weights='DEFAULT')
feature_extractor.fc = nn.Linear(512, 10)
feature_extractor.load_state_dict(_base.state_dict())
feature_extractor.fc = nn.Identity()
feature_extractor.eval()

normal_features = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
normal_labels = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))

strat_idx = []
for cls in range(2, 10):
    idx = np.where(normal_labels == cls)[0][:375]
    strat_idx.extend(idx)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_features[strat_idx])

# Data-driven threshold: 95th percentile of normal training scores
ref_distances, _ = knn.kneighbors(normal_features[strat_idx])
anomaly_threshold = np.percentile(ref_distances.mean(axis=1), 95)
print(f"Anomaly threshold (95th pct of normal): {anomaly_threshold:.4f}")


def analyze(image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_tensor = torch.FloatTensor(np.transpose(img_array, (2, 0, 1))).unsqueeze(0)

    with torch.no_grad():
        output = classifier(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        pred_class = class_names[predicted.item()]

        features = feature_extractor(img_tensor).numpy().flatten()
        distances, _ = knn.kneighbors(features.reshape(1, -1))
        anomaly_score = distances.mean()

    status = "ANOMALY — Requires attention!" if anomaly_score > anomaly_threshold else "NORMAL"
    return (
        f"Classification: {pred_class}\n"
        f"Confidence: {confidence.item():.1%}\n"
        f"Anomaly Score: {anomaly_score:.2f} (threshold: {anomaly_threshold:.2f})\n"
        f"Status: {status}"
    )


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Galaxy Anomaly Detector"
)

demo.launch(share=True)
