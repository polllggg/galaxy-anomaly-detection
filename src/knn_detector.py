import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')


def extract_and_save_features():
    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'galaxy_classifier.pth'), map_location=device))
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    normal_images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
    anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))

    def extract_features(images_array):
        features = []
        with torch.no_grad():
            for img in images_array:
                img = img / 255.0
                img = torch.FloatTensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
                feature = model(img)
                features.append(feature.cpu().numpy().flatten())
        return np.array(features)

    print('Extracting normal features...')
    normal_features = extract_features(normal_images)
    print('Extracting anomaly features...')
    anomaly_features = extract_features(anomaly_images)

    np.save(os.path.join(DATA_DIR, 'normal_features.npy'), normal_features)
    np.save(os.path.join(DATA_DIR, 'anomaly_features.npy'), anomaly_features)
    print('Features saved!')


normal_features = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
normal_labels = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
anomaly_features = np.load(os.path.join(DATA_DIR, 'anomaly_features.npy'))

normal_train_feat, normal_test_feat, labels_train, _ = train_test_split(
    normal_features, normal_labels, test_size=0.2, random_state=42
)

print(f"Train features: {len(normal_train_feat)}")
print(f"Test features:  {len(normal_test_feat)}")

strat_idx = []
for cls in range(2, 10):
    idx = np.where(labels_train == cls)[0][:375]
    strat_idx.extend(idx)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_train_feat[strat_idx])

normal_distances, _ = knn.kneighbors(normal_test_feat[:500])
anomaly_distances, _ = knn.kneighbors(anomaly_features[:500])

normal_scores = normal_distances.mean(axis=1)
anomaly_scores = anomaly_distances.mean(axis=1)

print(f"Normal mean distance:  {normal_scores.mean():.4f}")
print(f"Anomaly mean distance: {anomaly_scores.mean():.4f}")
