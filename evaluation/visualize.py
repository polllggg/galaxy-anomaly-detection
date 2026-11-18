import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')

normal_features = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
normal_labels = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
anomaly_features = np.load(os.path.join(DATA_DIR, 'anomaly_features.npy'))

normal_train, normal_test, labels_train, _ = train_test_split(
    normal_features, normal_labels, test_size=0.2, random_state=42
)

strat_idx = []
for cls in range(2, 10):
    idx = np.where(labels_train == cls)[0][:375]
    strat_idx.extend(idx)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_train[strat_idx])

anomaly_distances, _ = knn.kneighbors(anomaly_features[:500])
normal_distances, _ = knn.kneighbors(normal_test[:500])

normal_score = normal_distances.mean(axis=1)
anomaly_score = anomaly_distances.mean(axis=1)

plt.figure()
plt.hist(normal_score, bins=50, alpha=0.5, label='Normal', color='blue')
plt.hist(anomaly_score, bins=50, alpha=0.5, label='Anomaly', color='red')
plt.xlabel('Anomaly Score (kNN distance)')
plt.ylabel('Count')
plt.title('kNN Anomaly Score Distribution')
plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR, 'knn_distribution.png'))
plt.show()

ae_normal = np.load(os.path.join(DATA_DIR, 'ae_normal_scores.npy'))
ae_anomaly = np.load(os.path.join(DATA_DIR, 'ae_anomaly_scores.npy'))

plt.figure()
plt.hist(ae_normal, bins=50, alpha=0.5, label='Normal', color='blue')
plt.hist(ae_anomaly, bins=50, alpha=0.5, label='Anomaly', color='red')
plt.xlabel('Anomaly Score (reconstruction error)')
plt.ylabel('Count')
plt.title('Autoencoder Anomaly Score Distribution')
plt.legend()
plt.savefig(os.path.join(OUTPUTS_DIR, 'autoencoder_distribution.png'))
plt.show()
