import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

normal_features  = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
normal_labels    = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
anomaly_features = np.load(os.path.join(DATA_DIR, 'anomaly_features.npy'))

normal_train, normal_test, labels_train, _ = train_test_split(
    normal_features, normal_labels, test_size=0.2, random_state=42
)

strat_idx = []
for cls in range(2, 10):
    idx = np.where(labels_train == cls)[0][:375]
    strat_idx.extend(idx)
ref = normal_train[strat_idx]

# PCA reduces 512 → 380 dims before IF
# Raw IF on 512 dims gives AUROC ~0.59 (curse of dimensionality)
pca = PCA(n_components=380, random_state=42)
pca.fit(ref)
ref_pca  = pca.transform(ref)
test_pca = pca.transform(normal_test[:500])
anom_pca = pca.transform(anomaly_features[:500])

clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
clf.fit(ref_pca)

normal_scores  = -clf.score_samples(test_pca)
anomaly_scores = -clf.score_samples(anom_pca)

np.save(os.path.join(DATA_DIR, 'if_normal_scores.npy'),  normal_scores)
np.save(os.path.join(DATA_DIR, 'if_anomaly_scores.npy'), anomaly_scores)

all_scores = np.concatenate([normal_scores, anomaly_scores])
all_labels = np.concatenate([np.zeros(500), np.ones(500)])
auroc = roc_auc_score(all_labels, all_scores)

print(f"PCA(380) + Isolation Forest AUROC: {auroc:.4f}")
print(f"Normal mean score:  {normal_scores.mean():.4f}")
print(f"Anomaly mean score: {anomaly_scores.mean():.4f}")
