import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

normal_features = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
anomaly_features = np.load(os.path.join(DATA_DIR, 'anomaly_features.npy'))

normal_labels = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
normal_train, normal_test, labels_train, _ = train_test_split(
    normal_features, normal_labels, test_size=0.2, random_state=42
)

knn_orig = NearestNeighbors(n_neighbors=5)
knn_orig.fit(normal_train[:3000])

strat_idx = []
for cls in range(2, 10):
    idx = np.where(labels_train == cls)[0][:375]
    strat_idx.extend(idx)
knn_strat = NearestNeighbors(n_neighbors=5)
knn_strat.fit(normal_train[strat_idx])

eval_labels = np.concatenate([np.zeros(500), np.ones(500)])

def auroc_knn(knn, normal_test, anomaly_features):
    nd, _ = knn.kneighbors(normal_test[:500])
    ad, _ = knn.kneighbors(anomaly_features[:500])
    scores = np.concatenate([nd.mean(axis=1), ad.mean(axis=1)])
    return roc_auc_score(eval_labels, scores)

pca = PCA(n_components=380, random_state=42)
pca.fit(normal_train[strat_idx])
ref_pca  = pca.transform(normal_train[strat_idx])
test_pca = pca.transform(normal_test[:500])
anom_pca = pca.transform(anomaly_features[:500])
clf_if = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
clf_if.fit(ref_pca)
if_scores = np.concatenate([-clf_if.score_samples(test_pca), -clf_if.score_samples(anom_pca)])

print("=== Anomaly Detection AUROC ===")
print(f"kNN (original):          {auroc_knn(knn_orig,  normal_test, anomaly_features):.4f}")
print(f"kNN (stratified):        {auroc_knn(knn_strat, normal_test, anomaly_features):.4f}")
print(f"Isolation Forest (PCA):  {roc_auc_score(eval_labels, if_scores):.4f}")

ae_normal = np.load(os.path.join(DATA_DIR, 'ae_normal_scores.npy'))
ae_anomaly = np.load(os.path.join(DATA_DIR, 'ae_anomaly_scores.npy'))
ae_all = np.concatenate([ae_normal, ae_anomaly])
ae_labels = np.concatenate([np.zeros(len(ae_normal)), np.ones(len(ae_anomaly))])
print(f"Autoencoder: {roc_auc_score(ae_labels, ae_all):.4f}")

conf_normal = np.load(os.path.join(DATA_DIR, 'conf_normal_scores.npy'))
conf_anomaly = np.load(os.path.join(DATA_DIR, 'conf_anomaly_scores.npy'))
conf_all = np.concatenate([conf_normal, conf_anomaly])
conf_labels = np.concatenate([np.zeros(len(conf_normal)), np.ones(len(conf_anomaly))])
print(f"Confidence:  {roc_auc_score(conf_labels, conf_all):.4f}")

class_names = [
    'Disturbed', 'Merging', 'Round Smooth',
    'In-between Smooth', 'Cigar Shaped', 'Barred Spiral',
    'Unbarred Tight Spiral', 'Unbarred Loose Spiral',
    'Edge-on without Bulge', 'Edge-on with Bulge'
]

predictions = np.load(os.path.join(DATA_DIR, 'all_predictions.npy'))
true_labels = np.load(os.path.join(DATA_DIR, 'all_true_labels.npy'))

print("\n=== Classifier Performance ===")
print(classification_report(true_labels, predictions, target_names=class_names))
