import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')

normal_features  = np.load(os.path.join(DATA_DIR, 'normal_features.npy'))
normal_labels    = np.load(os.path.join(DATA_DIR, 'normal_labels.npy'))
anomaly_features = np.load(os.path.join(DATA_DIR, 'anomaly_features.npy'))
anomaly_labels   = np.load(os.path.join(DATA_DIR, 'anomaly_labels.npy'))

rng = np.random.default_rng(42)
normal_idx  = rng.choice(len(normal_features),  500, replace=False)
anomaly_idx = rng.choice(len(anomaly_features), 300, replace=False)

sample_feats  = np.concatenate([normal_features[normal_idx], anomaly_features[anomaly_idx]])
sample_labels = np.concatenate([normal_labels[normal_idx],   anomaly_labels[anomaly_idx]])

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
emb = tsne.fit_transform(sample_feats)
print("Done.")

class_names = [
    'Disturbed', 'Merging', 'Round Smooth', 'In-between Smooth',
    'Cigar Shaped', 'Barred Spiral', 'Unbarred Tight Spiral',
    'Unbarred Loose Spiral', 'Edge-on w/o Bulge', 'Edge-on w/ Bulge'
]
normal_colors = [
    '#4FC3F7', '#81C784', '#FFB74D', '#CE93D8',
    '#80DEEA', '#A5D6A7', '#FFCC80', '#B39DDB'
]
anomaly_colors = {'Disturbed': '#F44336', 'Merging': '#FF7043'}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0D1117')

for ax in axes:
    ax.set_facecolor('#0D1117')
    ax.tick_params(colors='#8B949E')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')

# ── Left: binary normal vs anomalous ────────────────────────────────────────
n_mask = sample_labels >= 2
a_mask = sample_labels < 2

axes[0].scatter(emb[n_mask, 0], emb[n_mask, 1],
                c='#3FB950', s=12, alpha=0.5, label='Normal (classes 2–9)')
axes[0].scatter(emb[a_mask, 0], emb[a_mask, 1],
                c='#F85149', s=25, alpha=0.9, label='Anomalous (Disturbed + Merging)',
                zorder=5)
axes[0].set_title('Normal vs Anomalous', color='white', fontsize=14, pad=12)
axes[0].legend(facecolor='#161B22', edgecolor='#30363D',
               labelcolor='white', fontsize=10)

# ── Right: per-class ─────────────────────────────────────────────────────────
normal_class_list = [cls for cls in range(2, 10)]
for i, cls in enumerate(normal_class_list):
    mask = sample_labels == cls
    if mask.sum() == 0:
        continue
    axes[1].scatter(emb[mask, 0], emb[mask, 1],
                    c=normal_colors[i], s=12, alpha=0.55,
                    label=class_names[cls])

for cls, color in anomaly_colors.items():
    cls_id = class_names.index(cls)
    mask = sample_labels == cls_id
    if mask.sum() == 0:
        continue
    axes[1].scatter(emb[mask, 0], emb[mask, 1],
                    c=color, s=30, alpha=0.95, label=cls,
                    marker='*', zorder=5)

axes[1].set_title('Per-class breakdown', color='white', fontsize=14, pad=12)
axes[1].legend(facecolor='#161B22', edgecolor='#30363D',
               labelcolor='white', fontsize=8, ncol=2)

for ax in axes:
    ax.set_xlabel('t-SNE dim 1', color='#8B949E', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', color='#8B949E', fontsize=10)

fig.suptitle('ResNet-18 Feature Space  (t-SNE, 512 → 2 dims)',
             color='white', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'tsne_features.png'),
            dpi=150, bbox_inches='tight', facecolor='#0D1117')
print(f"Saved: {OUTPUTS_DIR}/tsne_features.png")
