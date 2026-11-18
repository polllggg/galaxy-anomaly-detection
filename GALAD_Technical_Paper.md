# GALAD: Galaxy Anomaly Detection Using Deep Learning
## A Comparative Study of Feature-Space and Pixel-Level Anomaly Detection Methods on Astronomical Imagery

**Yersaiyn Maratov** (ID: 2112279) · **Malik Cholponkulov** (ID: 2112113)  
ACSAI — Applied Computer Science and Artificial Intelligence  
Sapienza University of Rome

---

## Abstract

Large-scale astronomical surveys produce galaxy imagery at a volume that precludes manual inspection, creating a practical demand for automated methods to flag morphologically unusual objects. This paper presents **GALAD** (Galaxy Anomaly Detection), a system that benchmarks four anomaly detection strategies on the Galaxy10 DECaLS dataset: a convolutional autoencoder trained on pixel reconstruction, k-Nearest Neighbors (kNN) in a learned feature space, Isolation Forest with PCA dimensionality reduction, and a softmax confidence-based baseline. A ResNet-18 classifier pre-trained on ImageNet and fine-tuned on 10 morphological galaxy classes serves as the shared feature extractor backbone, achieving 91% test accuracy. Both feature-space methods — kNN (AUROC 0.940) and Isolation Forest with PCA(380) (AUROC 0.941) — strongly outperform the pixel-level autoencoder (AUROC 0.610) and the confidence baseline (AUROC 0.615). A critical failure mode of kNN is identified: class imbalance in the reference set causes morphologically underrepresented but normal galaxy classes to receive higher anomaly scores than true anomalies, resolved completely through stratified sampling. Isolation Forest requires PCA dimensionality reduction from 512 to 380 components to overcome the curse of dimensionality, which collapses raw AUROC to 0.593. t-SNE visualization of the 512-dimensional feature space geometrically confirms that anomalous galaxies (Disturbed and Merging classes) occupy a well-separated peripheral region from all normal morphologies. A real-time Gradio-based web interface integrates classification and anomaly scoring into a single deployable tool.

**Keywords:** Anomaly Detection · Galaxy Morphology · Deep Learning · Transfer Learning · ResNet-18 · kNN · Isolation Forest · PCA · t-SNE · Computer Vision · Astronomical Surveys

---

## I. Introduction

### 1.1 Motivation

Modern astronomical surveys — most prominently the DESI Legacy Imaging Surveys — are generating galaxy imagery at a scale and rate that far exceeds human review capacity. The DESI survey alone covers roughly one third of the sky, cataloguing hundreds of millions of objects. Within this data volume, objects with morphologically unusual structures — tidally disrupted galaxies, close or post-merger pairs, galaxies with irregular or asymmetric arm configurations — are scientifically valuable precisely because they fall outside standard morphological classification schemes. These objects represent extreme physical processes: galaxy-galaxy interactions, feedback events, or rare evolutionary stages. Identifying them automatically is therefore not merely a computational convenience but an enabling capability for observational astrophysics.

Citizen science has historically addressed the classification problem through crowdsourcing. The Galaxy Zoo project demonstrated that volunteer classifiers can collectively produce reliable morphological labels for large galaxy catalogs. However, Galaxy Zoo operates on datasets of tens to hundreds of thousands of galaxies; upcoming surveys from the Rubin Observatory (LSST), the Euclid Space Telescope, and the Nancy Grace Roman Space Telescope are expected to produce tens of billions of source detections. Crowdsourced classification will not scale to these volumes.

Machine learning has emerged as the primary candidate for automated survey triage. The Astronomaly framework (Lochner & Bassett, 2021) combines unsupervised representation learning with active learning, allowing astronomers to iteratively guide a detector toward objects they find scientifically interesting. Paz (2024) demonstrated that ML pipelines can identify 1.5 million previously unknown variable objects in archival NEOWISE data, illustrating the practical impact of these methods at scale.

### 1.2 Problem Statement

This paper investigates a focused question: given a labeled dataset of galaxy images, how effectively can different anomaly detection strategies separate morphologically irregular galaxies from normal ones? Specifically, the study evaluates whether the choice of anomaly scoring algorithm (kNN, Isolation Forest, autoencoder) matters more or less than the choice of data representation (raw pixels versus learned deep features). This question has direct practical implications: if feature representation dominates, then even simple algorithms become powerful once good features are available, reducing the need for specialized anomaly detection architectures.

The experimental setup follows a semi-supervised paradigm: anomaly detectors are trained exclusively on normal galaxy samples and never exposed to anomalous examples during training. The ResNet-18 feature extractor, however, was trained on all 10 classes. This distinction is discussed in Section III.

### 1.3 Contributions

The primary contributions of this work are:

1. A systematic comparison of four anomaly detection methods — convolutional autoencoder, kNN in feature space, Isolation Forest with PCA, and softmax confidence — on a standardized galaxy morphology benchmark.
2. An empirical demonstration that class imbalance in the kNN reference set produces severe false positives for underrepresented morphological types, and that stratified sampling fully resolves this failure mode.
3. A quantitative characterization of the curse of dimensionality for Isolation Forest, with a PCA component sweep showing that 380 components (retaining ~99% of variance) is optimal, raising AUROC from 0.593 to 0.941.
4. A t-SNE analysis of the 512-dimensional ResNet-18 feature space providing geometric confirmation of why feature-space methods succeed.
5. A Gradio-based real-time web interface integrating classification and anomaly detection.

---

## II. Dataset

### 2.1 Galaxy10 DECaLS

All experiments use the **Galaxy10 DECaLS** dataset (Leung & Bovy, 2019), sourced from the DESI Legacy Imaging Surveys and labeled using Galaxy Zoo volunteer classifications. The dataset contains **17,736 color images**, each 256 × 256 pixels with three photometric channels corresponding to the g (green), r (red), and z (near-infrared) bands. Images are stored in an HDF5 archive and loaded via the `h5py` library.

The dataset covers **10 morphological classes**:

| Class ID | Morphological Type | Sample Count |
|---|---|---|
| 0 | Disturbed Galaxies | 1,081 |
| 1 | Merging Galaxies | 1,853 |
| 2 | Round Smooth Galaxies | 2,645 |
| 3 | In-between Round Smooth | 2,027 |
| 4 | Cigar Shaped Smooth | 334 |
| 5 | Barred Spiral Galaxies | 2,043 |
| 6 | Unbarred Tight Spiral | 1,829 |
| 7 | Unbarred Loose Spiral | 2,628 |
| 8 | Edge-on without Bulge | 1,423 |
| 9 | Edge-on with Bulge | 1,873 |

### 2.2 Class Imbalance

The dataset is substantially imbalanced. The largest class (Round Smooth, n=2,645) has nearly eight times as many samples as the smallest (Cigar Shaped, n=334). This imbalance has cascading consequences for anomaly detection methods that rely on density estimation (Section IV-B).

### 2.3 Anomaly Definition

For anomaly detection, classes 0 (Disturbed) and 1 (Merging) are designated as **anomalous**; classes 2–9 are **normal**. This partition is motivated astrophysically: disturbed and merging galaxies exhibit irregular morphologies — asymmetric structure, tidal tails, double nuclei, disrupted disks — that deviate from the smooth, spiral, or edge-on morphologies that dominate the catalog. The split yields:

- **Normal set:** 14,802 images (classes 2–9)
- **Anomalous set:** 2,934 images (classes 0–1)

The train/test split is 80/20 with stratification, resulting in approximately 11,841 training and 2,961 test images before the anomaly partition. All anomaly detection evaluations are performed on a balanced test set of 500 normal and 500 anomalous images.

---

## III. Methodology

### 3.1 System Architecture Overview

The GALAD pipeline consists of two stages. The first stage is a **ResNet-18 classifier** fine-tuned on all 10 galaxy classes, which serves both as a morphological classifier and as the shared feature extraction backbone for subsequent anomaly detection. The second stage is a set of **four anomaly detectors** that operate on either raw pixels (autoencoder) or the 512-dimensional feature vectors produced by the classifier backbone (kNN, Isolation Forest, confidence baseline).

The setup is best characterized as **semi-supervised**: the anomaly detectors themselves see only normal training data, but the feature extractor backbone was trained on all 10 classes including the anomalous ones. This means that the 512-dimensional feature vectors implicitly encode information about what anomalous morphologies look like, even though the anomaly scoring step is unsupervised. The implications of this for interpreting results are discussed in Section IV-D.

### 3.2 ResNet-18 Galaxy Classifier

#### Architecture

ResNet-18 (He et al., 2016) is a 18-layer residual network with skip connections that enable stable gradient flow during training. The network was pre-trained on ImageNet, a 1,000-class natural image dataset containing 1.2 million images. For this project, transfer learning was applied: the convolutional backbone (all layers except the final fully connected layer) was retained with ImageNet weights, and the final classification head was replaced with a linear layer mapping 512 input features to 10 output classes corresponding to the galaxy morphologies.

```python
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, 10)
```

This transfer learning approach is justified by evidence that low- and mid-level features learned on natural images (edges, textures, curves) transfer effectively to astronomical imagery, where morphological structure is encoded in similar gradient patterns.

#### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Batch size | 32 |
| Epochs | 10 |
| Train/test split | 80/20, stratified |
| Hardware target | MPS (Apple Silicon) / CUDA / CPU |

#### Data Augmentation

Data augmentation was critical for preventing overfitting. Without augmentation, the model memorized training data: 94.5% train accuracy versus 73.4% test accuracy — a 21.1 percentage point gap. The following transforms were applied to training images only:

- Random horizontal flip
- Random vertical flip
- Random rotation (±30°)

With augmentation, the train-test accuracy gap closed to 4.5% (82.8% train, 78.3% test after augmentation tuning; 91% test accuracy in the final reported configuration). The augmentation policy is astrophysically justified: galaxies have no preferred orientation in the sky, so flips and rotations preserve semantic class identity.

#### Feature Extraction

For the anomaly detection stage, the classification head is removed by replacing it with an identity mapping, converting the model into a pure feature extractor that outputs a 512-dimensional vector per image:

```python
model.fc = nn.Identity()
```

These 512-dimensional vectors encode high-level structural properties — arm morphology, nuclear concentration, symmetry, elongation — learned during classification training.

### 3.3 Convolutional Autoencoder

The autoencoder was trained exclusively on **normal galaxy images** (classes 2–9) to learn a compressed representation of typical galaxy morphologies. The anomaly detection hypothesis is that a model trained only on normal patterns will produce higher reconstruction error when processing anomalous inputs, because it lacks the capacity to represent morphological irregularities it has never seen.

#### Architecture

**Encoder:**
```
Input: 3 × 256 × 256
Conv2d(3 → 32,  kernel=3, stride=2, padding=1)  → 32 × 128 × 128
ReLU
Conv2d(32 → 64, kernel=3, stride=2, padding=1)  → 64 × 64 × 64
ReLU
Conv2d(64 → 128, kernel=3, stride=2, padding=1) → 128 × 32 × 32
ReLU
Flatten → 131,072
Linear(131072 → z)  [bottleneck]
ReLU
```

**Decoder:**
```
Linear(z → 131072)
ReLU
Reshape → 128 × 32 × 32
ConvTranspose2d(128 → 64,  stride=2) → 64 × 64 × 64
ReLU
ConvTranspose2d(64 → 32,   stride=2) → 32 × 128 × 128
ReLU
ConvTranspose2d(32 → 3,    stride=2) → 3 × 256 × 256
Sigmoid
```

The encoder progressively halves the spatial resolution through strided convolution while expanding channel depth (3→32→64→128), then flattens and projects to a bottleneck dimension `z`. The decoder mirrors this process using transposed convolutions. The Sigmoid output constrains reconstructed pixel values to [0, 1].

#### Bottleneck Hyperparameter Sweep

The bottleneck size `z` was treated as a hyperparameter. Three configurations were evaluated:

| Bottleneck Size (z) | Normal MSE | Anomaly MSE | % Difference |
|---|---|---|---|
| 128 | 0.004866 | 0.005314 | 9% |
| 32 | 0.005456 | 0.006346 | 16% |
| **8** | **0.006938** | **0.008212** | **18%** |

A bottleneck of z=8 provides the largest relative separation and was selected for final evaluation. However, visual inspection of the results revealed a critical failure mode (Section IV-C).

#### Anomaly Score

The anomaly score for each image is the per-pixel Mean Squared Error (MSE) between input and reconstruction:

```
score(x) = mean((x - decoder(encoder(x)))²)
```

Training used Adam (lr=0.001), MSE loss, batch size 32, for 10 epochs with the same data augmentation policy as the classifier.

### 3.4 kNN Anomaly Detection in Feature Space

This method exploits the observation that normal galaxy morphologies should cluster together in the 512-dimensional ResNet-18 feature space, while anomalous galaxies should lie farther from all normal clusters.

#### Procedure

1. Extract 512-dimensional feature vectors for all normal training images using the feature extractor backbone.
2. Fit a k-Nearest Neighbors index (k=5) on a reference set of normal feature vectors.
3. For any new image, extract its feature vector and compute the mean Euclidean distance to its 5 nearest normal neighbors.
4. High mean distance = high anomaly score.

```python
knn = NearestNeighbors(n_neighbors=5)
knn.fit(normal_reference_features)
distances, _ = knn.kneighbors(query_features)
anomaly_score = distances.mean(axis=1)
```

#### Reference Set Construction: Stratification

A key design decision is how to construct the normal reference set. Due to memory constraints (8 GB RAM), a subset of 3,000 normal training features was used rather than the full normal training set.

**Original configuration:** The first 3,000 entries from the sequential feature array, dominated by the most frequent classes (Round Smooth: 2,645 samples in the full dataset; Unbarred Loose Spiral: 2,628). This leaves rare but perfectly normal classes like Cigar Shaped (334 total) severely underrepresented.

**Stratified configuration:** 375 samples drawn from each of the 8 normal classes, constructed as:

```python
strat_idx = []
for cls in range(2, 10):
    idx = np.where(labels_train == cls)[0][:375]
    strat_idx.extend(idx)
knn.fit(normal_train_feat[strat_idx])
```

The consequences of this choice are quantified in Section IV-B.

### 3.5 Isolation Forest with PCA

Isolation Forest (Liu et al., 2008) detects anomalies by fitting an ensemble of random binary decision trees (isolation trees) to the data. Each tree recursively partitions the feature space with random axis-aligned splits. The key insight is that anomalies — being rare and structurally distinct — are isolated (reach leaf nodes) in fewer splits on average. The anomaly score is the inverse of the mean path length across all trees.

This method was applied to the same 512-dimensional ResNet-18 feature vectors used by kNN. However, a critical preprocessing step was required.

#### Curse of Dimensionality

Applying Isolation Forest directly to 512-dimensional features yields AUROC 0.593 — near random. In high-dimensional spaces, random axis-aligned splits carry almost no information: path lengths converge for all points regardless of their anomaly status, because any point can appear isolated along some arbitrary dimension.

#### PCA Dimensionality Reduction

PCA was applied prior to Isolation Forest fitting to concentrate the geometric signal into a lower-dimensional space. A component sweep from 50 to 450 components was performed:

| PCA Components | Variance Retained (approx.) | AUROC |
|---|---|---|
| 50 | ~85% | 0.853 |
| 100 | ~92% | 0.899 |
| 200 | ~97% | 0.929 |
| **380** | **~99%** | **0.941** |
| 450 | ~99.5% | 0.929 |

PCA(380) was selected, retaining approximately 99% of variance while reducing dimensionality by 26%. This configuration was used for both PCA fitting and Isolation Forest training, on the same stratified reference set as kNN (375 samples × 8 classes = 3,000 total).

**Final Isolation Forest configuration:**
- PCA components: 380
- n_estimators: 200
- contamination: 0.1
- Reference set: stratified 375 × 8 = 3,000

Anomaly scores are the negated raw scores from `IsolationForest.score_samples()`, so higher values indicate greater anomalousness.

### 3.6 Confidence-Based Baseline

The simplest possible approach uses the trained classifier's own output as an anomaly signal. For each image, the maximum softmax probability across all 10 classes is computed. The anomaly score is defined as:

```
score(x) = 1 − max(softmax(classifier(x)))
```

Low maximum confidence (high anomaly score) is interpreted as indicating that the image is unlike anything the classifier has learned to recognize confidently.

This method is included as a theoretical lower bound to demonstrate the limitations of softmax-based uncertainty for out-of-distribution detection. The reasons for its failure are analyzed in Section IV-C.

---

## IV. Results and Analysis

### 4.1 Classifier Performance

The ResNet-18 classifier achieved **91% overall accuracy** on the held-out test set. Per-class results are as follows:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Disturbed | 0.64 | 0.86 | 0.73 | 1,081 |
| Merging | 0.98 | 0.80 | 0.88 | 1,853 |
| Round Smooth | 0.95 | 0.93 | 0.94 | 2,645 |
| In-between Smooth | — | 0.94 | 0.94 | 2,027 |
| Cigar Shaped | 0.72 | 0.97 | 0.82 | 334 |
| Barred Spiral | 0.93 | 0.95 | 0.94 | 2,043 |
| Unbarred Tight Spiral | 0.90 | 0.90 | 0.90 | 1,829 |
| Unbarred Loose Spiral | 0.89 | 0.86 | 0.88 | 2,628 |
| Edge-on without Bulge | 0.91 | 0.98 | 0.94 | 1,423 |
| Edge-on with Bulge | 0.96 | 0.92 | 0.94 | 1,873 |

**Notable observations:**

- **Disturbed galaxies** achieved the lowest precision (0.64), consistent with their irregular morphology making them visually ambiguous. Their high recall (0.86) indicates the model detects most Disturbed galaxies, but many false positives come from other irregular-looking images.
- **Cigar Shaped galaxies** achieved the highest recall (0.97) despite having by far the fewest training samples (334). Their distinctive elongated morphology is a strong, discriminative signal that generalizes well from limited examples.
- **Merging galaxies** show a precision-recall asymmetry (precision: 0.98, recall: 0.80), suggesting the classifier is conservative — it only predicts Merging when very confident, but misses some merger cases.

These per-class characteristics have direct implications for anomaly detection: Disturbed galaxies, as the most ambiguous class even for the classifier, are also expected to be the hardest anomalies to detect.

### 4.2 Anomaly Detection Comparison

All methods were evaluated on a balanced test set: 500 normal samples (drawn from classes 2–9) and 500 anomalous samples (Disturbed and Merging). AUROC (Area Under the Receiver Operating Characteristic Curve) was used as the primary metric, as it summarizes detector performance across all possible decision thresholds without requiring a fixed operating point.

**Table 1: Anomaly Detection AUROC Summary**

| Method | AUROC |
|---|---|
| Isolation Forest — PCA(380) + stratified | **0.9414** |
| kNN — stratified reference set | **0.9398** |
| kNN — original (non-stratified) reference set | 0.9382 |
| Confidence-Based Baseline | 0.615 |
| Convolutional Autoencoder (bottleneck=8) | 0.610 |

The results bifurcate cleanly into two groups separated by more than 0.32 AUROC points:
- **Feature-space methods** (kNN, Isolation Forest+PCA): AUROC > 0.93
- **Pixel-level and softmax methods** (Autoencoder, Confidence): AUROC ≈ 0.61

The difference between the two top-performing methods (kNN stratified vs. IF+PCA) is 0.0016, which is within noise on a 1,000-sample evaluation set. This near-identical performance strongly implies that the **choice of anomaly scoring algorithm is secondary** to the choice of feature representation.

### 4.3 kNN Reference Set Imbalance: A Critical Failure Mode

The original (non-stratified) kNN configuration exposed a severe bias problem. Table 2 shows per-class mean anomaly scores for 10 representative samples per class under both configurations:

**Table 2: Per-class kNN anomaly scores**

| Class | True Label | kNN Original | kNN Stratified |
|---|---|---|---|
| Round Smooth | Normal | 2.86 | 3.59 |
| In-between Smooth | Normal | 2.82 | 2.78 |
| Cigar Shaped | Normal | 7.79 | 2.93 |
| Barred Spiral | Normal | 12.61 | 3.82 |
| Unbarred Tight Spiral | Normal | 8.15 | 3.74 |
| Unbarred Loose Spiral | Normal | 8.17 | 3.19 |
| Edge-on without Bulge | Normal | 13.82 | 2.39 |
| Edge-on with Bulge | Normal | **15.21** | 3.07 |
| **Disturbed** | **Anomalous** | 9.07 | **7.58** |
| **Merging** | **Anomalous** | 8.46 | **8.99** |

In the original configuration, Edge-on with Bulge galaxies (score: 15.21) receive dramatically higher anomaly scores than actual Disturbed anomalies (score: 9.07). The detector was effectively measuring **dissimilarity from Round Smooth galaxies**, not morphological irregularity, because Round Smooth and Unbarred Loose Spiral galaxies dominated the reference set.

After stratification (375 samples per class), all normal classes compress to scores of 2–4, while anomalous classes remain elevated at 7–9, producing clean separation. The AUROC improvement from stratification (0.9382 → 0.9398) understates the practical significance of this fix: the stratified detector is qualitatively correct, while the original would generate massive false positive rates for Edge-on and Barred Spiral galaxies in production.

**Lesson:** For density-based anomaly detection on imbalanced datasets, the reference distribution must be representative of the full normal population, not just the most frequent subgroup.

### 4.4 Isolation Forest: The PCA Necessity

Raw Isolation Forest on 512-dimensional features achieves AUROC 0.593. This is a direct consequence of the curse of dimensionality. In 512 dimensions, random axis-aligned partitions have extremely low probability of being informative about morphological structure. All data points — normal and anomalous alike — tend to require similar numbers of splits to isolate, because any point appears outlying along some subset of the 512 axes. The path length statistic that Isolation Forest relies on converges to the same expected value for all points.

PCA addresses this by rotating the coordinate system to align principal axes with the directions of maximum variance, concentrating the informative geometric signal into fewer dimensions. With 380 components (≈ 99% variance retained), the effective dimensionality drops enough that morphologically distinct points — the anomalous galaxies — become genuinely isolated earlier in the tree ensemble. This raises AUROC from 0.593 to 0.941.

Notably, 450 components (0.929) performs slightly worse than 380 (0.941), suggesting that the final ~1% of variance in dimensions 381–450 introduces noise that slightly degrades isolation scoring. The optimal component count is therefore not simply "keep as much variance as possible" but reflects a signal-to-noise trade-off in the tail of the spectrum.

### 4.5 Why the Autoencoder Fails

The convolutional autoencoder's AUROC of 0.610 represents a near-total failure to detect morphological anomalies. The root cause is the choice of loss function. MSE penalizes pixel-level intensity differences: bright pixels contribute proportionally more to total error than dark pixels. A galaxy at the center of the image surrounded by bright extended structure will register high error simply due to brightness, regardless of whether its morphology is regular or irregular.

The bottleneck experiment confirms this interpretation. At bottleneck z=8, the model cannot represent all image information, so it must choose what to prioritize. MSE optimization leads the model to prioritize accurate reconstruction of pixel intensities — principally the bright central galaxy body — rather than morphological structure like arm arrangement or asymmetry. Visual inspection at z=8 revealed that the highest-error images were predominantly **dark-background or diffuse-field images**, not morphologically disturbed galaxies. The detector learned to flag images with unusual brightness distributions, not unusual shapes.

At z=32, high-error images were more frequently genuinely disturbed morphologies (values 0.031–0.048), suggesting that a larger bottleneck can encode some structural information — but the effect is not strong enough to separate class distributions at the AUROC level.

A perceptual loss function — one that computes differences in a learned feature space rather than at the pixel level — would likely improve results substantially, as the feature space encodes structural properties. This remains a direction for future work.

### 4.6 Why the Confidence Baseline Fails

The confidence baseline achieves AUROC 0.615 despite conceptual appeal. The failure has a clear architectural cause: the ResNet-18 classifier was trained on **all 10 classes**, including Disturbed and Merging galaxies. As a result, the classifier has dedicated output neurons for the anomalous classes and assigns them high confidence — the mean softmax maximum probability for anomalous galaxies exceeds 90%. From the classifier's perspective, they are not out-of-distribution; they are a well-represented training category.

Additionally, softmax is architecturally overconfident: outputs sum to exactly 1.0, so the network cannot express genuine uncertainty even when it should. A logit of 2.0 for the "best" class and -1.0 for all others produces a high softmax probability regardless of how far the input is from the training distribution.

This failure mode highlights an important design principle: **the anomaly signal must come from a representation space where anomalies are actually out-of-distribution**. The classifier's final layer is explicitly trained to make anomalous galaxies in-distribution. Confidence-based detection works best when the classifier has never seen the anomalous class during training.

### 4.7 t-SNE Feature Space Visualization

t-SNE (van der Maaten & Hinton, 2008) was applied to visualize the structure of the 512-dimensional feature space. A sample of 500 normal and 300 anomalous feature vectors was embedded into 2 dimensions using perplexity=40 and 1,000 iterations.

The visualization reveals two key structural properties:

1. **Normal class separation:** The eight normal morphological classes form distinct, well-separated clusters in the 2D embedding. Round Smooth, Barred Spiral, Edge-on, Cigar Shaped, and Spiral classes each occupy recognizable sub-regions. This confirms that the ResNet-18 features encode fine-grained morphological differences.

2. **Anomaly isolation:** Disturbed and Merging galaxies (anomalous classes) land in a peripherally isolated region, clearly separated from the dense normal cluster. They do not form a single tight cluster of their own — their irregular morphologies are diverse — but they are consistently pushed away from the normal morphology manifold.

This geometric structure directly explains why both kNN (measuring distance to normal neighbors) and Isolation Forest (measuring ease of isolation) succeed: the feature space places anomalies where both methods are most effective. It also explains why both methods give almost identical AUROC scores — they are exploiting the same geometric signal through different algorithmic mechanisms.

---

## V. Implementation Details

### 5.1 Code Architecture

The project is organized into four functional layers:

```
GALAD-(Code)/
├── src/              # Core modules: dataset, training, feature extraction, detectors
│   ├── dataset.py           — GalaxyDataset class, data loading, train/test split
│   ├── train.py             — ResNet-18 fine-tuning with augmentation
│   ├── autoencoder.py       — Autoencoder definition and MSE score computation
│   ├── knn_detector.py      — Feature extraction and kNN fitting/evaluation
│   └── confidence_detector.py — Softmax confidence scoring
├── evaluation/       # Offline evaluation and visualization
│   ├── full_metrics.py      — Unified AUROC computation across all methods
│   ├── isolation_forest.py  — PCA + Isolation Forest pipeline
│   ├── visualize.py         — kNN and autoencoder score distribution plots
│   └── tsne.py              — t-SNE embedding and visualization
├── scripts/          # One-time data preparation utilities
│   ├── save_classifier_predictions.py — Saves classifier outputs for all data
│   ├── save_confidence.py             — Saves confidence scores
│   ├── save_demo_data.py              — Prepares inference subset for demo
│   └── examples.py                   — Saves example images for inspection
├── demo/
│   └── demo_gradio.py       — Real-time web interface
└── models/
    ├── galaxy_classifier.pth    — Fine-tuned ResNet-18 weights
    └── Galaxy_Autoencoder.pth   — Trained autoencoder weights
```

### 5.2 Hardware and Device Handling

All training and inference modules use dynamic device selection:

```python
device = torch.device(
    'mps' if torch.backends.mps.is_available()   # Apple Silicon GPU
    else 'cuda' if torch.cuda.is_available()      # NVIDIA GPU
    else 'cpu'
)
```

This allows the code to run on Apple M-series hardware (via Metal Performance Shaders), CUDA-equipped GPUs, or CPU without modification.

### 5.3 Gradio Web Demo

The real-time interface integrates both classification and anomaly detection in a single inference pass:

```
Input: Galaxy image (PIL, any resolution)
→ Resize to 256 × 256
→ Normalize to [0, 1]
→ ResNet-18 classifier → morphological class + confidence
→ ResNet-18 feature extractor (fc = Identity) → 512-dim vector
→ kNN (stratified, k=5) → anomaly score
→ Threshold comparison → Normal / Anomaly decision
```

The decision threshold is **data-driven**: the 95th percentile of kNN anomaly scores on the normal training reference set, computed at startup:

```python
ref_distances, _ = knn.kneighbors(normal_features[strat_idx])
anomaly_threshold = np.percentile(ref_distances.mean(axis=1), 95)
```

This contrasts with the fixed threshold of 7.0 reported in the static evaluation, making the demo threshold adaptive to the actual score distribution of the loaded reference set.

Example outputs from the demo:
- **Round Smooth galaxy:** Classification: Round Smooth (94.2% confidence), Anomaly Score: 3.1 → **NORMAL**
- **Merging galaxy:** Classification: Merging (87.5% confidence), Anomaly Score: 9.4 → **ANOMALY**

---

## VI. Discussion

### 6.1 Feature Representation as the Primary Factor

The central empirical finding is unambiguous: the anomaly detection algorithm matters far less than the feature representation. kNN and Isolation Forest+PCA both achieve AUROC ≈ 0.94 using the same ResNet-18 features. The autoencoder and confidence baseline, using pixel-level or classifier-output representations, both achieve AUROC ≈ 0.61. The gap between representation classes (~0.33 AUROC) vastly exceeds the gap within the feature-space class (~0.001 AUROC).

This finding has a practical implication for practitioners building anomaly detection systems for visual data: **invest in the feature extractor, not the anomaly scoring algorithm**. A pre-trained deep network with good domain coverage will likely produce feature vectors where simple methods like kNN perform near-optimally, without any specialized anomaly detection architecture.

### 6.2 The Semi-Supervised Setup and Its Implications

An important caveat is that the ResNet-18 backbone was trained on all 10 classes, including the anomalous ones. The feature space therefore encodes not just "what normal galaxies look like" but also "what Disturbed and Merging galaxies look like, and how they differ from normal ones." The t-SNE visualization showing anomalies in a distinct peripheral region is partly a consequence of this supervised training signal.

A stricter semi-supervised setup — in which the backbone is trained only on normal classes — would produce a feature space with different geometry. It is plausible that distance-based anomaly detection would still perform well (since the features would encode what normal looks like without reference to the anomalous classes), but the AUROC numbers would likely change. This experiment is a natural extension for future work.

### 6.3 Scalability Considerations

The current implementation operates on a 17,736-image subset of the Galaxy Zoo data. At the scale of full DESI or Rubin surveys (hundreds of millions of objects), several engineering considerations arise:

- **kNN memory:** The full normal feature set at 512 floats × 10,000+ images requires tens of megabytes of memory, manageable. But kNN query time scales as O(n × d) per query; approximate nearest neighbor indices (FAISS, ScaNN) would be needed at >10M scale.
- **Autoencoder reconstruction:** Per-image batch inference is trivially parallelizable across GPUs.
- **PCA fitting:** PCA on millions of 512-dimensional vectors is feasible with randomized SVD, and PCA transformation is a simple matrix multiplication at inference.

The Isolation Forest requires more care at scale: fitting 200 trees on millions of samples increases memory and fit time significantly. Random subsampling (already standard in sklearn's implementation) provides natural scalability.

---

## VII. Conclusions

This paper presented GALAD, a system benchmarking four anomaly detection methods for morphologically unusual galaxy identification on the Galaxy10 DECaLS dataset. The key findings are:

**1. Feature representation dominates algorithm choice.**  
kNN and Isolation Forest+PCA both achieve AUROC ≈ 0.94 in the ResNet-18 feature space. The autoencoder and confidence baseline both achieve AUROC ≈ 0.61. The algorithmic gap within each representation class is negligible compared to the gap between representation classes.

**2. Stratified reference sets are non-negotiable for kNN.**  
Class imbalance in the kNN reference set causes underrepresented but morphologically normal classes to receive anomaly scores exceeding those of true anomalies. Stratified sampling (375 samples per class) resolves this completely and is essential for any deployment scenario.

**3. Pixel-level MSE is a poor proxy for morphological anomaly.**  
The autoencoder's MSE loss encodes brightness and contrast rather than structural irregularity. Aggressive bottleneck compression exacerbates this by forcing the model to prioritize luminosity over morphology. Perceptual loss functions that operate on deep feature differences would likely resolve this limitation.

**4. High-dimensional spaces require explicit treatment.**  
Isolation Forest on raw 512-dimensional features scores 0.593. PCA to 380 components raises this to 0.941. This is a concrete, measurable manifestation of the curse of dimensionality and is generalizable to any isolation-based or density-based method applied to deep feature vectors.

**5. ImageNet pre-training transfers effectively to astronomical imagery.**  
ResNet-18 achieves 91% accuracy on 10 galaxy morphology classes with 10 epochs of fine-tuning, confirming that low- and mid-level visual features generalize across domains.

### Future Work

- **Scale to full Galaxy Zoo (441,000+ images)** to test whether failure modes invisible at 17,736-image scale emerge at larger volumes.
- **Replace MSE loss with perceptual loss** in the autoencoder, computing reconstruction error in a learned feature space rather than at the pixel level.
- **Cross-telescope generalization testing** across Hubble, Euclid, and Rubin LSST imaging to assess domain shift effects.
- **Active learning integration** following the Astronomaly framework, enabling expert feedback to iteratively refine the anomaly detector toward astronomically significant classes.
- **Strictly semi-supervised backbone training** — training the feature extractor only on normal classes — to characterize the contribution of the supervised signal to anomaly separation in the feature space.

---

## References

[1] Leung, H. W. & Bovy, J. (2019). "Deep learning of multi-element abundances from high-resolution spectroscopic data." *Monthly Notices of the Royal Astronomical Society*, 483(3), 3255–3277.

[2] Lochner, M. & Bassett, B. A. (2021). "Astronomaly: at the intersection of unsupervised and active anomaly detection in astronomical data." *Astronomy and Computing*, 36, 100481.

[3] He, K., Zhang, X., Ren, S. & Sun, J. (2016). "Deep residual learning for image recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

[4] Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). "Isolation forest." *IEEE International Conference on Data Mining (ICDM)*, 413–422.

[5] van der Maaten, L. & Hinton, G. (2008). "Visualizing data using t-SNE." *Journal of Machine Learning Research*, 9, 2579–2605.

[6] Paz, M. (2024). "VarWISE: Machine Learning Classification of Variable Objects in NEOWISE Data." *The Astronomical Journal*.

[7] Dieleman, S., Willett, K. W. & Dambre, J. (2015). "Rotation-invariant convolutional neural networks for galaxy morphology prediction." *Monthly Notices of the Royal Astronomical Society*, 450(2), 1441–1459.

[8] Galaxy10 DECaLS Dataset. astroNN library. https://astronn.readthedocs.io/en/latest/galaxy10.html

---

*Authors: Yersaiyn Maratov and Malik Cholponkulov — ACSAI, Sapienza University of Rome*  
*Project repository: GALAD (Galaxy Anomaly Detection)*
