from PIL import Image
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')

normal_images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))

for i in range(5):
    Image.fromarray(normal_images[i]).save(os.path.join(OUTPUTS_DIR, f'normal_example_{i}.png'))
    Image.fromarray(anomaly_images[i]).save(os.path.join(OUTPUTS_DIR, f'anomaly_example_{i}.png'))

print("Saved!")
