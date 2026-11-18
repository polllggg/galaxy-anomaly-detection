import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_encode = nn.Linear(128 * 32 * 32, 64)
        self.fc_decode = nn.Linear(64, 128 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc_encode(x))
        x = torch.relu(self.fc_decode(x))
        x = x.view(x.size(0), 128, 32, 32)
        x = self.decoder(x)
        return x


device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = Autoencoder()
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'Galaxy_Autoencoder.pth'), map_location=device))
model = model.to(device)
model.eval()

normal_images = np.load(os.path.join(DATA_DIR, 'normal_images.npy'))
anomaly_images = np.load(os.path.join(DATA_DIR, 'anomaly_images.npy'))


def calc_errors(images_array, model, device, batch_size=64):
    imgs = torch.FloatTensor(np.transpose(images_array / 255.0, (0, 3, 1, 2)))
    loader = DataLoader(TensorDataset(imgs), batch_size=batch_size)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            mse = torch.mean((batch - output) ** 2, dim=(1, 2, 3))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


print("Calculating normal errors...")
normal_errors = calc_errors(normal_images, model, device)
print("Calculating anomaly errors...")
anomaly_errors = calc_errors(anomaly_images, model, device)

np.save(os.path.join(DATA_DIR, 'ae_normal_scores.npy'), normal_errors)
np.save(os.path.join(DATA_DIR, 'ae_anomaly_scores.npy'), anomaly_errors)
print("Saved!")
