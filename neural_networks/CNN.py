import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from neural_networks.utils import generate_dataset
import os

# --- Data ---
if os.path.exists('K_tensor.pt') and os.path.exists('P_tensor.pt'):
    print("Loading existing tensors...")
    K_tensor = torch.load('K_tensor.pt')
    P_tensor = torch.load('P_tensor.pt')
else:
    print("Generating new tensors...")
    K_data, P_data = generate_dataset()

    # Normalize
    K_data = (K_data - K_data.mean()) / K_data.std()
    P_data = (P_data - P_data.mean()) / P_data.std()

    K_tensor = torch.tensor(K_data, dtype=torch.float32).unsqueeze(1)
    P_tensor = torch.tensor(P_data, dtype=torch.float32).unsqueeze(1)

    torch.save(K_tensor, 'K_tensor.pt')
    torch.save(P_tensor, 'P_tensor.pt')

n_samples = len(K_tensor)
num_epochs = 2000

# Train-test split
split = int(0.8 * n_samples)
train_loader = DataLoader(
    TensorDataset(K_tensor[:split], P_tensor[:split]),
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(K_tensor[split:], P_tensor[split:]),
    batch_size=32,
    shuffle=False
)


# --- Define CNN ---
class DarcyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


# --- Training setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DarcyCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# --- Training loop ---
for epoch in trange(num_epochs):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if epoch % 50 == 0: print(f"[Epoch {epoch + 1}] Training loss: {train_loss / len(train_loader):.4f}")

# --- Evaluation & visualization ---
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

# Stack all predictions and targets
all_preds = torch.cat(all_preds, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()

# Flatten for global MSE/MAE
mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())

print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")

# Visualize a few predictions
n_show = 3
x = x.cpu()
y = y.cpu()

fig, axes = plt.subplots(n_show, 4, figsize=(10, 4 * n_show))
for i in range(n_show):
    axes[i, 0].imshow(x[i, 0], cmap='viridis')
    axes[i, 0].set_title("Input K")
    axes[i, 1].imshow(y[i, 0], cmap='jet')
    axes[i, 1].set_title("Ground Truth P")
    axes[i, 2].imshow(all_preds[i, 0], cmap='jet')
    axes[i, 2].set_title("Predicted P")
    error = np.abs(y[i, 0] - all_preds[i, 0])
    im = axes[i, 3].imshow(error, cmap='coolwarm')
    axes[i, 3].set_title("Absolute Error")
    axes[i, 3].axis('off')
    cbar = fig.colorbar(im, ax=axes[i, 3], shrink=0.4)
    cbar.set_label('Abs Error')
    for ax in axes[i, :3]:
        ax.axis('off')

plt.tight_layout()
plt.savefig('cnn.jpg')
plt.show()