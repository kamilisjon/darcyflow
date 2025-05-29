import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import trange
from neuralop.models import FNO
from darcyflow.solver import solve
from darcyflow.porus_media import DarcyDomain

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Dataset generation
def generate_dataset(n_samples, domain, solver_method):
    K_list, P_list = [], []
    for _ in trange(n_samples, desc='Generating data'):
        K = domain.exp_uniform_k()
        P = solve(domain, K, solver_method)
        K_list.append(K)
        P_list.append(P)
    return np.array(K_list), np.array(P_list)

# Parameters
n_samples = 1000
solver_method = 'fdm'
domain = DarcyDomain()

K_data, P_data = generate_dataset(n_samples, domain, solver_method)

# Normalize
K_data = (K_data - K_data.mean()) / K_data.std()
P_data = (P_data - P_data.mean()) / P_data.std()

# Tensors
K_tensor = torch.tensor(K_data, dtype=torch.float32).unsqueeze(1)
P_tensor = torch.tensor(P_data, dtype=torch.float32).unsqueeze(1)

# Dataset and dataloader
dataset = TensorDataset(K_tensor, P_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = FNO(in_channels=1,
            out_channels=1,
            n_modes=(16, 16),
            hidden_channels=32,
            projection_channel_ratio=2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in trange(1000, desc="Training model"):  # Use more epochs for better performance
    model.train()
    total_loss = 0
    for K_batch, P_batch in train_loader:
        K_batch = K_batch.to(device)
        P_batch = P_batch.to(device)
        optimizer.zero_grad()
        pred = model(K_batch)
        loss = loss_fn(pred, P_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")

# Testing on new sample
print("Testing...")
model.eval()
with torch.no_grad():
    test_K = domain.exp_uniform_k()
    test_K_tensor = torch.tensor(test_K, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    pred_P = model(test_K_tensor).squeeze().cpu().numpy()  # move prediction back to CPU for plotting

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_K, cmap='viridis')
plt.title("Input K")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(pred_P, cmap='jet')
plt.title("Predicted P")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(solve(domain, test_K), cmap='jet')
plt.title(f"{solver_method.capitalize()} P (Ground Truth)")
plt.colorbar()

plt.tight_layout()
plt.show()
