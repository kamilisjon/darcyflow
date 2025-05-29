import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import trange
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from solver import solve
from porus_media import DarcyDomain
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.set_device(device)

# Generate your dataset
n_samples = 1000
domain = DarcyDomain()
solver_method = 'fdm'


def dict_collate_fn(batch):
    x, y = zip(*batch)
    return {'x': torch.stack(x), 'y': torch.stack(y)}


def generate_dataset(n_samples, domain, solver_method):
    K_list, P_list = [], []
    for _ in trange(n_samples, desc='Generating data'):
        K = domain.exp_uniform_k()
        P = solve(domain, K, solver_method)
        K_list.append(K)
        P_list.append(P)
    return np.array(K_list), np.array(P_list)


if K_tensor is None and P_tensor is None:
    K_data, P_data = generate_dataset(n_samples, domain, solver_method)

    # Normalize
    K_data = (K_data - K_data.mean()) / K_data.std()
    P_data = (P_data - P_data.mean()) / P_data.std()

    # Convert to tensors
    K_tensor = torch.tensor(K_data, dtype=torch.float32).unsqueeze(1)
    P_tensor = torch.tensor(P_data, dtype=torch.float32).unsqueeze(1)

# Split into train/test
split = int(0.8 * n_samples)
train_loader = DataLoader(
    TensorDataset(K_tensor[:split], P_tensor[:split]),
    batch_size=32,
    shuffle=True,
    collate_fn=dict_collate_fn
)
test_loader = DataLoader(
    TensorDataset(K_tensor[split:], P_tensor[split:]),
    batch_size=32,
    shuffle=False,
    collate_fn=dict_collate_fn
)

# Wrap test loader into dict to match the Trainer API
test_loaders = {16: test_loader}

# Model
model = FNO(in_channels=1,
            out_channels=1,
            n_modes=(16, 16),
            hidden_channels=32,
            projection_channel_ratio=2)
model = model.to(device)
n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

optimizer = AdamW(model.parameters(),
                  lr=8e-3,
                  weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

# print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

test_samples = test_loaders[16].dataset  # Just your TensorDataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    x, y = test_samples[index]
    x = x.to(device)
    y = y.to(device)

    # Model prediction
    with torch.no_grad():
        out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0].cpu(), cmap='gray')  # permeability field K
    if index == 0:
        ax.set_title('Input x (Permeability)')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y[0].cpu())  # ground truth pressure
    if index == 0:
        ax.set_title('Ground-truth y (Pressure)')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze(0).squeeze(0).cpu())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output, and prediction', y=0.98)
plt.tight_layout()
plt.show()