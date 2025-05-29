import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

from neural_networks.utils import generate_dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.set_device(device)

# Initiate
num_epochs = 2000

def dict_collate_fn(batch):
    x, y = zip(*batch)
    return {'x': torch.stack(x), 'y': torch.stack(y)}

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

# Split into train/test
n_samples = len(K_tensor)
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

optimizer = AdamW(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')

trainer = Trainer(model=model, n_epochs=num_epochs,
                  device=device,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)
start=time.time()
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)
print(f"Training took {time.time()-start:.6f} s")
test_samples = test_loaders[16].dataset  # Just your TensorDataset


model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = model(x)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

all_preds = torch.cat(all_preds, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()
mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
print(f"[Neural Operator] Test MSE: {mse:.6f}, MAE: {mae:.6f}")

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    x, y = test_samples[index]
    x = x.to(device)
    y = y.to(device)

    # Model prediction
    with torch.no_grad():
        out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 4, index*4 + 1)
    ax.imshow(x[0].cpu(), cmap='viridis')  # permeability field K
    if index == 0:
        ax.set_title('Input x (Permeability)')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 4, index*4 + 2)
    ax.imshow(y[0].cpu(), cmap='jet')  # ground truth pressure
    if index == 0:
        ax.set_title('Ground-truth y (Pressure)')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 4, index*4 + 3)
    ax.imshow(out.squeeze(0).squeeze(0).cpu(), cmap='jet')
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 4, index*4 + 4)
    im = ax.imshow(np.abs(y[0].cpu()-out.squeeze(0).squeeze(0).cpu()), cmap='coolwarm')
    if index == 0:
        ax.set_title('Diference')
    plt.xticks([], [])
    plt.yticks([], [])
    cbar = fig.colorbar(im, ax=ax, shrink=0.4)
    cbar.set_label('Absolute Error')

fig.suptitle('Inputs, ground-truth output, prediction and diference', y=0.98)
plt.tight_layout()
plt.savefig("no.jpg")
plt.show()
