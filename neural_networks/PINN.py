"""

Hand-crafted PINN implementation
Currently the results are non-existing for heterogeneous permeability K (and for homogeneous)
There is a possibility to implement PINN using DeepXDE (https://deepxde.readthedocs.io/en/latest/index.html), but still had issues

CURRENTLY NOT WORKING GOOD

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
import os
import numpy as np
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
grid_size = 40
source, sink = 300, -300
number_of_collocation = 1000000
num_of_boundaries = int(number_of_collocation / 20)


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)


# def K_func(xy):
#     x, y = xy[:, 0:1], xy[:, 1:2]
#     Kx = 1 + 0.5 * torch.sin(2 * torch.pi * x)
#     Ky = 1 + 0.5 * torch.cos(2 * torch.pi * y)
#     return torch.cat([Kx, Ky], dim=1)

# def K_func(xy):
#     x, y = xy[:, 0:1], xy[:, 1:2]
#     K = 1 + 0.5 * torch.sin(2 * torch.pi * x / 40) * torch.cos(2 * torch.pi * y / 40)
#     return K

def bilinear_interpolate(Nx, Ny, xy, K_values):
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = 1 / (Nx - 1), 1 / (Ny - 1)
    ix, iy = x / dx, y / dy
    ix0, iy0 = torch.floor(ix).long().clamp(0, Nx - 2), torch.floor(iy).long().clamp(0, Ny - 2)
    ix1, iy1 = ix0 + 1, iy0 + 1
    wx, wy = (ix - ix0.float()).unsqueeze(1), (iy - iy0.float()).unsqueeze(1)
    f00 = K_values[ix0, iy0].unsqueeze(1)
    f10 = K_values[ix1, iy0].unsqueeze(1)
    f01 = K_values[ix0, iy1].unsqueeze(1)
    f11 = K_values[ix1, iy1].unsqueeze(1)
    Kxy = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11
    return Kxy


def q_func(xy):
    return torch.zeros((xy.shape[0], 1), device=device)


def pde_residual(xy, K_at_xy, model):
    xy.requires_grad_(True)
    P = model(xy)
    grads = torch.autograd.grad(P, xy, torch.ones_like(P), create_graph=True)[0]
    Px, Py = grads[:, 0:1], grads[:, 1:2]
    Pxx = torch.autograd.grad(Px, xy, torch.ones_like(Px), create_graph=True)[0][:, 0:1]
    Pyy = torch.autograd.grad(Py, xy, torch.ones_like(Py), create_graph=True)[0][:, 1:2]
    div = K_at_xy * (Pxx + Pyy)  # simplified PDE residual assuming constant K at point
    return div + q_func(xy)


def neumann_residual(xy, model, normals):
    xy.requires_grad_(True)
    P = model(xy)
    grads = torch.autograd.grad(P, xy, torch.ones_like(P), create_graph=True)[0]
    return torch.sum(grads * normals, dim=1, keepdim=True)


def loss_fn(model, collocation_points, xy_d, P_d, xy_n, n_vecs, K_func, q_func):
    f_pred = pde_residual(collocation_points, model, K_func, q_func)
    mse_pde = torch.mean(f_pred ** 2)

    # Dirichlet BC
    P_bc_pred = model(xy_d)
    mse_dirichlet = torch.mean((P_bc_pred - P_d) ** 2)

    # Neumann BC
    n_flux = neumann_residual(xy_n, model, n_vecs)
    mse_neumann = torch.mean(n_flux ** 2)

    return mse_pde + mse_dirichlet + mse_neumann


def generate_data():
    xy_f = torch.rand((number_of_collocation, 2), device=device)

    xy_d = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    P_d = torch.tensor([[float(source)], [float(sink)]], device=device)

    N = num_of_boundaries
    x = torch.linspace(0, 1, N, device=device).unsqueeze(1)
    y0 = torch.zeros_like(x)
    y1 = torch.ones_like(x)

    xy_n = torch.cat([
        torch.cat([x, y0], 1),  # bottom
        torch.cat([x, y1], 1),  # top
        torch.cat([y0, x], 1),  # left
        torch.cat([y1, x], 1)  # right
    ])
    normals = torch.cat([
        torch.tensor([[0.0, -1.0]], device=device).repeat(N, 1),
        torch.tensor([[0.0, 1.0]], device=device).repeat(N, 1),
        torch.tensor([[-1.0, 0.0]], device=device).repeat(N, 1),
        torch.tensor([[1.0, 0.0]], device=device).repeat(N, 1)
    ])
    return xy_f, xy_d, P_d, xy_n, normals


def train(layers=[2, 64, 64, 64, 1], epochs=2000, smooth_K=False, smooth_sigma=1, N=(100, 100), mu=1.0, sigma=0.5,
          use_lbfgs=False, lr=1e-3):
    # Permeability
    Nx, Ny = N
    K_values_np = mu + sigma * np.random.rand(Nx, Ny)
    if smooth_K:
        K_values_np = gaussian_filter(K_values_np, sigma=smooth_sigma)
    K_values = torch.tensor(K_values_np, dtype=torch.float32, device=device)

    model = PINN(layers).to(device)
    xy_f, xy_d, P_d, xy_n, normals = generate_data()
    K_train = bilinear_interpolate(Nx, Ny, xy_f, K_values).squeeze(1).unsqueeze(1)

    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=epochs, history_size=50,
                            line_search_fn="strong_wolfe") if use_lbfgs else optim.Adam(model.parameters(), lr=lr)

    def closure():
        optimizer.zero_grad()
        loss_pde = torch.mean(pde_residual(xy_f, K_train, model) ** 2)
        loss_d = torch.mean((model(xy_d) - P_d) ** 2)
        loss_n = torch.mean(neumann_residual(xy_n, model, normals) ** 2)
        loss = loss_pde + loss_d + loss_n
        loss.backward()
        return loss

    if use_lbfgs:
        optimizer.step(closure)
        print("Final Loss:", closure().item())
    else:
        for epoch in trange(epochs):
            loss = closure()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    return model, xy_f, K_train, K_values_np


def plot_all(model, xy_f, K, K_raw, fig_name):
    model.eval()
    N = 100
    x = torch.linspace(0, 1.0, N)
    y = torch.linspace(0, 1.0, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1).to(device)
    xy.requires_grad_(True)

    # Compute Pressure
    P = model(xy)
    P_np = P.detach().cpu().numpy().reshape(N, N)

    # Compute ∇P
    grads = torch.autograd.grad(P, xy, torch.ones_like(P), create_graph=True)[0]
    Px, Py = grads[:, 0:1], grads[:, 1:2]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    x = np.linspace(0, grid_size, N)
    y = np.linspace(0, grid_size, N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Pressure Field
    im0 = axs[0].contourf(X, Y, P_np, levels=50, cmap='coolwarm')
    axs[0].set_title("Pressure Field P(x, y)")
    axs[0].set_xlabel("x");
    axs[0].set_ylabel("y")
    fig.colorbar(im0, ax=axs[0])

    # Permeability Field
    xy_np = xy_f.cpu().detach().numpy() * grid_size
    K_np = K.cpu().detach().numpy().squeeze()
    sc = axs[1].scatter(xy_np[:, 0], xy_np[:, 1], c=K_np, cmap='viridis', marker='o')
    axs[1].set_title('Permeability K at Collocation Points')
    axs[1].set_xlabel("x");
    axs[1].set_ylabel("y")
    fig.colorbar(sc, ax=axs[1])

    Nx, Ny = K_raw.shape

    r0 = axs[2].contourf(X, Y, K_raw, levels=50, cmap='viridis')
    fig.colorbar(r0, ax=axs[2], label='Permeability K')
    axs[2].set_title("Raw Permeability Field")

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

if __name__=='__main__':
    model, xy, K, K_raw = train(smooth_K=True, smooth_sigma=5, epochs=10000, use_lbfgs=False, lr=0.001)
    save_dir = 'raw_pinn_results'
    os.makedirs(save_dir, exist_ok=True)
    plot_all(model, xy, K, K_raw, os.path.join(save_dir, 'runAdam100002.jpg'))