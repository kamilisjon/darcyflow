import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

def homogenous_k(Nx, Ny):
    return np.ones((Ny, Nx))

def layered_k(Nx, Ny, *, k_top=1e-2, k_bottom=1.0, ratio=0.5):
    split = int(Ny * ratio)
    K = np.ones((Ny, Nx)) * k_top
    K[split:, :] = k_bottom
    return K

def channel_k(Nx, Ny, *, k_high=5.0, k_low=1e-2, width=4, xpos=None):
    K = np.ones((Ny, Nx)) * k_low
    if xpos is None:
        xpos = Nx // 2
    l = max(0, xpos - width // 2)
    r = min(Nx, xpos + width // 2)
    K[:, l:r] = k_high
    return K


def checkerboard_k(Nx, Ny, *, block=5, k_high=1.0, k_low=1e-3):
    K = np.empty((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            K[i, j] = k_high if ((i // block + j // block) & 1) == 0 else k_low
    return K


def gaussian_random_k(Nx, Ny, *, mean_logK=-5, sigma=1.0, corr_len=4, seed=0):
    rng = np.random.default_rng(seed)
    field = rng.normal(mean_logK, sigma, size=(Ny, Nx))
    field = gaussian_filter(field, corr_len)
    return 10.0 ** field

def exp_uniform_k(Nx, Ny, *, scale=5.0, size=3, mode='reflect', seed=0):
    U = np.random.rand(Nx,Ny)
    U = uniform_filter(U, size=size, mode=mode)
    U = uniform_filter(U, size=size, mode=mode)   # second pass = extra smoothing
    return np.exp(scale * U)


def radial_inclusion_k(Nx, Ny, *, r=10, k_in=20.0, k_out=1e-5, center=None):
    if center is None:
        center = (Ny // 2, Nx // 2)
    Y, X = np.indices((Ny, Nx))
    dist = np.hypot(Y - center[0], X - center[1])
    K = np.full((Ny, Nx), k_out)
    K[dist <= r] = k_in
    return K

def calculate_flow(Nx, Ny, P, K):
    ny, nx = P.shape
    dx = Nx / (nx - 1)
    dy = Ny / (ny - 1)

    # Compute pressure gradients using central differences
    dPdx = np.zeros_like(P)
    dPdy = np.zeros_like(P)
    # Interior points: central difference
    dPdx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2*dx)
    dPdy[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2*dy)
    # Boundaries: one-sided difference
    dPdx[:, 0] = (P[:, 1] - P[:, 0]) / dx
    dPdx[:, -1] = (P[:, -1] - P[:, -2]) / dx
    dPdy[0, :] = (P[1, :] - P[0, :]) / dy
    dPdy[-1, :] = (P[-1, :] - P[-2, :]) / dy
    # Darcy velocity components: u = -K * grad P
    u_x = -K * dPdx
    u_y = -K * dPdy

    return u_x, u_y