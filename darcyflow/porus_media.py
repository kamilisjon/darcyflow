import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from darcyflow.solver import gidx

np.random.seed(0)

class DarcyDomain:
    def __init__(self, **kwargs):
        self.Nx = kwargs.get('Nx', 40)
        self.Ny = kwargs.get('Ny', 40)
        self.pressure_bc = kwargs.get('pressure_bc', {gidx(0, 0, self.Nx): 300.0,
                                                      gidx(self.Nx - 1, self.Ny - 1, self.Nx): -300.0})
    
    def homogenous_k(self):
        return np.ones((self.Ny, self.Nx))

    def layered_k(self, *, k_top=1e-2, k_bottom=1.0, ratio=0.5):
        split = int(self.Ny * ratio)
        K = np.ones((self.Ny, self.Nx)) * k_top
        K[split:, :] = k_bottom
        return K

    def channel_k(self, *, k_high=5.0, k_low=1e-2, width=4, xpos=None):
        K = np.ones((self.Ny, self.Nx)) * k_low
        if xpos is None:
            xpos = self.Nx // 2
        l = max(0, xpos - width // 2)
        r = min(self.Nx, xpos + width // 2)
        K[:, l:r] = k_high
        return K

    def checkerboard_k(self, *, block=5, k_high=1.0, k_low=1e-3):
        K = np.empty((self.Ny, self.Nx))
        for i in range(self.Ny):
            for j in range(self.Nx):
                K[i, j] = k_high if ((i // block + j // block) & 1) == 0 else k_low
        return K

    def gaussian_random_k(self, *, mean_logK=-5, sigma=1.0, corr_len=4):
        rng = np.random.default_rng()
        field = rng.normal(mean_logK, sigma, size=(self.Ny, self.Nx))
        field = gaussian_filter(field, corr_len)
        return 10.0 ** field

    def exp_uniform_k(self, *, scale=5.0, size=3, mode='reflect'):
        U = np.random.rand(self.Nx, self.Ny)
        U = uniform_filter(U, size=size, mode=mode)
        U = uniform_filter(U, size=size, mode=mode)   # second pass = extra smoothing
        return np.exp(scale * U)

    def radial_inclusion_k(self, *, r=10, k_in=20.0, k_out=1e-5, center=None):
        if center is None:
            center = (self.Ny // 2, self.Nx // 2)
        Y, X = np.indices((self.Ny, self.Nx))
        dist = np.hypot(Y - center[0], X - center[1])
        K = np.full((self.Ny, self.Nx), k_out)
        K[dist <= r] = k_in
        return K

def calculate_flow(domain, P, K):
    ny, nx = P.shape
    dx = domain.Nx / (nx - 1)
    dy = domain.Ny / (ny - 1)

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