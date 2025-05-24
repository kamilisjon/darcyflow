import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def compute_inv(k: np.ndarray) -> np.ndarray: return 1.0 / k
def idx(i: int, j: int, Nx: int) -> int: return i + j * Nx

def TPFA(Nx:int, Ny:int, permiability_field:np.ndarray, pressure_bc:dict[int, float]) -> np.ndarray:
    # Inverse permeabilities
    perm_inv = compute_inv(permiability_field)

    # Transmissibilities - T
    TX = np.zeros((Nx+1, Ny))
    TY = np.zeros((Nx, Ny+1))
    hx, hy = 1/Nx, 1/Ny
    tx, ty = 2*hy/hx, 2*hx/hy
    for i in range(1, Nx):
        for j in range(Ny):
            TX[i,j] = tx / (perm_inv[0,i-1,j] + perm_inv[0,i,j])
    for i in range(Nx):
        for j in range(1, Ny):
            TY[i,j] = ty / (perm_inv[1,i,j-1] + perm_inv[1,i,j])

    # Assemble pressure matrix - A
    rows, cols, data = [], [], []
    for j in range(Ny):
        for i in range(Nx):
            m = idx(i,j,Nx)
            diag = 0.0
            if i>0:
                rows.append(m)
                cols.append(m-1)
                t = TX[i,j]
                data.append(-t)
                diag+=t
            if i<Nx-1:
                rows.append(m)
                cols.append(m+1)
                t = TX[i+1,j]
                data.append(-t)
                diag+=t
            if j>0:
                rows.append(m)
                cols.append(idx(i,j-1,Nx))
                t = TY[i,j]
                data.append(-t)
                diag+=t
            if j<Ny-1:
                rows.append(m)
                cols.append(idx(i,j+1,Nx))
                t = TY[i,j+1]
                data.append(-t)
                diag+=t
            rows.append(m)
            cols.append(m)
            data.append(diag)
    A = lil_matrix((Nx*Ny, Nx*Ny))
    A[rows, cols] = data

    # impose Dirichlet BCs
    q = np.zeros(Nx*Ny)
    for node, pressure in pressure_bc.items():
        A[node,:] = 0
        A[node,node] = 1.0
        q[node] = pressure

    return spla.spsolve(A.tocsr(), q).reshape((Nx,Ny))

if __name__=='__main__':
    np.random.seed(0)
    homogeneous = False
    Nx = 40
    Ny = 40
    pressure_bc: dict[int, float] = {0: 300.0, Nx*Ny-1: -300.0}

    if homogeneous:
        perm = np.ones((2,Nx,Ny))
    else:
        perm = np.exp(5*uniform_filter(uniform_filter(np.random.randn(2,Nx,Ny), size=3, mode='reflect'), size=3, mode='reflect'))

    pressure_tpfa = TPFA(Nx, Ny, perm, pressure_bc)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    im = ax1.imshow(np.log10(perm[0]), origin='lower', aspect='equal')
    ax1.set_title('log10(K)'); fig.colorbar(im, ax=ax1)

    cf = ax2.contourf(pressure_tpfa, levels=20)
    ax2.set_title('Pressure (TPFA)'); fig.colorbar(cf, ax=ax2)
    plt.tight_layout()
    plt.show()