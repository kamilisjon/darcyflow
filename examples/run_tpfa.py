import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def TPFA(grid:dict[str, int], permiability_filed:NDArray, pressure_bc:dict[int, int]) -> NDArray:
    # Inverse permeabilities
    perm_inv = 1.0 / permiability_filed

    # Transmissibilities
    TX = np.zeros((grid['Nx']+1, grid['Ny']))
    TY = np.zeros((grid['Nx'], grid['Ny']+1))
    cell_size: dict[str, float] = {'hx':1/grid['Nx'], 'hy':1/grid['Ny']}
    tx, ty = 2*cell_size['hy']/cell_size['hx'], 2*cell_size['hx']/cell_size['hy']

    for i in range(1, grid['Nx']):
        for j in range(grid['Ny']):
            TX[i,j] = tx / (perm_inv[0,i-1,j] + perm_inv[0,i,j])
    for i in range(grid['Nx']):
        for j in range(1, grid['Ny']):
            TY[i,j] = ty / (perm_inv[1,i,j-1] + perm_inv[1,i,j])

    # Assemble A
    def idx(i:int, j:int) -> int: return i + j*grid['Nx']
    rows, cols, data = [], [], []
    for j in range(grid['Ny']):
        for i in range(grid['Nx']):
            m = idx(i,j)
            diag = 0.0
            if i>0:
                t = TX[i,j]; rows.append(m); cols.append(m-1); data.append(-t); diag+=t
            if i<grid['Nx']-1:
                t = TX[i+1,j]; rows.append(m); cols.append(m+1); data.append(-t); diag+=t
            if j>0:
                t = TY[i,j]; rows.append(m); cols.append(idx(i,j-1)); data.append(-t); diag+=t
            if j<grid['Ny']-1:
                t = TY[i,j+1]; rows.append(m); cols.append(idx(i,j+1)); data.append(-t); diag+=t
            rows.append(m); cols.append(m); data.append(diag)

    area = grid['Nx'] * grid['Ny']
    A = lil_matrix((area, area))
    A[rows, cols] = data

    # impose Dirichlet BCs
    q = np.zeros(area)
    for node, pval in pressure_bc.items():
        A[node,:] = 0
        A[node,node] = 1.0
        q[node] = pval

    u = spla.spsolve(A.tocsr(), q)
    return u.reshape((grid['Nx'],grid['Ny']))

if __name__=='__main__':
    np.random.seed(0)
    homogeneous = False
    grid: dict[str, int] = {'Nx':40, 'Ny':40}

    if homogeneous:
        perm = np.ones((2,grid['Nx'],grid['Ny']))
    else:
        perm = np.exp(5*uniform_filter(uniform_filter(np.random.randn(2,grid['Nx'],grid['Ny']), size=3, mode='reflect'), size=3, mode='reflect'))

    pressure_tpfa = TPFA(grid, perm, {0: +300, grid['Nx']*grid['Ny']-1: -300})

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    im = ax1.imshow(np.log10(perm[0]), origin='lower', aspect='equal')
    ax1.set_title('log10(K)'); fig.colorbar(im, ax=ax1)

    cf = ax2.contourf(pressure_tpfa, levels=20)
    ax2.set_title('Pressure (TPFA)'); fig.colorbar(cf, ax=ax2)
    plt.tight_layout()
    plt.show()