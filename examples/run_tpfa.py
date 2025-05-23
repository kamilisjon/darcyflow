import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def TPFA(grid, K, q, p_bc):
    Nx, Ny = grid['Nx'], grid['Ny']
    hx, hy = grid['hx'], grid['hy']
    N = Nx * Ny

    # Inverse permeabilities
    L = 1.0 / K

    # Transmissibilities
    TX = np.zeros((Nx+1, Ny))
    TY = np.zeros((Nx, Ny+1))
    tx, ty = 2*hy/hx, 2*hx/hy
    for i in range(1, Nx):
        for j in range(Ny):
            TX[i,j] = tx / (L[0,i-1,j] + L[0,i,j])
    for i in range(Nx):
        for j in range(1, Ny):
            TY[i,j] = ty / (L[1,i,j-1] + L[1,i,j])

    # Assemble A
    def idx(i,j): return i + j*Nx
    rows, cols, data = [], [], []
    for j in range(Ny):
        for i in range(Nx):
            m = idx(i,j)
            diag = 0.0
            if i>0:
                t = TX[i,j]; rows.append(m); cols.append(m-1); data.append(-t); diag+=t
            if i<Nx-1:
                t = TX[i+1,j]; rows.append(m); cols.append(m+1); data.append(-t); diag+=t
            if j>0:
                t = TY[i,j]; rows.append(m); cols.append(idx(i,j-1)); data.append(-t); diag+=t
            if j<Ny-1:
                t = TY[i,j+1]; rows.append(m); cols.append(idx(i,j+1)); data.append(-t); diag+=t
            rows.append(m); cols.append(m); data.append(diag)

    A = sp.csr_matrix((data,(rows,cols)), shape=(N,N))
    # impose Dirichlet BCs
    for node, pval in p_bc.items():
        A[node,:] = 0
        A[node,node] = 1.0
        q[node] = pval

    u = spla.spsolve(A, q)
    return u.reshape((Nx,Ny))

if __name__=='__main__':
    np.random.seed(0)
    homogeneous = True

    grid = {'Nx':40, 'Ny':40, 'hx':1/40, 'hy':1/40}

    if homogeneous:
        K = np.ones((2,40,40))
    else:
        rnd = np.random.randn(2,40,40)
        S = uniform_filter(rnd, size=3, mode='reflect')
        K = np.exp(5*uniform_filter(S, size=3, mode='reflect'))

    N = 40*40
    q = np.zeros(N)
    p_bc = {0: +300, N-1: -300}

    P_tpfa = TPFA(grid, K, q.copy(), p_bc)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    im = ax1.imshow(np.log10(K[0]), origin='lower', aspect='equal')
    ax1.set_title('log10(K)'); fig.colorbar(im, ax=ax1)

    cf = ax2.contourf(P_tpfa, levels=20)
    ax2.set_title('Pressure (TPFA)'); fig.colorbar(cf, ax=ax2)
    plt.tight_layout()
    plt.show()