import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# Good reference of TPFA implementation, however the calculations order differs: 
# https://www.duo.uio.no/bitstream/handle/10852/9721/Lunde.pdf?sequence=1
# What is an orthogonal grid?:
# https://www.researchgate.net/publication/363336649_Isovists_in_a_Grid_benefits_and_limitations
# What are finite volume methods?

# Here is the MatLab reference, that Mayur Pal gave:
# https://andreas.folk.ntnu.no/papers/ResSimMatlab.pdf
# What is Single-Phase flow? Only one type of fluid flowing through porus medium. Source!?!?!
# What is uniform cartesian grid?
# What is the difference between finite-difference and finite-volume methods? TPFA is finite-volume method
# Difference between porousity and permeability:
# https://soil.evs.buffalo.edu/images/thumb/9/98/9FDE2B38-94FC-495B-A12B-F57898660EFC.png/500px-9FDE2B38-94FC-495B-A12B-F57898660EFC.png
# https://soil.evs.buffalo.edu/index.php/File:9FDE2B38-94FC-495B-A12B-F57898660EFC.png#filehistory
# Difference between speed and velocity
# https://energywavetheory.com/forces/velocity/
# "Unless stated otherwise we shall follow common practice and use no-flow boundary conditions." -> "isolated flow system where no water can enter or exit the reservoir"
# Authors use "uniform Cartesian grid" for simulations -> every grid cell has same dimensions

def TPFA(Nx:int, Ny:int, permiability_field:np.ndarray, pressure_bc:dict[int, float]) -> np.ndarray:
    # Source is chapter 3 (Incompressible Single-Phase Flow) of: https://andreas.folk.ntnu.no/papers/ResSimMatlab.pdf 

    # "in most reservoir simulation models, the permeability K is cell-wise constant, and hence not well-defined at the interfaces"
    #           Interface - line at which two grid cells are joined.
    # This means that we have to approximate permeability on the interfaces
    # "In the TPFA method this is done by taking a distance-weighted harmonic average of the respective directional cell permeabilities"
    # Exact implementation logic is taken from the source.
    TX = np.zeros((Nx+1, Ny))
    TY = np.zeros((Nx, Ny+1))
    cell_dim_x, cell_dim_y = 1/Nx, 1/Ny  # Cell size is an inverse of the cell count here to get [0, 1] size grid. Why so?
    # Not sure how did they got this expression. Formula is different. If it would be clear what ∆xi, ∆xj and 2|γij| mean, then the tij formula could be used here
    perm_inv = 1.0 / permiability_field
    TX[1:-1, :] = (2 * cell_dim_y / cell_dim_x) / (perm_inv[:-1, :] + perm_inv[1:, :])
    TY[:, 1:-1] = (2 * cell_dim_x / cell_dim_y) / (perm_inv[:, :-1] + perm_inv[:, 1:])

    # Assemble pressure matrix - A
    # Same logic as in MatLab code. Could be rewriten to be more gracefull
    def idx(i: int, j: int, Nx: int) -> int: return i + j * Nx
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

    # impose BCs
    q = np.zeros(Nx*Ny)
    for node, pressure in pressure_bc.items():
        # ChatGPT gave this solution. Did not find other source. Also fixes the MatLab code of the author
        A[node,:] = 0       # zero out entire row
        A[node,node] = 1.0  # set diagonal to 1
        q[node] = pressure  # enforce pressure on specific point

    return spla.spsolve(A.tocsr(), q).reshape((Nx,Ny))

if __name__=='__main__':
    np.random.seed(0)
    homogeneous = False
    Nx = 40
    Ny = 40
    pressure_bc: dict[int, float] = {0: 300.0, Nx*Ny-1: -300.0}

    if homogeneous:
        porus_media = np.ones((Nx,Ny))
    else:
        porus_media = np.exp(5*uniform_filter(uniform_filter(np.random.rand(Nx,Ny), size=3, mode='reflect'), size=3, mode='reflect'))

    pressure_field = TPFA(Nx, Ny, porus_media, pressure_bc)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.set_title('Porus media')
    ax1.set_xlim(0, Nx)
    ax1.set_ylim(0, Ny)
    fig.colorbar(ax1.imshow(porus_media, extent=(0, Nx, 0, Ny)), ax=ax1)
    ax2.set_title('Pressure field')
    ax2.set_xlim(0, Nx)
    ax2.set_ylim(0, Ny)
    fig.colorbar(ax2.contourf(pressure_field, levels=20, extent=(0, Nx, 0, Ny)), ax=ax2)
    plt.tight_layout()
    plt.show()