import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def idx(i: int, j: int, Nx: int) -> int: return i + j * Nx

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
        # These A updates were given by ChatGPT. These make the result look good. But why we need them? Seems it should be enough to have source/sink pressures with q.
        A[node,:] = 0
        A[node,node] = 1.0
        q[node] = pressure

    return spla.spsolve(A.tocsr(), q).reshape((Nx,Ny))

def compute_local_T(Kcells, hx, hy):
    """
    Compute the 4×4 local transmissibility matrix T_loc for one interaction region.
    Kcells = [K_SW, K_SE, K_NE, K_NW], each a 2×2 permeability tensor.
    T_loc[f, c] gives the contribution of P_cell[c] to q_subface[f].
    Subfaces f = 0:south, 1:east, 2:north, 3:west; cells c = [SW,SE,NE,NW].
    """
    KSW, KSE, KNE, KNW = Kcells
    # Build A (8×8) and B (8×4) for continuity eqns
    A = np.zeros((8,8))
    B = np.zeros((8,4))
    # 1) south pressure continuity
    A[0, 0] = hx/2;   A[0, 2] = hx/2
    B[0, 0] = -1;     B[0, 1] = +1
    # 2) south flux continuity (n=[0,-1])
    A[1, 0] =  KSW[1,0];  A[1,1] =  KSW[1,1]
    A[1, 2] = -KSE[1,0];  A[1,3] = -KSE[1,1]
    # 3) east pressure continuity
    A[2, 3] = hy/2;   A[2, 5] = hy/2
    B[2, 1] = -1;     B[2, 2] = +1
    # 4) east flux continuity (n=[1,0])
    A[3, 2] =  KSE[0,0];  A[3,3] =  KSE[0,1]
    A[3, 4] = -KNE[0,0];  A[3,5] = -KNE[0,1]
    # 5) north pressure continuity
    A[4, 6] = hx/2;   A[4, 4] = hx/2
    B[4, 2] = +1;     B[4, 3] = -1
    # 6) north flux continuity (n=[0,1])
    A[5, 6] =  KNW[0,1];  A[5,7] =  KNW[1,1]
    A[5, 4] = -KNE[0,1];  A[5,5] = -KNE[1,1]
    # 7) west pressure continuity
    A[6, 1] = hy/2;   A[6, 7] = hy/2
    B[6, 0] = -1;     B[6, 3] = +1
    # 8) west flux continuity (n=[-1,0])
    A[7, 0] =  KSW[0,0];  A[7,1] =  KSW[0,1]
    A[7, 6] = -KNW[0,0];  A[7,7] = -KNW[0,1]

    invA = np.linalg.inv(A)

    # Build M (4×8) to turn gradients → fluxes on each subface
    M = np.zeros((4,8))
    # south: use SW side, n_south=[0,-1], length= hx
    M[0, 0:2] = -np.array([KSW[1,0], KSW[1,1]]) * hx
    # east: use SE side, n_east=[1,0], length= hy
    M[1, 2:4] =  np.array([KSE[0,0], KSE[0,1]]) * hy
    # north: use NW side, n_north=[0,1], length= hx
    M[2, 6:8] =  np.array([KNW[0,1], KNW[1,1]]) * hx
    # west: use SW side, n_west=[-1,0], length= hy
    M[3, 0:2] = -np.array([KSW[0,0], KSW[0,1]]) * hy

    # Local T_loc: subface flux = T_loc @ P_loc
    T_loc = M.dot(invA).dot(B)
    return T_loc

def MPFA_O(Nx, Ny, K_field, pressure_bc):
    """
    Solve -∇·(K ∇p)=0 on [0,1]^2 with Dirichlet BCs.
    K_field[i,j] is a 2×2 tensor in cell (i,j).
    pressure_bc is { global_index: p_value }.
    Returns p as an (Nx,Ny) array.
    """
    hx, hy = 1.0/Nx, 1.0/Ny
    rows, cols, data = [], [], []

    # Loop over all interior nodes i=1..Nx-1, j=1..Ny-1
    for i in range(1, Nx):
        for j in range(1, Ny):
            # gather the 4 cell tensors around node (i,j)
            KSW = K_field[i-1, j-1]
            KSE = K_field[i  , j-1]
            KNE = K_field[i  , j  ]
            KNW = K_field[i-1, j  ]
            Tloc = compute_local_T([KSW,KSE,KNE,KNW], hx, hy)

            # global cell‐indices in the order [SW,SE,NE,NW]
            ids = [
                idx(i-1, j-1, Nx),
                idx(i  , j-1, Nx),
                idx(i  , j  , Nx),
                idx(i-1, j  , Nx),
            ]

            # for each of the 4 subfaces f, add its Tloc[f,*] contributions
            # f=0: SW↔SE, f=1: SE↔NE, f=2: NW↔NE, f=3: SW↔NW
            neigh = [(0,1), (1,2), (3,2), (0,3)]
            for f, (a_idx,b_idx) in enumerate(neigh):
                Tval = 0.5*( Tloc[f, b_idx] - Tloc[f, a_idx] )
                mi = ids[a_idx]; mj = ids[b_idx]
                # assemble 2×2 block [ +T  -T; -T  +T ]
                for ii, jj, vv in [
                    (mi, mi, +Tval),
                    (mj, mj, +Tval),
                    (mi, mj, -Tval),
                    (mj, mi, -Tval),
                ]:
                    rows.append(ii); cols.append(jj); data.append(vv)

    # build global sparse A
    A = coo_matrix((data,(rows,cols)),shape=(Nx*Ny, Nx*Ny)).tocsr()

    # RHS and Dirichlet BCs
    q = np.zeros(Nx*Ny)
    for node, pval in pressure_bc.items():
        A[node,:] = 0
        A[node,node] = 1.0
        q[node] = pval

    p = spla.spsolve(A, q)
    return p.reshape((Nx,Ny))

if __name__=='__main__':
    np.random.seed(0)
    homogeneous = False
    Nx = 40
    Ny = 40
    pressure_bc: dict[int, float] = {0: 300.0, 512: 200, 1212: -200, Nx*Ny-1: -300.0}

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