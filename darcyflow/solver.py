import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)

# ---------- helpers ----------------------------------------------------------
def gidx(i, j, Nx):                  # (i,j) → global cell index
    return i + j * Nx

def four_cells_around_node(i, j, Nx, Ny):
    c = []
    if i > 0 and j > 0:       c.append(gidx(i - 1, j - 1, Nx))   # SW
    if i < Nx and j > 0:      c.append(gidx(i,     j - 1, Nx))   # SE
    if i < Nx and j < Ny:     c.append(gidx(i,     j,     Nx))   # NE
    if i > 0 and j < Ny:      c.append(gidx(i - 1, j,     Nx))   # NW
    return c

# ---------- local O-method transmissibilities (heterogeneous) ---------------
def harm(k1, k2):
    """Flux-consistent harmonic mean."""
    return 0.0 if k1 + k2 == 0.0 else 2.0 * k1 * k2 / (k1 + k2)


def local_T(Kloc, hx, hy):
    """
    Six transmissibilities for one interaction region with *heterogeneous*
    **isotropic** permeabilities Kloc = [K_SW, K_SE, K_NE, K_NW].
    """
    Th = hy / hx          # geometric prefactors (face area / distance)
    Tv = hx / hy

    kx  = Th * harm(Kloc[0], Kloc[1])   # SW–SE
    ky  = Tv * harm(Kloc[1], Kloc[2])   # SE–NE
    kx2 = Th * harm(Kloc[2], Kloc[3])   # NE–NW
    ky2 = Tv * harm(Kloc[3], Kloc[0])   # NW–SW

    # Cross-face transmissibilities: same closed-form as uniform case but
    # now built from the four neighbouring face coefficients
    kdiag1 = 0.25 * (kx + ky)           # SW–NE
    kdiag2 = 0.25 * (kx2 + ky2)         # SE–NW
    return (kx, ky, kx2, ky2, kdiag1, kdiag2)


# ---------- global assembly (unchanged) --------------------------------------
def assemble_mpfa_o(Nx, Ny, K):
    hx, hy = 1.0 / Nx, 1.0 / Ny
    N = Nx * Ny
    rows, cols, data = [], [], []
    rhs = np.zeros(N)

    for j in range(Ny + 1):
        for i in range(Nx + 1):
            cells = four_cells_around_node(i, j, Nx, Ny)
            if len(cells) != 4:
                continue
            Kloc = K.reshape(-1)[cells]
            Tij = local_T(Kloc, hx, hy)

            links = [(cells[0], cells[1], Tij[0]),
                     (cells[1], cells[2], Tij[1]),
                     (cells[2], cells[3], Tij[2]),
                     (cells[3], cells[0], Tij[3]),
                     (cells[0], cells[2], Tij[4]),
                     (cells[1], cells[3], Tij[5])]

            for a, b, t in links:
                rows += [a, b, a, b]
                cols += [b, a, a, b]
                data += [-t, -t,  t,  t]

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A, rhs

def assemble_tpfa(Nx, Ny, K):
    N = Nx*Ny
    rhs = np.zeros(N)
    TX = np.zeros((Nx+1, Ny))
    TY = np.zeros((Nx, Ny+1))
    cell_dim_x, cell_dim_y = 1/Nx, 1/Ny
    perm_inv = 1.0 / K
    TX[1:-1, :] = (2 * cell_dim_y / cell_dim_x) / (perm_inv[:-1, :] + perm_inv[1:, :])
    TY[:, 1:-1] = (2 * cell_dim_x / cell_dim_y) / (perm_inv[:, :-1] + perm_inv[:, 1:])
    rows, cols, data = [], [], []
    for j in range(Ny):
        for i in range(Nx):
            m = gidx(i,j,Nx)
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
                cols.append(gidx(i,j-1,Nx))
                t = TY[i,j]
                data.append(-t)
                diag+=t
            if j<Ny-1:
                rows.append(m)
                cols.append(gidx(i,j+1,Nx))
                t = TY[i,j+1]
                data.append(-t)
                diag+=t
            rows.append(m)
            cols.append(m)
            data.append(diag)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A, rhs

def assemble_fdm(Nx, Ny, K):
    N = Nx * Ny
    hx, hy = 1.0 / Nx, 1.0 / Ny
    rhs = np.zeros(N)
    rows, cols, data = [], [], []

    def k_avg(k1, k2):
        return 2.0 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0.0

    for j in range(Ny):
        for i in range(Nx):
            idx = gidx(i, j, Nx)
            diag = 0.0
            kij = K[i, j]

            # West neighbor
            if i > 0:
                kw = k_avg(kij, K[i - 1, j])
                rows.append(idx)
                cols.append(gidx(i - 1, j, Nx))
                data.append(-kw / hx**2)
                diag += kw / hx**2

            # East neighbor
            if i < Nx - 1:
                ke = k_avg(kij, K[i + 1, j])
                rows.append(idx)
                cols.append(gidx(i + 1, j, Nx))
                data.append(-ke / hx**2)
                diag += ke / hx**2

            # South neighbor
            if j > 0:
                ks = k_avg(kij, K[i, j - 1])
                rows.append(idx)
                cols.append(gidx(i, j - 1, Nx))
                data.append(-ks / hy**2)
                diag += ks / hy**2

            # North neighbor
            if j < Ny - 1:
                kn = k_avg(kij, K[i, j + 1])
                rows.append(idx)
                cols.append(gidx(i, j + 1, Nx))
                data.append(-kn / hy**2)
                diag += kn / hy**2

            # Center
            rows.append(idx)
            cols.append(idx)
            data.append(diag)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A, rhs


# ---------- Dirichlet BC utility --------------------------------------------
def apply_dirichlet(A, rhs, pressure_bc):
    for cell, pval in pressure_bc.items():
        A[cell, :] = 0.0
        A[cell, cell] = 1.0
        rhs[cell] = pval
    return A, rhs


# ---------- top-level solver -------------------------------------------------
def solve(domain, K, method="mpfa_o"):
    """
    Returns Ny×Nx array of cell pressures for an MPFA-O or an TPFA solve
    """
    if method == "mpfa_o":
        A, rhs = assemble_mpfa_o(domain.Nx, domain.Ny, K)
    elif method == "tpfa":
        A, rhs = assemble_tpfa(domain.Nx, domain.Ny, K)
    elif method == "fdm":
        A, rhs = assemble_fdm(domain.Nx, domain.Ny, K)
    else:
        raise NotImplementedError(f"Method {method} is not implemented. Implemented: 'mpfa_o', 'tpfa', 'fdm'")
    A, rhs = apply_dirichlet(A, rhs, domain.pressure_bc)
    p = spla.spsolve(A, rhs)
    return p.reshape(domain.Ny, domain.Nx)