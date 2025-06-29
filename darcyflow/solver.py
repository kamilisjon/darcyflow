import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import partial
import warnings

from darcyflow.helpers import gidx, harmonic_mean_2point
warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)

# ---------- helpers ----------------------------------------------------------

class FiniteMethodsSolver:
    def __init__(self, Nx: int, Ny: int, pressure_bc: dict[int, float]):
        self.Nx, self.Ny, self.hx, self.hy, self.N, self.pressure_bc  = Nx, Ny, 1.0/Nx, 1.0/Ny, Nx*Ny, pressure_bc
        self.__methods: dict[str, callable] = {
            "fdm": partial(self.__5_point, hor_func=lambda h: h / self.hx**2, vert_func=lambda h: h / self.hy**2),
            "tpfa": partial(self.__5_point, hor_func=lambda h: (self.hy / self.hx) * h, vert_func=lambda h:(self.hx / self.hy) * h),
            "mpfa_o": self.__mpfa_o
        }

    @staticmethod
    def __four_cells_around_node(i, j, Nx, Ny):
        c = []
        if i > 0 and j > 0:       c.append(gidx(i - 1, j - 1, Nx))   # SW
        if i < Nx and j > 0:      c.append(gidx(i,     j - 1, Nx))   # SE
        if i < Nx and j < Ny:     c.append(gidx(i,     j,     Nx))   # NE
        if i > 0 and j < Ny:      c.append(gidx(i - 1, j,     Nx))   # NW
        return c

    @staticmethod
    def __local_T(Kloc, hx, hy):
        """
        Six transmissibilities for one interaction region with *heterogeneous*
        **isotropic** permeabilities Kloc = [K_SW, K_SE, K_NE, K_NW].
        """
        Th = hy / hx          # geometric prefactors (face area / distance)
        Tv = hx / hy

        kx  = Th * harmonic_mean_2point(Kloc[0], Kloc[1])   # SW–SE
        ky  = Tv * harmonic_mean_2point(Kloc[1], Kloc[2])   # SE–NE
        kx2 = Th * harmonic_mean_2point(Kloc[2], Kloc[3])   # NE–NW
        ky2 = Tv * harmonic_mean_2point(Kloc[3], Kloc[0])   # NW–SW

        # Cross-face transmissibilities: same closed-form as uniform case but
        # now built from the four neighbouring face coefficients
        kdiag1 = 0.25 * (kx + ky)           # SW–NE
        kdiag2 = 0.25 * (kx2 + ky2)         # SE–NW
        return (kx, ky, kx2, ky2, kdiag1, kdiag2)


    def __mpfa_o(self, K):
        rows, cols, data = [], [], []
        for j in range(self.Ny + 1):
            for i in range(self.Nx + 1):
                cells = self.__four_cells_around_node(i, j, self.Nx, self.Ny)
                if len(cells) != 4:
                    continue
                Kloc = K.reshape(-1)[cells]
                Tij = self.__local_T(Kloc, self.hx, self.hy)

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

        return sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()

    def __5_point(self, K:np.ndarray, hor_func:callable, vert_func:callable) -> sp.csr_array:
        rows, cols, data = [], [], []
        for j in range(self.Ny):
            for i in range(self.Nx):
                diag = 0.0
                target_cell_gidx = gidx(i, j, self.Nx)
                target_cell_perm = K[i, j]
                # Cell to the left
                # If i is equal to 0, then this an edge and now cell to the left exists. Same logic for the following conditions.
                if i > 0:
                    t = hor_func(harmonic_mean_2point(target_cell_perm, K[i - 1, j]))
                    diag += t
                    rows.append(target_cell_gidx)
                    cols.append(gidx(i - 1, j, self.Nx))
                    data.append(-t)
                # Cell to the right
                if i < self.Nx - 1:
                    t = hor_func(harmonic_mean_2point(target_cell_perm, K[i + 1, j]))
                    diag += t
                    rows.append(target_cell_gidx)
                    cols.append(gidx(i + 1, j, self.Nx))
                    data.append(-t)
                # Cell below
                if j > 0:
                    t = vert_func(harmonic_mean_2point(target_cell_perm, K[i, j - 1]))
                    diag += t
                    rows.append(target_cell_gidx)
                    cols.append(gidx(i, j - 1, self.Nx))
                    data.append(-t)
                # Cell above
                if j < self.Ny - 1:
                    t = vert_func(harmonic_mean_2point(target_cell_perm, K[i, j + 1]))
                    diag += t
                    rows.append(target_cell_gidx)
                    cols.append(gidx(i, j + 1, self.Nx))
                    data.append(-t)
                rows.append(target_cell_gidx)
                cols.append(target_cell_gidx)
                data.append(diag)   
        return sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()

    def __apply_dirichlet_bc(self, A):
        rhs = np.zeros(self.N)
        for cell, pval in self.pressure_bc.items():
            A[cell, :] = 0.0
            A[cell, cell] = 1.0
            rhs[cell] = pval
        return A, rhs

    def solve(self, K, method="fdm"):
        if method not in list(self.__methods.keys()):
            raise NotImplementedError(f"Method {method} is not implemented. Implemented methods: {list(self.__methods.keys())}")
        A = self.__methods[method](K)
        A, rhs = self.__apply_dirichlet_bc(A)
        P = spla.spsolve(A, rhs)
        return P.reshape(self.Ny, self.Nx)