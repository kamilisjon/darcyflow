import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings

from darcyflow.helpers import gidx, harmonic_mean_2point
warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)

# ---------- helpers ----------------------------------------------------------

class FiniteMethodsSolver:
    def __init__(self, Nx: int, Ny: int, pressure_bc: dict[int, float]):
        self.Nx, self.Ny, self.hx, self.hy, self.N, self.pressure_bc  = Nx, Ny, 1.0/Nx, 1.0/Ny, Nx*Ny, pressure_bc
        self.__methods: dict[str, callable] = {"fdm": self.__fdm, "tpfa": self.__tpfa, "mpfa_o": self.__mpfa_o}

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

        A = sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        return A

    def __tpfa(self, K):
        TX = np.zeros((self.Nx+1, self.Ny))
        TY = np.zeros((self.Nx, self.Ny+1))
        perm_inv = 1.0 / K
        TX[1:-1, :] = (2 * self.hy / self.hx) / (perm_inv[:-1, :] + perm_inv[1:, :])
        TY[:, 1:-1] = (2 * self.hx / self.hy) / (perm_inv[:, :-1] + perm_inv[:, 1:])
        rows, cols, data = [], [], []
        for j in range(self.Ny):
            for i in range(self.Nx):
                m = gidx(i,j,self.Nx)
                diag = 0.0
                if i>0:
                    rows.append(m)
                    cols.append(m-1)
                    t = TX[i,j]
                    data.append(-t)
                    diag+=t
                if i<self.Nx-1:
                    rows.append(m)
                    cols.append(m+1)
                    t = TX[i+1,j]
                    data.append(-t)
                    diag+=t
                if j>0:
                    rows.append(m)
                    cols.append(gidx(i,j-1,self.Nx))
                    t = TY[i,j]
                    data.append(-t)
                    diag+=t
                if j<self.Ny-1:
                    rows.append(m)
                    cols.append(gidx(i,j+1,self.Nx))
                    t = TY[i,j+1]
                    data.append(-t)
                    diag+=t
                rows.append(m)
                cols.append(m)
                data.append(diag)
        A = sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        return A

    def __fdm(self, K):
        rows, cols, data = [], [], []
        for j in range(self.Ny):
            for i in range(self.Nx):
                idx = gidx(i, j, self.Nx)
                diag = 0.0
                kij = K[i, j]

                # West neighbor
                if i > 0:
                    kw = harmonic_mean_2point(kij, K[i - 1, j])
                    rows.append(idx)
                    cols.append(gidx(i - 1, j, self.Nx))
                    data.append(-kw / self.hx**2)
                    diag += kw / self.hx**2

                # East neighbor
                if i < self.Nx - 1:
                    ke = harmonic_mean_2point(kij, K[i + 1, j])
                    rows.append(idx)
                    cols.append(gidx(i + 1, j, self.Nx))
                    data.append(-ke / self.hx**2)
                    diag += ke / self.hx**2

                # South neighbor
                if j > 0:
                    ks = harmonic_mean_2point(kij, K[i, j - 1])
                    rows.append(idx)
                    cols.append(gidx(i, j - 1, self.Nx))
                    data.append(-ks / self.hy**2)
                    diag += ks / self.hy**2

                # North neighbor
                if j < self.Ny - 1:
                    kn = harmonic_mean_2point(kij, K[i, j + 1])
                    rows.append(idx)
                    cols.append(gidx(i, j + 1, self.Nx))
                    data.append(-kn / self.hy**2)
                    diag += kn / self.hy**2

                # Center
                rows.append(idx)
                cols.append(idx)
                data.append(diag)

        A = sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        return A

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