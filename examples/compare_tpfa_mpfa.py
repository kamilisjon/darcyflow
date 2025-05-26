import matplotlib.pyplot as plt
from darcyflow.porus_media import exp_uniform_k
from darcyflow.solver import solve, gidx
from darcyflow.plotting import plot_K, plot_P

if __name__ == "__main__":
    #TODO: Current implementation does not play well with rectangles. Porus media and pressure field rectangles have different aspect ratios.
    Nx = Ny = 40
    pressure_bc: dict[int, float] = {gidx(0, 0, Nx): -50.0,
                                     gidx(0, 30, Nx): 400.0,
                                     gidx(Nx-1, Ny-1, Nx): -100.0}
    K = exp_uniform_k(Nx, Ny)
    P_tpfa = solve(Nx, Ny, pressure_bc, K, method="tpfa")
    P_mpfa = solve(Nx, Ny, pressure_bc, K, method="mpfa_o")
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(14,4))
    plot_K(ax1, K)
    plot_P(ax2, P_tpfa, "Pressure field (TPFA)")
    plot_P(ax3, P_mpfa, "Pressure field (MPFA)")
    plt.tight_layout()
    plt.show()