import matplotlib.pyplot as plt
from darcyflow.porus_media import exp_uniform_k, calculate_flow
from darcyflow.solver import solve, gidx
from darcyflow.plotting import plot_K, plot_P, plot_diferences

if __name__ == "__main__":
    #TODO: Current implementation does not play well with rectangles. Porus media and pressure field rectangles have different aspect ratios.
    Nx = Ny = 40
    pressure_bc: dict[int, float] = {gidx(0, 0, Nx): -50.0,
                                     gidx(0, 30, Nx): 400.0,
                                     gidx(Nx-1, Ny-1, Nx): -100.0}
    K = exp_uniform_k(Nx, Ny)
    P_fdm = solve(Nx, Ny, pressure_bc, K, method="fdm")
    U_fdm = calculate_flow(Nx, Ny, K, P_fdm)
    P_tpfa = solve(Nx, Ny, pressure_bc, K, method="tpfa")
    U_tpfa = calculate_flow(Nx, Ny, K, P_tpfa)
    P_mpfa = solve(Nx, Ny, pressure_bc, K, method="mpfa_o")
    U_mpfa = calculate_flow(Nx, Ny, K, P_mpfa)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    plot_K(axs[0], K)
    plot_P(axs[1], P_fdm, U_fdm, "Pressure field (FDM)")
    plot_P(axs[2], P_tpfa, U_tpfa, "Pressure field (TPFA)")
    plot_P(axs[3], P_mpfa, U_mpfa, "Pressure field (MPFA)")

    plt.tight_layout()
    plt.show()

    plot_diferences(fdm=P_fdm, tpfa=P_tpfa, mpfa=P_mpfa)