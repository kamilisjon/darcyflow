import time
import matplotlib.pyplot as plt
from darcyflow.porus_media import DarcyDomain, calculate_flow
from darcyflow.solver import solve, gidx
from darcyflow.plotting import plot_K, plot_P, plot_diferences

if __name__ == "__main__":
    # TODO: Current implementation does not play well with rectangles. Porus media and pressure field rectangles have different aspect ratios.
    Nx=Ny=40
    iterations = 100  # for speed benchamarks
    domain = DarcyDomain(Nx=Nx, Ny=Ny, pressure_bc={gidx(0, 0, Nx): 300.0,
                                                    gidx(Nx - 1, Ny - 1, Nx): -300.0})
    K = domain.exp_uniform_k()
    print("Running and timing solution. 100 itterations per method.")
    start = time.time()
    for _ in range(iterations):
        P_fdm = solve(domain, K, method="fdm")
    print(f"FDM solution time: {int((time.time()-start)*1000/iterations)}ms")
    U_fdm = calculate_flow(domain, K, P_fdm)
    start = time.time()
    for _ in range(iterations):
        P_tpfa = solve(domain, K, method="tpfa")
    print(f"TPFA solution time: {int((time.time()-start)*1000/iterations)}ms")
    U_tpfa = calculate_flow(domain, K, P_tpfa)
    start = time.time()
    for _ in range(iterations):
       P_mpfa = solve(domain, K, method="mpfa_o")
    print(f"MPFA solution time: {int((time.time()-start)*1000/iterations)}ms")
    U_mpfa = calculate_flow(domain, K, P_mpfa)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    plot_K(axs[0], K)
    plot_P(axs[1], P_fdm, U_fdm, "Pressure field (FDM)")
    plot_P(axs[2], P_tpfa, U_tpfa, "Pressure field (TPFA)")
    plot_P(axs[3], P_mpfa, U_mpfa, "Pressure field (MPFA)")

    plt.tight_layout()
    plt.show()

    plot_diferences(fdm=P_fdm, tpfa=P_tpfa, mpfa=P_mpfa)