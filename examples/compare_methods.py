import time
import matplotlib.pyplot as plt
from darcyflow.porus_media import DarcyDomain, calculate_flow
from darcyflow.solver import solve, gidx
from darcyflow.plotting import plot_K, plot_P, plot_diferences
import torch

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

    start = time.time()
    for _ in range(iterations):
        P_tpfa = solve(domain, K, method="tpfa")
    print(f"TPFA solution time: {int((time.time()-start)*1000/iterations)}ms")

    start = time.time()
    for _ in range(iterations):
       P_mpfa = solve(domain, K, method="mpfa_o")
    print(f"MPFA solution time: {int((time.time()-start)*1000/iterations)}ms")

    cnn = torch.load('cnn_model_full.pth', weights_only=False)
    fno = torch.load('fno_model_full.pth', weights_only=False)

    # TODO: I believe the model will be poor, since the training data was normalized
    P_cnn = cnn(torch.tensor(K, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    P_fno = fno(torch.tensor(K, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    fig, axs = plt.subplots(3, 2, figsize=(10, 14))
    axs = axs.flatten()
    plot_K(axs[0], K)
    plot_P(axs[1], P_fdm, calculate_flow(domain, K, P_fdm), "Pressure field (FDM)")
    plot_P(axs[2], P_tpfa, calculate_flow(domain, K, P_tpfa), "Pressure field (TPFA)")
    plot_P(axs[3], P_mpfa, calculate_flow(domain, K, P_mpfa), "Pressure field (MPFA)")
    plot_P(axs[4], P_cnn, calculate_flow(domain, K, P_cnn), "Pressure field (CNN)")
    plot_P(axs[5], P_fno, calculate_flow(domain, K, P_fno), "Pressure field (FNO)")

    plt.tight_layout()
    plt.show()

    plot_diferences(fdm=P_fdm, tpfa=P_tpfa, mpfa=P_mpfa, cnn=P_cnn, fno=P_fno)