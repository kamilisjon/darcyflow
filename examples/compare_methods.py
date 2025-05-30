import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from neural_networks.CNN import DarcyCNN

from darcyflow.porus_media import DarcyDomain, calculate_flow
from darcyflow.solver import solve, gidx
from darcyflow.plotting import plot_K, plot_P, plot_diferences

def calculate_accuracy(P_fdm, P_target, name_target):
    mae, mse = np.mean(np.abs(P_target - P_fdm)), np.mean((P_target - P_fdm)**2)
    print(f"{name_target}  –  MAE: {mae:.4e},  MSE: {mse:.4e}")

if __name__ == "__main__":
    # TODO: Current implementation does not play well with rectangles. Porus media and pressure field rectangles have different aspect ratios.
    Nx=Ny=40
    iterations = 100  # for speed benchamarks
    iterations = max(iterations, 1)
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

    cnn = torch.load('cnn_model_full.pth', map_location=torch.device('cpu'), weights_only=False)
    fno = torch.load('fno_model_full.pth', map_location=torch.device('cpu'), weights_only=False)

    stats = torch.load('norm_stats.pt')
    K_mean, K_std = stats['K_mean'], stats['K_std']
    P_mean, P_std = stats['P_mean'], stats['P_std']
    K_new_norm = torch.tensor((K - K_mean) / K_std, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    start = time.time()
    for _ in range(iterations):
        P_cnn = cnn(K_new_norm).detach().numpy().squeeze(0).squeeze(0)
        P_cnn = P_cnn * P_std + P_mean
    print(f"CNN solution time: {int((time.time()-start)*1000/iterations)}ms")
    start = time.time()
    for _ in range(iterations):
        P_fno = fno(K_new_norm).detach().numpy().squeeze(0).squeeze(0)
        P_fno = P_fno * P_std + P_mean
    print(f"FNO solution time: {int((time.time()-start)*1000/iterations)}ms")

    calculate_accuracy(P_fdm, P_tpfa, "TPFA")
    calculate_accuracy(P_fdm, P_mpfa, "MPFA")
    calculate_accuracy(P_fdm, P_cnn, "CNN")
    calculate_accuracy(P_fdm, P_fno, "FNO")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    plot_K(axs[0], K)
    plot_P(axs[1], P_cnn, calculate_flow(domain, K, P_cnn), "Pressure field (CNN)")
    plot_P(axs[2], P_fno, calculate_flow(domain, K, P_fno), "Pressure field (FNO)")
    plot_P(axs[3], P_fdm, calculate_flow(domain, K, P_fdm), "Pressure field (FDM)")
    plot_P(axs[4], P_tpfa, calculate_flow(domain, K, P_tpfa), "Pressure field (TPFA)")
    plot_P(axs[5], P_mpfa, calculate_flow(domain, K, P_mpfa), "Pressure field (MPFA)")

    plt.tight_layout()
    plt.show()

    plot_diferences(FDM=P_fdm, TPFA=P_tpfa, MPFA=P_mpfa, CNN=P_cnn, FNO=P_fno)
