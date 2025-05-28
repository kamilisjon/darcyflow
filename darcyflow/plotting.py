import math

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_K(ax, K, title="Permeability field"):
    im = ax.imshow(K, origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_P(ax, P, U, title="Presure field"):
    ny, nx = P.shape
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    ax.set_title(title, fontsize=10)
    plt.colorbar(ax.contourf(P, levels=50), ax=ax)
    ax.quiver(X, Y, U[0], U[1], color='black', scale=5e3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_diferences(*, fdm, tpfa, mpfa):
    solvers = {"FDM": fdm,
               "TPFA": tpfa,
               "MPFA": mpfa}
    pairs = list(itertools.combinations(solvers.items(), 2))
    n_pairs = len(pairs)
    ncols = math.ceil(np.sqrt(n_pairs))
    nrows = math.ceil(n_pairs / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, ((name1, data1), (name2, data2)) in enumerate(pairs):
        ax = axes[idx]
        diff = data1 - data2
        im = ax.imshow(diff, cmap='coolwarm', origin='lower')
        ax.set_title(f"{name1} vs {name2}")
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes[n_pairs:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
