import matplotlib.pyplot as plt
from darcyflow.porus_media import homogenous_k, layered_k, channel_k, checkerboard_k, gaussian_random_k, exp_uniform_k, radial_inclusion_k
from darcyflow.plotting import plot_K

if __name__ == "__main__":
    Nx = Ny = 40
    fields = [
        ("Homogenous",         homogenous_k(Nx, Ny)),
        ("Layered",            layered_k(Nx, Ny)),
        ("Channel",            channel_k(Nx, Ny)),
        ("Checkerboard",       checkerboard_k(Nx, Ny)),
        ("Gaussian field",     gaussian_random_k(Nx, Ny)),
        ("Gaussian field v2",  exp_uniform_k(Nx, Ny)),
        ("Circular inclusion", radial_inclusion_k(Nx, Ny)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (name, K) in zip(axes.flat, fields): plot_K(ax, K, name)
    plt.tight_layout()
    plt.show()