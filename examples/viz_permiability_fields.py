import matplotlib.pyplot as plt
from darcyflow.porus_media import DarcyDomain
from darcyflow.plotting import plot_K

if __name__ == "__main__":
    domain = DarcyDomain()
    fields = [
        ("Homogenous",         domain.homogenous_k()),
        ("Layered",            domain.layered_k()),
        ("Channel",            domain.channel_k()),
        ("Checkerboard",       domain.checkerboard_k()),
        ("Gaussian field",     domain.gaussian_random_k()),
        ("Gaussian field v2",  domain.exp_uniform_k()),
        ("Circular inclusion", domain.radial_inclusion_k()),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (name, K) in zip(axes.flat, fields): plot_K(ax, K, name)
    axes.flat[-1].axis('off')
    plt.tight_layout()
    plt.show()