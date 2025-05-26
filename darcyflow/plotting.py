import matplotlib.pyplot as plt

def plot_K(ax, K, title="Permeability field"):
    im = ax.imshow(K, origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='')

def plot_P(ax, P, title="Presure field"):
    ax.set_title(title, fontsize=10)
    plt.colorbar(ax.contourf(P, levels=50), ax=ax)