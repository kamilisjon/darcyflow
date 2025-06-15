import numpy as np
import torch
from tqdm import trange
from darcyflow.helpers import gidx
from darcyflow.solver import FiniteMethodsSolver
from darcyflow.porus_media import DarcyDomain

def generate_dataset(n_samples = 10000, domain = DarcyDomain(Nx=40, Ny=40), solver_method = 'fdm'):
    K_list, P_list = [], []
    solver = FiniteMethodsSolver(Nx=domain.Nx, Ny=domain.Nx, 
                                 pressure_bc={gidx(0, 0, domain.Nx): 300.0,
                                              gidx(domain.Nx - 1, domain.Ny - 1, domain.Nx): -300.0})
    for _ in trange(n_samples, desc='Generating data'):
        K = domain.exp_uniform_k()
        P = solver(K, solver_method)
        K_list.append(K)
        P_list.append(P)
    # Normalize data
    K_data, P_data = np.array(K_list), np.array(P_list)
    K_data = (K_data - K_data.mean()) / K_data.std()
    P_data = (P_data - P_data.mean()) / P_data.std()
    K_tensor = torch.tensor(K_data, dtype=torch.float32).unsqueeze(1)
    P_tensor = torch.tensor(P_data, dtype=torch.float32).unsqueeze(1)
    torch.save(K_tensor, 'K_tensor.pt')
    torch.save(P_tensor, 'P_tensor.pt')
    torch.save({'K_mean': K_data.mean(), 'K_std': K_data.std(),
                'P_mean': P_data.mean(), 'P_std': P_data.std()}, 'norm_stats.pt')

    return K_tensor, P_tensor
