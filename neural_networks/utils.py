import torch
from tqdm import trange
from darcyflow.solver import solve
from darcyflow.porus_media import DarcyDomain

def generate_dataset(n_samples = 10000, domain = DarcyDomain(), solver_method = 'fdm'):
    K_list, P_list = [], []
    for _ in trange(n_samples, desc='Generating data'):
        K = domain.exp_uniform_k()
        P = solve(domain, K, solver_method)
        K_list.append(K)
        P_list.append(P)
    # Normalize data
    K_data, P_data = generate_dataset(n_samples, domain, solver_method)
    K_data = (K_data - K_data.mean()) / K_data.std()
    P_data = (P_data - P_data.mean()) / P_data.std()
    K_tensor = torch.tensor(K_data, dtype=torch.float32).unsqueeze(1)
    P_tensor = torch.tensor(P_data, dtype=torch.float32).unsqueeze(1)
    torch.save(K_tensor, 'K_tensor.pt')
    torch.save(P_tensor, 'P_tensor.pt')