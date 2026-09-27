import torch
from cvxopt import solvers, matrix
import numpy as np

def solve_qp_cvxopt(Q, p, G, h, device, dtype, warm_start_x=None):
    # Convert PyTorch tensors to numpy arrays
    if isinstance(Q, torch.Tensor):
        Q = Q.detach().cpu().numpy()
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(G, torch.Tensor):
        G = G.detach().cpu().numpy()
    if isinstance(h, torch.Tensor):
        h = h.detach().cpu().numpy()

    # Use float64 inside QP for better numerical stability
    Q = np.asarray(Q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    mat_Q = matrix(Q)
    mat_p = matrix(p)
    mat_G = matrix(G)
    mat_h = matrix(h)

    initvals = None
    if warm_start_x is not None:
        if isinstance(warm_start_x, torch.Tensor):
            warm_start_x = warm_start_x.detach().cpu().numpy()
        warm_start_x = np.asarray(warm_start_x, dtype=np.float64).reshape(-1)
        if warm_start_x.shape[0] == Q.shape[0]:
            initvals = {'x': matrix(warm_start_x)}

    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 100

    try:
        if initvals is not None:
            sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h, initvals=initvals)
        else:
            sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    except Exception:
        n_vars = Q.shape[0]
        x = torch.zeros((1, n_vars), device=device, dtype=dtype)
        return x, True

    infeasible = sol.get('status') != 'optimal' or sol['x'] is None
    if sol['x'] is None:
        n_vars = Q.shape[0]
        x = torch.zeros((1, n_vars), device=device, dtype=dtype)
    else:
        # cvxopt returns a column vector (n_vars, 1); convert to torch row (1, n_vars)
        x_np = np.array(sol['x'], dtype=np.float64)
        x = torch.tensor(x_np, device=device, dtype=dtype).view(1, -1)
    return x, infeasible
