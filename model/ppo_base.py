import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def resolve_activation(act):
    act = str(act or "relu").strip().lower()
    if act == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {act}")


class FCNet(nn.Module):
    def __init__(self, 
                 n_features,
                 output_dim,
                 hidden_dim = 256,
                 hidden_dim2 = 256,
                 act = "relu",
                 safe_dist = 0.8,
                 alpha = 2.0,
                 beta = 0.2,
                 robot_type='single_integrator',
                 vmax = 3.0, omega_max = 3.0,
                 slack_weight=10.0):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.safe_dist = safe_dist
        self.act_name = str(act)
        self.act = resolve_activation(act)


        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, output_dim)

        if self.output_dim != 1:
            print(f"[FCNet] robot_type={robot_type}, safe_dist={safe_dist:.3f}",flush=True)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        obs = obs.to(self.fc1.weight.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.reshape(obs.size(0), -1)

        x = self.act(self.fc1(obs))
        x21 = self.act(self.fc21(x))
        x31 = self.fc31(x21)
        
        return x31
        
        
