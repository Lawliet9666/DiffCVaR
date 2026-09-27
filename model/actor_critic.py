import torch
from torch import nn
from torch.distributions import Independent, Normal

from model.ppo_base import FCNet


class ActorCritic(nn.Module):
    """Shared train/eval model wrapper for PPO policies."""

    def __init__(
        self,
        policy_class,
        obs_dim,
        act_dim,
        actor_kwargs,
        critic_kwargs,
        action_low,
        action_high,
        action_std_init=0.5,
    ):
        super().__init__()
        self.actor = policy_class(obs_dim, act_dim, **actor_kwargs)
        self.critic = FCNet(obs_dim, 1, **critic_kwargs)
        self.log_std = nn.Parameter(torch.log(torch.full((int(act_dim),), float(action_std_init))))

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("act_low", action_low)
        self.register_buffer("act_high", action_high)
        self.register_buffer("act_scale", 0.5 * (action_high - action_low))
        self.register_buffer("act_bias", 0.5 * (action_high + action_low))

    def forward(self, obs):
        return self.actor(obs)

    def build_action_dist(self, mean):
        std = torch.exp(self.log_std).clamp_min(1e-6)
        return Independent(Normal(mean, std), 1)

    def get_action_deterministic(self, obs):
        return self.actor(obs)

    def policy_action_to_env_action(self, obs, policy_action, return_squashed=False):
        squashed = torch.tanh(policy_action)
        action = self.act_bias + self.act_scale * squashed
        if return_squashed:
            return action, squashed
        return action

    def env_action_to_policy_action(self, action):
        scale = self.act_scale
        bias = self.act_bias
        while scale.dim() < action.dim():
            scale = scale.unsqueeze(0)
            bias = bias.unsqueeze(0)

        squashed = (action - bias) / scale.clamp_min(1e-6)
        squashed = squashed.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        policy_action = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        return policy_action, squashed

    def squash_log_prob(self, base_log_prob, squashed):
        tanh_logdet = torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        scale_logdet = torch.log(self.act_scale.clamp_min(1e-6)).sum()
        return base_log_prob - tanh_logdet - scale_logdet

    def reset_episode_cache(self):
        targets = [self.actor]
        seen = set()
        while targets:
            obj = targets.pop()
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            if hasattr(obj, "_qp_warm_start"):
                obj._qp_warm_start = None
            if hasattr(obj, "_u_prev"):
                obj._u_prev = None
            if hasattr(obj, "infeasible"):
                obj.infeasible = False

            nested_actor = getattr(obj, "actor", None)
            if nested_actor is not None and nested_actor is not obj:
                targets.append(nested_actor)
            nested_module = getattr(obj, "module", None)
            if nested_module is not None and nested_module is not obj:
                targets.append(nested_module)
