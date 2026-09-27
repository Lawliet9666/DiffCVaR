"""Pure PPO actor/critic with a shared sparse spatio-temporal GNN encoder."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type

from model import jax_actor_critic as actor_critic
from model.jax_gnn_encoder import (
    actor_latent_and_mean,
    critic_value_from_latent,
    init_shared_gnn_actor,
    init_shared_gnn_critic,
    shared_critic_value,
)
from model.jax_ppo_base import cfg_get, require_config_value


@dataclass(frozen=True)
class PPOGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    history_len: int
    obs_frame_dim: int
    num_humans: int
    graph_hidden_dim: int
    graph_latent_dim: int
    graph_layers: int
    position_scale: float
    velocity_scale: float


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "ppo_gnn", "model")
    require_config_value(actor_cfg, "graph_coordinate_frame", "robot_body", "model.actor")
    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    if history_len <= 0:
        raise ValueError("ppo_gnn requires history_len > 0")
    if int(act_dim) != 2:
        raise ValueError("ppo_gnn requires act_dim=2")
    return PPOGNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        num_humans=max(0, (obs_frame_dim - 6) // 6),
        graph_hidden_dim=int(cfg_get(actor_cfg, "graph_hidden_dim", 64)),
        graph_latent_dim=int(cfg_get(actor_cfg, "graph_latent_dim", 16)),
        graph_layers=int(cfg_get(actor_cfg, "graph_layers", 3)),
        position_scale=float(cfg_get(actor_cfg, "position_scale", 20.0)),
        velocity_scale=float(cfg_get(actor_cfg, "velocity_scale", 1.0)),
    )


def init_params_from_config(
    key,
    model_cfg,
    critic_cfg,
    act_dim,
    action_low,
    action_high,
    use_init_weights=True,
):
    actor_cfg = model_cfg.actor
    hparams = hparams_from_config(model_cfg, critic_cfg, act_dim=act_dim)
    hidden_dim = int(cfg_get(actor_cfg, "hidden_dim", 256))
    control_hidden_dim = int(cfg_get(actor_cfg, "control_hidden_dim", hidden_dim))
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    keys = jax.random.split(key, 18)
    actor = init_shared_gnn_actor(
        keys[:4],
        hparams,
        act_dim,
        hidden_dim,
        control_hidden_dim,
        actor_cfg.action_std_init,
        "control_head",
        use_init_weights,
    )
    critic = init_shared_gnn_critic(keys[7:10], hparams, critic_hidden_dim, critic_hidden_dim2, use_init_weights)
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _actor_outputs(params, hparams, obs):
    return actor_latent_and_mean(params, hparams, obs, "control_head")


def policy_action_mean(params, hparams, obs):
    """Return the deterministic pre-tanh action mean."""
    _robot_latent, mean = _actor_outputs(params, hparams, obs)
    return mean


policy_action_to_env_action = actor_critic.policy_action_to_env_action


def critic_value(params, hparams, obs):
    return shared_critic_value(params, hparams, obs)


def get_action_and_value(
    params,
    hparams,
    obs,
    action=None,
    noise=None,
    policy_action=None,
    return_policy_action=False,
):
    robot_latent, mean = _actor_outputs(params, hparams, obs)
    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        mean,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )
    value = critic_value_from_latent(params, hparams, robot_latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    return outputs
