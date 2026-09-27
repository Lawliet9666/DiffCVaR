"""JAX diff-CVaR actor with spatio-temporal GNN encoder."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.nn as jnn
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type

from model import jax_actor_critic as actor_critic
from model import jax_cvar_qp
from model.jax_gnn_encoder import (
    critic_value_from_latent as shared_critic_value_from_latent,
    current_obs_history,
    init_shared_gnn_actor,
    init_shared_gnn_critic,
    shared_critic_value,
    spatio_temporal_graph_encode,
)
from model.jax_ppo_base import activation, cfg_get, dense, init_linear, mlp, require_config_value


@dataclass(frozen=True)
class DiffCVaRGNNHParams:
    act_dim: int
    robot_type: str
    actor_act: str
    critic_act: str
    margin_base: float
    margin_extra_max: float
    alpha: float
    beta_min: float
    beta_max: float
    lookahead_distance: float
    gmm_lateral_ratio: float
    qp_max_iter: int
    qp_eps: float
    qp_not_improved_lim: int
    qp_verbose: int
    qp_check_q_spd: bool
    qp_slack_weight: float
    qp_residual_reward_weight: float
    qp_full_diagnostics: bool
    history_len: int
    obs_frame_dim: int
    num_humans: int
    graph_hidden_dim: int
    graph_latent_dim: int
    graph_layers: int
    position_scale: float
    velocity_scale: float
    qp_obs_top_k: int


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "diff_cvar_gnn", "model")
    require_config_value(actor_cfg, "graph_coordinate_frame", "robot_body", "model.actor")
    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    num_humans = max(0, (obs_frame_dim - 6) // 6)
    qp_obs_top_k = min(int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans)
    if history_len <= 0:
        raise ValueError("diff_cvar_gnn requires history_len > 0")
    if int(act_dim) != 2:
        raise ValueError("diff_cvar_gnn requires act_dim=2")
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cvar_gnn requires qp_obs_top_k > 0")
    beta_min = float(cfg_get(actor_cfg, "beta_min", 0.05))
    beta_max = float(actor_cfg.beta)
    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError("diff_cvar_gnn requires 0 < beta_min < beta < 1")
    if qp_obs_top_k * beta_min > beta_max:
        raise ValueError("diff_cvar_gnn requires qp_obs_top_k * beta_min <= beta")
    margin_base = float(actor_cfg.margin_base)
    margin_extra_max = float(actor_cfg.margin_extra_max)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")
    if margin_extra_max < 0.0:
        raise ValueError("model.actor.margin_extra_max must be >= 0")
    return DiffCVaRGNNHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(cfg_get(model_cfg, "robot_type", "unicycle")),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        margin_base=margin_base,
        margin_extra_max=margin_extra_max,
        alpha=float(actor_cfg.alpha),
        beta_min=beta_min,
        beta_max=beta_max,
        lookahead_distance=float(cfg_get(actor_cfg, "lookahead_distance", 0.2)),
        gmm_lateral_ratio=float(actor_cfg.gmm_lateral_ratio),
        qp_max_iter=int(cfg_get(actor_cfg, "qp_max_iter", 40)),
        qp_eps=float(cfg_get(actor_cfg, "qp_eps", 1e-12)),
        qp_not_improved_lim=int(cfg_get(actor_cfg, "qp_not_improved_lim", 3)),
        qp_verbose=int(cfg_get(actor_cfg, "qp_verbose", -1)),
        qp_check_q_spd=bool(cfg_get(actor_cfg, "qp_check_q_spd", False)),
        qp_slack_weight=float(cfg_get(actor_cfg, "qp_slack_weight", 10.0)),
        qp_residual_reward_weight=float(actor_cfg.qp_residual_reward_weight),
        qp_full_diagnostics=bool(cfg_get(actor_cfg, "qp_full_diagnostics", False)),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        num_humans=num_humans,
        graph_hidden_dim=int(cfg_get(actor_cfg, "graph_hidden_dim", 64)),
        graph_latent_dim=int(cfg_get(actor_cfg, "graph_latent_dim", 16)),
        graph_layers=int(cfg_get(actor_cfg, "graph_layers", 3)),
        position_scale=float(cfg_get(actor_cfg, "position_scale", 20.0)),
        velocity_scale=float(cfg_get(actor_cfg, "velocity_scale", 1.0)),
        qp_obs_top_k=qp_obs_top_k,
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
    scalar_hidden_dim = int(cfg_get(actor_cfg, "scalar_hidden_dim", hidden_dim))
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    sqrt2 = math.sqrt(2.0)

    keys = jax.random.split(key, 18)
    actor = init_shared_gnn_actor(
        keys[:4],
        hparams,
        act_dim,
        hidden_dim,
        control_hidden_dim,
        actor_cfg.action_std_init,
        "nominal_head",
        use_init_weights,
    )
    actor.update({
        "human_fc1": init_linear(keys[4], hparams.graph_latent_dim, scalar_hidden_dim, sqrt2, use_init_weights),
        "human_beta_head": init_linear(keys[5], scalar_hidden_dim, 1, sqrt2 * 0.01, use_init_weights),
        "human_rsafe_head": init_linear(keys[6], scalar_hidden_dim, 1, sqrt2 * 0.01, use_init_weights),
        "gmm_weights": jnp.asarray(actor_cfg.gmm_weights, dtype=jnp.float32),
        "gmm_variances": jnp.square(jnp.asarray(actor_cfg.gmm_stds, dtype=jnp.float32)),
    })

    critic = init_shared_gnn_critic(keys[7:10], hparams, critic_hidden_dim, critic_hidden_dim2, use_init_weights)
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _body_to_world(vectors, theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return jnp.stack([c * x - s * y, s * x + c * y], axis=1)


def _current_obs(obs, hparams):
    obs_history = current_obs_history(obs, hparams)
    return obs_history, obs_history[:, -1, :]


def _select_qp_topk(current_obs, beta_all, r_safe_all, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape((batch_size, hparams.num_humans, 6))
    mask = jnp.clip(blocks[:, :, 5], 0.0, 1.0) > 0.5
    dist = jnp.linalg.norm(blocks[:, :, 0:2], axis=2)
    sort_key = jnp.where(mask, dist, jnp.inf)
    _topk_values, topk_ids = jax.lax.top_k(-sort_key, hparams.qp_obs_top_k)
    batch_idx = jnp.arange(batch_size)[:, None]
    topk_blocks = blocks[batch_idx, topk_ids]
    beta_topk = beta_all[batch_idx, topk_ids]
    r_safe_topk = r_safe_all[batch_idx, topk_ids]
    qp_obs = jnp.concatenate([current_obs[:, :6], topk_blocks.reshape((batch_size, -1))], axis=1)
    return qp_obs, beta_topk, r_safe_topk


def build_qp_context(obs, hparams):
    """Build observation-only QP context for reuse during PPO updates."""
    _obs_history, current_obs = _current_obs(obs, hparams)
    batch_size = current_obs.shape[0]
    num_humans = int(hparams.num_humans)
    human_ids = jnp.broadcast_to(
        jnp.arange(num_humans, dtype=jnp.int32)[None, :],
        (batch_size, num_humans),
    )
    zeros = jnp.zeros((batch_size, num_humans), dtype=jnp.float32)
    qp_obs, qp_topk_ids, _unused = _select_qp_topk(current_obs, human_ids, zeros, hparams)
    return {
        "qp_obs": qp_obs,
        "qp_topk_ids": qp_topk_ids.astype(jnp.int32),
    }


def _gather_topk(values, qp_topk_ids):
    batch_idx = jnp.arange(values.shape[0])[:, None]
    return values[batch_idx, qp_topk_ids.astype(jnp.int32)]


def _allocate_beta_budget(beta_logits, mask, hparams):
    active = jnp.clip(mask, 0.0, 1.0) > 0.5
    active_float = active.astype(beta_logits.dtype)
    masked_logits = jnp.where(
        active, beta_logits, jnp.asarray(-1e9, dtype=beta_logits.dtype)
    )
    weights = jnn.softmax(masked_logits, axis=-1) * active_float
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)

    valid_count = jnp.sum(active_float, axis=-1, keepdims=True)
    remaining = jnp.maximum(hparams.beta_max - hparams.beta_min * valid_count, 0.0)
    beta = hparams.beta_min + remaining * weights
    return jnp.where(
        active, beta, jnp.asarray(hparams.beta_min, dtype=beta_logits.dtype)
    )


def _safe_distances_from_margin_logits(current_obs, margin_logits, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape((batch_size, hparams.num_humans, 6))
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(hparams.margin_base, dtype=current_obs.dtype)
    margin_extra_max = jnp.asarray(hparams.margin_extra_max, dtype=current_obs.dtype)
    learned_margin = margin_base + margin_extra_max * jnn.sigmoid(margin_logits)
    return robot_radius + human_radii + learned_margin


def _actor_outputs(params, hparams, obs, qp_context=None):
    obs_history, current_obs = _current_obs(obs, hparams)
    actor_params = params["actor"]
    robot_latent, human_latents = spatio_temporal_graph_encode(
        actor_params["graph_encoder"],
        hparams,
        obs_history,
        return_human_latents=True,
    )
    x = activation(dense(actor_params["fc1"], robot_latent), hparams.actor_act)
    nominal_xy = mlp(actor_params["nominal_head"], x, hparams.actor_act)

    human_x = activation(dense(actor_params["human_fc1"], human_latents), hparams.actor_act)
    beta_logits_all = dense(actor_params["human_beta_head"], human_x).squeeze(-1)
    margin_logits = dense(actor_params["human_rsafe_head"], human_x).squeeze(-1)
    r_safe_all = _safe_distances_from_margin_logits(current_obs, margin_logits, hparams)

    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    qp_topk_ids = qp_context["qp_topk_ids"]
    qp_blocks = qp_obs[:, 6:].reshape((qp_obs.shape[0], hparams.qp_obs_top_k, 6))
    beta_logits_topk = _gather_topk(beta_logits_all, qp_topk_ids)
    beta_topk = _allocate_beta_budget(
        beta_logits_topk, qp_blocks[:, :, 5], hparams
    )
    r_safe_topk = _gather_topk(r_safe_all, qp_topk_ids)
    nominal_xy = _body_to_world(nominal_xy, qp_obs[:, 4])
    safe_xy, slack = jax_cvar_qp.solve_slack_qp(
        params, hparams, qp_obs, nominal_xy, beta_topk, r_safe_topk
    )
    return robot_latent, qp_obs, nominal_xy, beta_topk, r_safe_topk, safe_xy, slack


def safe_action_mean(params, hparams, obs, qp_context=None):
    _robot_latent, _qp_obs, nominal_xy, _beta, _r_safe, safe_xy, _slack = _actor_outputs(
        params,
        hparams,
        obs,
        qp_context=qp_context,
    )
    return safe_xy


policy_action_mean = safe_action_mean
policy_action_to_env_action = actor_critic.policy_action_to_env_action


def _critic_value_from_latent(params, hparams, robot_latent):
    return shared_critic_value_from_latent(params, hparams, robot_latent)


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
    return_diagnostics=False,
    qp_context=None,
):
    robot_latent, qp_obs, nominal_xy, beta, r_safe, safe_xy, slack = _actor_outputs(
        params,
        hparams,
        obs,
        qp_context=qp_context,
    )
    mean = safe_xy

    action, logprob, entropy, sampled_policy_action = actor_critic.action_logprob_entropy(
        params,
        mean,
        action=action,
        noise=noise,
        policy_action=policy_action,
    )

    value = _critic_value_from_latent(params, hparams, robot_latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        if hparams.qp_full_diagnostics:
            diagnostics = jax_cvar_qp.full_diagnostics(params, hparams, qp_obs, nominal_xy, beta, r_safe, safe_xy, slack)
        else:
            diagnostics = jax_cvar_qp.lightweight_diagnostics(params, hparams, qp_obs, nominal_xy, beta, r_safe, safe_xy, slack)
        outputs = outputs + (diagnostics,)
    return outputs
