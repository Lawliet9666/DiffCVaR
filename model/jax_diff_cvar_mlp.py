"""JAX diff-CVaR actor with a shared MLP encoder and no graph network."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.nn as jnn
import jax.numpy as jnp

from env.jax_robot import normalize_robot_type
from model import jax_actor_critic as actor_critic
from model import jax_cvar_qp
from model.jax_ppo_base import (
    activation,
    cfg_get,
    dense,
    init_head,
    init_linear,
    mlp,
    require_config_value,
)


@dataclass(frozen=True)
class DiffCVaRMLPHParams:
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
    obs_dim: int
    num_humans: int
    encoder_hidden_dim: int
    encoder_latent_dim: int
    encoder_layers: int
    position_scale: float
    velocity_scale: float
    qp_obs_top_k: int

    @property
    def encoder_input_dim(self):
        processed_frame_dim = 5 + 6 * int(self.num_humans)
        return int(self.history_len) * processed_frame_dim


def hparams_from_config(model_cfg, critic_cfg, act_dim=2):
    actor_cfg = model_cfg.actor
    require_config_value(model_cfg, "type", "diff_cvar_mlp", "model")
    require_config_value(
        actor_cfg, "coordinate_frame", "robot_body", "model.actor"
    )

    history_len = int(model_cfg.history_len)
    obs_frame_dim = int(model_cfg.obs_frame_dim)
    obs_dim = int(model_cfg.obs_dim)
    if history_len <= 0:
        raise ValueError("diff_cvar_mlp requires history_len > 0")
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError(
            "diff_cvar_mlp requires obs_frame_dim = 6 + num_humans * 6"
        )
    if obs_dim != history_len * obs_frame_dim:
        raise ValueError(
            "diff_cvar_mlp requires obs_dim = history_len * obs_frame_dim"
        )
    if int(act_dim) != 2:
        raise ValueError("diff_cvar_mlp requires act_dim=2")

    num_humans = (obs_frame_dim - 6) // 6
    qp_obs_top_k = min(
        int(cfg_get(actor_cfg, "qp_obs_top_k", 5)), num_humans
    )
    if qp_obs_top_k <= 0:
        raise ValueError("diff_cvar_mlp requires qp_obs_top_k > 0")

    beta_min = float(cfg_get(actor_cfg, "beta_min", 0.05))
    beta_max = float(actor_cfg.beta)
    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError("diff_cvar_mlp requires 0 < beta_min < beta < 1")
    if qp_obs_top_k * beta_min > beta_max:
        raise ValueError(
            "diff_cvar_mlp requires qp_obs_top_k * beta_min <= beta"
        )

    margin_base = float(actor_cfg.margin_base)
    margin_extra_max = float(actor_cfg.margin_extra_max)
    if margin_base < 0.0:
        raise ValueError("model.actor.margin_base must be >= 0")
    if margin_extra_max < 0.0:
        raise ValueError("model.actor.margin_extra_max must be >= 0")

    encoder_hidden_dim = int(cfg_get(actor_cfg, "encoder_hidden_dim", 64))
    encoder_latent_dim = int(cfg_get(actor_cfg, "encoder_latent_dim", 16))
    encoder_layers = int(cfg_get(actor_cfg, "encoder_layers", 3))
    position_scale = float(cfg_get(actor_cfg, "position_scale", 20.0))
    velocity_scale = float(cfg_get(actor_cfg, "velocity_scale", 1.0))
    if encoder_hidden_dim <= 0 or encoder_latent_dim <= 0 or encoder_layers <= 0:
        raise ValueError(
            "diff_cvar_mlp encoder dimensions and layer count must be > 0"
        )
    if position_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError(
            "diff_cvar_mlp position_scale and velocity_scale must be > 0"
        )

    return DiffCVaRMLPHParams(
        act_dim=int(act_dim),
        robot_type=normalize_robot_type(
            cfg_get(model_cfg, "robot_type", "unicycle")
        ),
        actor_act=str(cfg_get(actor_cfg, "act", "relu")),
        critic_act=str(cfg_get(critic_cfg, "act", "relu")),
        margin_base=margin_base,
        margin_extra_max=margin_extra_max,
        alpha=float(actor_cfg.alpha),
        beta_min=beta_min,
        beta_max=beta_max,
        lookahead_distance=float(
            cfg_get(actor_cfg, "lookahead_distance", 0.2)
        ),
        gmm_lateral_ratio=float(actor_cfg.gmm_lateral_ratio),
        qp_max_iter=int(cfg_get(actor_cfg, "qp_max_iter", 40)),
        qp_eps=float(cfg_get(actor_cfg, "qp_eps", 1e-12)),
        qp_not_improved_lim=int(
            cfg_get(actor_cfg, "qp_not_improved_lim", 3)
        ),
        qp_verbose=int(cfg_get(actor_cfg, "qp_verbose", -1)),
        qp_check_q_spd=bool(cfg_get(actor_cfg, "qp_check_q_spd", False)),
        qp_slack_weight=float(cfg_get(actor_cfg, "qp_slack_weight", 10.0)),
        qp_residual_reward_weight=float(actor_cfg.qp_residual_reward_weight),
        qp_full_diagnostics=bool(
            cfg_get(actor_cfg, "qp_full_diagnostics", False)
        ),
        history_len=history_len,
        obs_frame_dim=obs_frame_dim,
        obs_dim=obs_dim,
        num_humans=num_humans,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_latent_dim=encoder_latent_dim,
        encoder_layers=encoder_layers,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
        qp_obs_top_k=qp_obs_top_k,
    )


def _init_encoder(keys, hparams, use_init_weights):
    dims = (
        [hparams.encoder_input_dim]
        + [hparams.encoder_hidden_dim]
        * max(hparams.encoder_layers - 1, 0)
        + [hparams.encoder_latent_dim]
    )
    sqrt2 = math.sqrt(2.0)
    return [
        init_linear(key, in_dim, out_dim, sqrt2, use_init_weights)
        for key, in_dim, out_dim in zip(keys, dims[:-1], dims[1:])
    ]


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
    control_hidden_dim = int(
        cfg_get(actor_cfg, "control_hidden_dim", hidden_dim)
    )
    scalar_hidden_dim = int(
        cfg_get(actor_cfg, "scalar_hidden_dim", hidden_dim)
    )
    critic_hidden_dim = int(cfg_get(critic_cfg, "hidden_dim", 256))
    critic_hidden_dim2 = int(cfg_get(critic_cfg, "hidden_dim2", 256))
    if min(
        hidden_dim,
        control_hidden_dim,
        scalar_hidden_dim,
        critic_hidden_dim,
        critic_hidden_dim2,
    ) <= 0:
        raise ValueError(
            "diff_cvar_mlp actor and critic hidden dimensions must be > 0"
        )

    keys = jax.random.split(key, hparams.encoder_layers + 9)
    encoder_end = hparams.encoder_layers
    sqrt2 = math.sqrt(2.0)
    actor = {
        "logstd": jnp.full(
            (act_dim,),
            math.log(float(actor_cfg.action_std_init)),
            dtype=jnp.float32,
        ),
        "mlp_encoder": _init_encoder(
            keys[:encoder_end], hparams, use_init_weights
        ),
        "fc1": init_linear(
            keys[encoder_end],
            hparams.encoder_latent_dim,
            hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "nominal_head": init_head(
            keys[encoder_end + 1 : encoder_end + 3],
            hidden_dim,
            control_hidden_dim,
            act_dim,
            use_init_weights,
        ),
    }
    actor["human_fc1"] = init_linear(
        keys[encoder_end + 3],
        hparams.encoder_latent_dim,
        scalar_hidden_dim,
        sqrt2,
        use_init_weights,
    )
    actor["human_beta_head"] = init_linear(
        keys[encoder_end + 4],
        scalar_hidden_dim,
        hparams.num_humans,
        sqrt2 * 0.01,
        use_init_weights,
    )
    actor["human_rsafe_head"] = init_linear(
        keys[encoder_end + 5],
        scalar_hidden_dim,
        hparams.num_humans,
        sqrt2 * 0.01,
        use_init_weights,
    )
    actor.update({
        "gmm_weights": jnp.asarray(
            actor_cfg.gmm_weights, dtype=jnp.float32
        ),
        "gmm_variances": jnp.square(
            jnp.asarray(actor_cfg.gmm_stds, dtype=jnp.float32)
        ),
    })

    critic = {
        "fc1": init_linear(
            keys[encoder_end + 6],
            hparams.encoder_latent_dim,
            critic_hidden_dim,
            sqrt2,
            use_init_weights,
        ),
        "fc21": init_linear(
            keys[encoder_end + 7],
            critic_hidden_dim,
            critic_hidden_dim2,
            sqrt2,
            use_init_weights,
        ),
        "fc31": init_linear(
            keys[encoder_end + 8],
            critic_hidden_dim2,
            1,
            1.0,
            use_init_weights,
        ),
    }
    return {
        "actor": actor,
        "critic": critic,
        "action_low": jnp.asarray(action_low, dtype=jnp.float32),
        "action_high": jnp.asarray(action_high, dtype=jnp.float32),
    }


def _world_to_body(vectors, robot_theta):
    c = jnp.cos(robot_theta)
    s = jnp.sin(robot_theta)
    while c.ndim < vectors.ndim - 1:
        c = c[..., None]
        s = s[..., None]
    x = vectors[..., 0]
    y = vectors[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)


def _body_to_world(vectors, theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return jnp.stack([c * x - s * y, s * x + c * y], axis=1)


def _obs_history(obs, hparams):
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim != 2:
        raise ValueError(
            f"diff_cvar_mlp expects rank-2 observations, got ndim={obs.ndim}"
        )
    if obs.shape[-1] != hparams.obs_dim:
        raise ValueError(
            f"diff_cvar_mlp expects obs_dim={hparams.obs_dim}, got {obs.shape[-1]}"
        )
    return obs.reshape(
        (obs.shape[0], hparams.history_len, hparams.obs_frame_dim)
    )


def _preprocess_obs(obs, hparams):
    frames = _obs_history(obs, hparams)
    robot_theta = frames[:, :, 4]
    goal_rel = (
        _world_to_body(frames[:, :, 0:2], robot_theta)
        / hparams.position_scale
    )
    robot_vel = (
        _world_to_body(frames[:, :, 2:4], robot_theta)
        / hparams.velocity_scale
    )
    robot_radius = frames[:, :, 5:6] / hparams.position_scale

    human_blocks = frames[:, :, 6:].reshape(
        (frames.shape[0], hparams.history_len, hparams.num_humans, 6)
    )
    human_mask = jnp.clip(human_blocks[:, :, :, 5:6], 0.0, 1.0)
    human_rel = (
        _world_to_body(human_blocks[:, :, :, 0:2], robot_theta)
        / hparams.position_scale
    )
    human_vel = (
        _world_to_body(human_blocks[:, :, :, 2:4], robot_theta)
        / hparams.velocity_scale
    )
    human_radius = human_blocks[:, :, :, 4:5] / hparams.position_scale
    human_values = jnp.concatenate(
        [human_rel, human_vel, human_radius], axis=-1
    ) * human_mask
    human_features = jnp.concatenate(
        [human_values, human_mask], axis=-1
    )
    processed_frames = jnp.concatenate(
        [
            goal_rel,
            robot_vel,
            robot_radius,
            human_features.reshape(
                (frames.shape[0], hparams.history_len, -1)
            ),
        ],
        axis=-1,
    )
    return processed_frames.reshape((frames.shape[0], -1))


def _current_obs(obs, hparams):
    history = _obs_history(obs, hparams)
    return history, history[:, -1, :]


def _select_qp_topk(current_obs, human_ids, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    mask = jnp.clip(blocks[:, :, 5], 0.0, 1.0) > 0.5
    distance = jnp.linalg.norm(blocks[:, :, 0:2], axis=2)
    sort_key = jnp.where(mask, distance, jnp.inf)
    _values, topk_ids = jax.lax.top_k(
        -sort_key, hparams.qp_obs_top_k
    )
    batch_idx = jnp.arange(batch_size)[:, None]
    topk_blocks = blocks[batch_idx, topk_ids]
    selected_human_ids = human_ids[batch_idx, topk_ids]
    qp_obs = jnp.concatenate(
        [current_obs[:, :6], topk_blocks.reshape((batch_size, -1))],
        axis=1,
    )
    return qp_obs, selected_human_ids


def build_qp_context(obs, hparams):
    """Build observation-only QP context for reuse during PPO updates."""
    _history, current_obs = _current_obs(obs, hparams)
    batch_size = current_obs.shape[0]
    human_ids = jnp.broadcast_to(
        jnp.arange(hparams.num_humans, dtype=jnp.int32)[None, :],
        (batch_size, hparams.num_humans),
    )
    qp_obs, qp_topk_ids = _select_qp_topk(
        current_obs, human_ids, hparams
    )
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
    weights = weights / jnp.maximum(
        jnp.sum(weights, axis=-1, keepdims=True), 1e-8
    )
    valid_count = jnp.sum(active_float, axis=-1, keepdims=True)
    remaining = jnp.maximum(
        hparams.beta_max - hparams.beta_min * valid_count, 0.0
    )
    beta = hparams.beta_min + remaining * weights
    return jnp.where(
        active,
        beta,
        jnp.asarray(hparams.beta_min, dtype=beta_logits.dtype),
    )


def _safe_distances_from_margin_logits(current_obs, margin_logits, hparams):
    batch_size = current_obs.shape[0]
    blocks = current_obs[:, 6:].reshape(
        (batch_size, hparams.num_humans, 6)
    )
    robot_radius = current_obs[:, 5:6]
    human_radii = blocks[:, :, 4]
    margin_base = jnp.asarray(
        hparams.margin_base, dtype=current_obs.dtype
    )
    margin_extra_max = jnp.asarray(
        hparams.margin_extra_max, dtype=current_obs.dtype
    )
    learned_margin = margin_base + margin_extra_max * jnn.sigmoid(
        margin_logits
    )
    return robot_radius + human_radii + learned_margin


def _shared_latent(params, hparams, obs):
    features = _preprocess_obs(obs, hparams)
    return mlp(
        params["actor"]["mlp_encoder"], features, hparams.actor_act
    )


def _actor_outputs(params, hparams, obs, qp_context=None):
    latent = _shared_latent(params, hparams, obs)
    _history, current_obs = _current_obs(obs, hparams)
    actor_params = params["actor"]
    hidden = activation(
        dense(actor_params["fc1"], latent), hparams.actor_act
    )
    nominal_xy = mlp(
        actor_params["nominal_head"], hidden, hparams.actor_act
    )

    human_x = activation(
        dense(actor_params["human_fc1"], latent), hparams.actor_act
    )
    beta_logits_all = dense(actor_params["human_beta_head"], human_x)
    margin_logits_all = dense(actor_params["human_rsafe_head"], human_x)
    r_safe_all = _safe_distances_from_margin_logits(
        current_obs, margin_logits_all, hparams
    )

    if qp_context is None:
        qp_context = build_qp_context(obs, hparams)
    qp_obs = qp_context["qp_obs"]
    qp_topk_ids = qp_context["qp_topk_ids"]
    qp_blocks = qp_obs[:, 6:].reshape(
        (qp_obs.shape[0], hparams.qp_obs_top_k, 6)
    )
    beta_logits_topk = _gather_topk(beta_logits_all, qp_topk_ids)
    beta_topk = _allocate_beta_budget(
        beta_logits_topk, qp_blocks[:, :, 5], hparams
    )
    r_safe_topk = _gather_topk(r_safe_all, qp_topk_ids)
    nominal_xy = _body_to_world(nominal_xy, qp_obs[:, 4])
    safe_xy, slack = jax_cvar_qp.solve_slack_qp(
        params,
        hparams,
        qp_obs,
        nominal_xy,
        beta_topk,
        r_safe_topk,
    )
    return (
        latent,
        qp_obs,
        nominal_xy,
        beta_topk,
        r_safe_topk,
        safe_xy,
        slack,
    )


def safe_action_mean(params, hparams, obs, qp_context=None):
    outputs = _actor_outputs(
        params, hparams, obs, qp_context=qp_context
    )
    return outputs[-2]


policy_action_mean = safe_action_mean
policy_action_to_env_action = actor_critic.policy_action_to_env_action


def _critic_value_from_latent(params, hparams, latent):
    hidden = activation(
        dense(params["critic"]["fc1"], latent), hparams.critic_act
    )
    hidden = activation(
        dense(params["critic"]["fc21"], hidden), hparams.critic_act
    )
    return dense(params["critic"]["fc31"], hidden).squeeze(-1)


def critic_value(params, hparams, obs):
    latent = _shared_latent(params, hparams, obs)
    return _critic_value_from_latent(params, hparams, latent)


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
    (
        latent,
        qp_obs,
        nominal_xy,
        beta,
        r_safe,
        safe_xy,
        slack,
    ) = _actor_outputs(params, hparams, obs, qp_context=qp_context)
    action, logprob, entropy, sampled_policy_action = (
        actor_critic.action_logprob_entropy(
            params,
            safe_xy,
            action=action,
            noise=noise,
            policy_action=policy_action,
        )
    )
    value = _critic_value_from_latent(params, hparams, latent)
    outputs = (action, logprob, entropy, value)
    if return_policy_action:
        outputs = outputs + (sampled_policy_action,)
    if return_diagnostics:
        if hparams.qp_full_diagnostics:
            diagnostics = jax_cvar_qp.full_diagnostics(
                params,
                hparams,
                qp_obs,
                nominal_xy,
                beta,
                r_safe,
                safe_xy,
                slack,
            )
        else:
            diagnostics = jax_cvar_qp.lightweight_diagnostics(
                params,
                hparams,
                qp_obs,
                nominal_xy,
                beta,
                r_safe,
                safe_xy,
                slack,
            )
        outputs = outputs + (diagnostics,)
    return outputs
