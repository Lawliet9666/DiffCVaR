"""Deterministic evaluation for the supported JAX GNN policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from env.jax_social_nav import config_from_omegaconf, fixed_order_policy_obs_from_state, reset_env_with_keys, step_env


DEFAULT_EVAL_SEEDS = tuple(range(100, 1001, 100))


def _eval_seed_values(episodes, seeds=None):
    base_seeds = DEFAULT_EVAL_SEEDS if seeds is None else tuple(int(seed) for seed in seeds)
    if not base_seeds:
        raise ValueError("at least one evaluation seed is required")
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    seed_values = []
    for seed in base_seeds:
        for episode in range(episodes):
            seed_values.append(int(seed) + episode)
    return np.asarray(seed_values, dtype=np.uint32)


def _select_tree_by_active(active, old_tree, new_tree):
    def select(old_value, new_value):
        shape = (active.shape[0],) + (1,) * (new_value.ndim - 1)
        return jnp.where(active.reshape(shape), new_value, old_value)

    return jax.tree_util.tree_map(select, old_tree, new_tree)


def _default_policy_functions():
    from model.jax_diff_cvar_gnn import policy_action_mean, policy_action_to_env_action

    return policy_action_mean, policy_action_to_env_action


def _evaluate_seed_values(
    params,
    hparams,
    env_cfg,
    seed_values,
    policy_action_mean_fn=None,
    policy_action_to_env_action_fn=None,
):
    """Evaluate all requested episodes as one batched JAX env rollout."""
    if policy_action_mean_fn is None or policy_action_to_env_action_fn is None:
        default_mean_fn, default_action_fn = _default_policy_functions()
        policy_action_mean_fn = policy_action_mean_fn or default_mean_fn
        policy_action_to_env_action_fn = policy_action_to_env_action_fn or default_action_fn

    keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seed_values], axis=0)
    state = reset_env_with_keys(env_cfg, keys)
    batch_size = int(seed_values.shape[0])
    frame = fixed_order_policy_obs_from_state(env_cfg, state)
    history = jnp.repeat(frame[:, None, :], int(hparams.history_len), axis=1)

    init_done = jnp.zeros((batch_size,), dtype=bool)
    init_returns = jnp.zeros((batch_size,), dtype=jnp.float32)
    init_success = jnp.zeros((batch_size,), dtype=bool)
    init_collision = jnp.zeros((batch_size,), dtype=bool)
    init_timeout = jnp.zeros((batch_size,), dtype=bool)
    init_key = jax.random.PRNGKey(0)
    max_steps = jnp.asarray(int(env_cfg.max_steps), dtype=jnp.int32)
    init_step = jnp.asarray(0, dtype=jnp.int32)

    def cond(carry):
        _state, _history, done, _returns, _success, _collision, _timeout, _key, step = carry
        return (step < max_steps) & (~jnp.all(done))

    def body(carry):
        curr_state, curr_history, done, returns, success, collision, timeout, key, step = carry
        key, step_key = jax.random.split(key)
        policy_obs = curr_history.reshape((batch_size, -1))
        policy_action = policy_action_mean_fn(params, hparams, policy_obs)
        env_action = policy_action_to_env_action_fn(params, policy_action)
        next_state, reward, step_done, metrics = step_env(env_cfg, curr_state, env_action, step_key)

        active = ~done
        next_done = done | (active & (step_done > 0.5))
        returns = returns + jnp.where(active, reward, 0.0)
        success = success | (active & (metrics["success"] > 0.5))
        collision = collision | (active & (metrics["collision"] > 0.5))
        timeout = timeout | (active & (metrics["timeout"] > 0.5))
        selected_state = _select_tree_by_active(active, curr_state, next_state)
        next_frame = fixed_order_policy_obs_from_state(env_cfg, selected_state)
        shifted_history = jnp.concatenate([curr_history[:, 1:, :], next_frame[:, None, :]], axis=1)
        next_history = jnp.where(active[:, None, None], shifted_history, curr_history)
        return selected_state, next_history, next_done, returns, success, collision, timeout, key, step + 1

    _state, _history, _done, returns, success, collision, timeout, _key, _step = jax.lax.while_loop(
        cond,
        body,
        (
            state,
            history,
            init_done,
            init_returns,
            init_success,
            init_collision,
            init_timeout,
            init_key,
            init_step,
        ),
    )
    return returns, success, collision, timeout


def evaluate_jax_env(
    params,
    hparams,
    env_cfg_or_config,
    episodes=None,
    seeds=None,
    policy_action_mean_fn=None,
    safe_action_mean_fn=None,
    policy_action_to_env_action_fn=None,
    history_len=None,
):
    """Run deterministic evaluation using fixed-ID observation history."""
    env_cfg = env_cfg_or_config
    if not hasattr(env_cfg_or_config, "max_steps"):
        env_cfg = config_from_omegaconf(env_cfg_or_config)
    if policy_action_mean_fn is None and safe_action_mean_fn is not None:
        policy_action_mean_fn = safe_action_mean_fn

    seed_values = _eval_seed_values(1 if episodes is None else episodes, seeds=seeds)
    returns, success, collision, timeout = _evaluate_seed_values(
        params,
        hparams,
        env_cfg,
        seed_values,
        policy_action_mean_fn=policy_action_mean_fn,
        policy_action_to_env_action_fn=policy_action_to_env_action_fn,
    )
    returns_np = np.asarray(returns, dtype=np.float32)
    success_np = np.asarray(success, dtype=np.float32)
    collision_np = np.asarray(collision, dtype=np.float32)
    timeout_np = np.asarray(timeout, dtype=np.float32)
    total = int(returns_np.size)
    return {
        "mean_return": float(np.mean(returns_np)),
        "std_return": float(np.std(returns_np)),
        "success_rate": float(success_np.sum()) / total,
        "collision_rate": float(collision_np.sum()) / total,
        "timeout_rate": float(timeout_np.sum()) / total,
    }
