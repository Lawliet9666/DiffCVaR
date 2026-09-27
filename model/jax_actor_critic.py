"""Shared squashed-Gaussian actor-critic helpers for JAX PPO models."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def actor_std(params):
    logstd = params["actor"]["logstd"]
    logstd_min = math.log(1e-6)
    logstd_max = math.log(10.0)
    in_bounds = jnp.all((logstd >= logstd_min) & (logstd <= logstd_max))
    return jax.lax.cond(
        in_bounds,
        lambda value: jnp.clip(jnp.exp(value), 1e-6, 10.0),
        lambda value: jnp.exp(jnp.clip(value, logstd_min, logstd_max)),
        logstd,
    )


def action_scale_bias(params):
    action_low = jnp.asarray(params["action_low"], dtype=jnp.float32)
    action_high = jnp.asarray(params["action_high"], dtype=jnp.float32)
    return 0.5 * (action_high - action_low), 0.5 * (action_high + action_low)


def base_logprob_entropy_from_policy_action(params, mean, policy_action):
    policy_action = jnp.asarray(policy_action, dtype=mean.dtype)
    std = actor_std(params).astype(mean.dtype)
    log_std = jnp.log(std)
    log_prob = -0.5 * (((policy_action - mean) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
    return jnp.sum(log_prob, axis=-1), jnp.broadcast_to(jnp.sum(entropy), (mean.shape[0],))


def policy_action_to_env_action(params, policy_action, return_squashed=False):
    scale, bias = action_scale_bias(params)
    policy_action = jnp.asarray(policy_action, dtype=jnp.float32)
    squashed = jnp.tanh(policy_action)
    action = bias + scale * squashed
    if return_squashed:
        return action, squashed
    return action


def env_action_to_policy_action(params, action):
    scale, bias = action_scale_bias(params)
    action = jnp.asarray(action, dtype=jnp.float32)
    squashed = (action - bias) / jnp.maximum(scale, 1e-6)
    squashed = jnp.clip(squashed, -1.0 + 1e-6, 1.0 - 1e-6)
    policy_action = 0.5 * (jnp.log1p(squashed) - jnp.log1p(-squashed))
    return policy_action, squashed


def squash_log_prob(params, base_log_prob, squashed):
    scale, _bias = action_scale_bias(params)
    tanh_logdet = jnp.log(1.0 - jnp.square(squashed) + 1e-6).sum(axis=-1)
    scale_logdet = jnp.log(jnp.maximum(scale, 1e-6)).sum()
    return base_log_prob - tanh_logdet - scale_logdet


def action_logprob_entropy(
    params,
    mean,
    action=None,
    noise=None,
    policy_action=None,
):
    """Evaluate or sample the shared squashed-Gaussian policy distribution."""
    sampled_policy_action = policy_action
    if action is None:
        if policy_action is not None:
            action, squashed = policy_action_to_env_action(params, policy_action, return_squashed=True)
            base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, policy_action)
            logprob = squash_log_prob(params, base_logprob, squashed)
        elif noise is not None:
            noise = jnp.asarray(noise, dtype=mean.dtype)
            sampled_policy_action = mean + actor_std(params).astype(mean.dtype) * noise
            action, squashed = policy_action_to_env_action(params, sampled_policy_action, return_squashed=True)
            base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, sampled_policy_action)
            logprob = squash_log_prob(params, base_logprob, squashed)
        else:
            raise ValueError("noise or policy_action must be provided when action is None")
    elif policy_action is not None:
        _action, squashed = policy_action_to_env_action(params, policy_action, return_squashed=True)
        base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, policy_action)
        logprob = squash_log_prob(params, base_logprob, squashed)
    else:
        inferred_policy_action, squashed = env_action_to_policy_action(params, action)
        base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, inferred_policy_action)
        logprob = squash_log_prob(params, base_logprob, squashed)
    return action, logprob, entropy, sampled_policy_action
