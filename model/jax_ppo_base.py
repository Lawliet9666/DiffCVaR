"""Shared JAX MLP building blocks for PPO-style models."""

from __future__ import annotations

import math
import numpy as np

import jax
import jax.nn as jnn
import jax.numpy as jnp


def orthogonal_linear(key, in_dim, out_dim, gain):
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    flat_shape = (out_dim, in_dim)
    matrix_shape = (in_dim, out_dim) if out_dim < in_dim else flat_shape
    matrix = np.asarray(jax.random.normal(key, matrix_shape, dtype=jnp.float32))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    if out_dim < in_dim:
        q = q.T
    weight = jnp.asarray(float(gain) * q[:out_dim, :in_dim], dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight, "bias": bias}


def torch_default_linear(key, in_dim, out_dim):
    weight_key, bias_key = jax.random.split(key)
    bound = 1.0 / math.sqrt(float(in_dim))
    return {
        "weight": jax.random.uniform(
            weight_key,
            (int(out_dim), int(in_dim)),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
        "bias": jax.random.uniform(
            bias_key,
            (int(out_dim),),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
    }


def init_linear(key, in_dim, out_dim, gain, use_init_weights):
    if bool(use_init_weights):
        return orthogonal_linear(key, in_dim, out_dim, gain)
    return torch_default_linear(key, in_dim, out_dim)


def dense(layer, x):
    """Apply a small linear layer without lowering it as a GEMM."""
    return jnp.sum(x[..., None, :] * layer["weight"], axis=-1) + layer["bias"]


def mlp(layers, x, act_name):
    for layer in layers[:-1]:
        x = activation(dense(layer, x), act_name)
    return dense(layers[-1], x)


def init_head(keys, in_dim, hidden_dim, out_dim, use_init_weights):
    sqrt2 = math.sqrt(2.0)
    k1, k2 = keys
    return [
        init_linear(k1, in_dim, hidden_dim, sqrt2, use_init_weights),
        init_linear(k2, hidden_dim, out_dim, sqrt2 * 0.01, use_init_weights),
    ]


def cfg_get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def require_config_value(cfg, name, expected, path):
    value = cfg_get(cfg, name, expected)
    if value != expected:
        raise ValueError(f"{path}.{name} is fixed to {expected!r}; got {value!r}")


def activation(x, act_name):
    act_name = str(act_name)
    if act_name == "relu":
        return jnn.relu(x)
    if act_name == "tanh":
        return jnp.tanh(x)
    if act_name == "silu":
        return jnn.silu(x)
    raise ValueError(f"unsupported activation: {act_name}")
