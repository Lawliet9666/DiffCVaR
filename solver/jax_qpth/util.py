"""Small qpth-compatible shape helpers for dense batched QPs."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ExpandedQP:
    Q: jnp.ndarray
    p: jnp.ndarray
    G: jnp.ndarray
    h: jnp.ndarray
    A: jnp.ndarray
    b: jnp.ndarray
    Q_expanded: bool
    p_expanded: bool
    G_expanded: bool
    h_expanded: bool
    A_expanded: bool
    b_expanded: bool
    A_empty: bool
    b_empty: bool


def extract_n_batch(Q, p, G, h, A, b) -> int:
    dims = (3, 2, 3, 2, 3, 2)
    for param, expected_dim in zip((Q, p, G, h, A, b), dims):
        arr = jnp.asarray(param)
        if arr.ndim == expected_dim:
            return int(arr.shape[0])
    return 1


def expand_param(x, n_batch: int, n_dim: int):
    arr = jnp.asarray(x)
    if arr.ndim in (0, n_dim) or arr.size == 0:
        return arr, False
    if arr.ndim == n_dim - 1:
        return jnp.broadcast_to(jnp.expand_dims(arr, 0), (n_batch, *arr.shape)), True
    raise ValueError(f"unexpected number of dimensions: got {arr.ndim}, expected {n_dim} or {n_dim - 1}")


def expand_qp_params(Q, p, G, h, A, b) -> ExpandedQP:
    n_batch = extract_n_batch(Q, p, G, h, A, b)
    Q_exp, Q_e = expand_param(Q, n_batch, 3)
    p_exp, p_e = expand_param(p, n_batch, 2)
    G_exp, G_e = expand_param(G, n_batch, 3)
    h_exp, h_e = expand_param(h, n_batch, 2)

    if G_exp.ndim != 3:
        raise ValueError("G must resolve to shape (batch, nineq, nz)")
    nz = int(G_exp.shape[2])

    A_arr = jnp.asarray(A)
    b_arr = jnp.asarray(b)
    A_empty = bool(A_arr.size == 0)
    b_empty = bool(b_arr.size == 0)
    if A_empty:
        A_exp = jnp.zeros((n_batch, 0, nz), dtype=Q_exp.dtype)
        A_e = False
    else:
        A_exp, A_e = expand_param(A_arr, n_batch, 3)
    if b_empty:
        b_exp = jnp.zeros((n_batch, 0), dtype=Q_exp.dtype)
        b_e = False
    else:
        b_exp, b_e = expand_param(b_arr, n_batch, 2)

    return ExpandedQP(
        Q=Q_exp,
        p=p_exp,
        G=G_exp,
        h=h_exp,
        A=A_exp,
        b=b_exp,
        Q_expanded=Q_e,
        p_expanded=p_e,
        G_expanded=G_e,
        h_expanded=h_e,
        A_expanded=A_e,
        b_expanded=b_e,
        A_empty=A_empty,
        b_empty=b_empty,
    )


def reduce_grad(grad, original, expanded: bool):
    original = jnp.asarray(original)
    if original.size == 0:
        return jnp.zeros_like(original)
    if expanded:
        return jnp.mean(grad, axis=0)
    return grad


def bger(x, y):
    return jnp.einsum("bi,bj->bij", x, y)
