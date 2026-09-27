"""qpth-compatible implicit differentiation for the JAX QP layer."""

from __future__ import annotations

import jax.numpy as jnp

from .pdipm import factor_kkt, pre_factor_kkt, solve_kkt_factored
from .util import bger


def qpth_backward(Q, p, G, h, A, b, zhats, lams, slacks, nus, dl_dzhat):
    """Return qpth-style gradients for Q, p, G, h, A, b."""
    del h
    d = jnp.clip(lams, min=1e-8) / jnp.clip(slacks, min=1e-8)
    n_batch, nineq, _nz = G.shape
    neq = A.shape[1]
    zeros_ineq = jnp.zeros((n_batch, nineq), dtype=G.dtype)
    zeros_eq = jnp.zeros((n_batch, neq), dtype=G.dtype)
    kkt_factors = pre_factor_kkt(Q, G, A)
    S_LU = factor_kkt(kkt_factors, d, neq)
    dx, _ds, dlam, dnu = solve_kkt_factored(kkt_factors, d, G, A, S_LU, dl_dzhat, zeros_ineq, zeros_ineq, zeros_eq)

    dps = dx
    dGs = bger(dlam, zhats) + bger(lams, dx)
    dhs = -dlam
    dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
    if neq > 0:
        dAs = bger(dnu, zhats) + bger(nus, dx)
        dbs = -dnu
    else:
        dAs = jnp.zeros_like(A)
        dbs = jnp.zeros_like(b)
    return dQs, dps, dGs, dhs, dAs, dbs
