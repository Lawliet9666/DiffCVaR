"""qpth-like dense QPFunction implemented with JAX and custom VJP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError
import numpy as np

from .autograd import qpth_backward
from .pdipm import forward as pdipm_forward
from .util import expand_qp_params, reduce_grad


def _check_q_spd(Q):
    chol = jnp.linalg.cholesky(Q)
    ok = jnp.all(jnp.isfinite(chol))
    try:
        ok_bool = bool(ok)
    except TracerBoolConversionError:
        def _raise_if_not_spd(value):
            if not bool(np.asarray(value)):
                raise RuntimeError("Q is not SPD.")

        jax.debug.callback(_raise_if_not_spd, ok)
        return
    if not ok_bool:
        raise RuntimeError("Q is not SPD.")


def solve_qp(Q, p, G, h, A, b, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, check_Q_spd=True):
    expanded = expand_qp_params(Q, p, G, h, A, b)
    if expanded.A.shape[1] == 0 and expanded.G.shape[1] == 0:
        raise AssertionError("at least one equality or inequality constraint is required")
    if check_Q_spd:
        _check_q_spd(expanded.Q)
    return pdipm_forward(
        expanded.Q,
        expanded.p,
        expanded.G,
        expanded.h,
        expanded.A,
        expanded.b,
        eps=eps,
        verbose=verbose,
        notImprovedLim=notImprovedLim,
        maxIter=maxIter,
    )


def QPFunction(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, check_Q_spd=True):
    """Return a qpth-like differentiable dense QP solve function."""

    @jax.custom_vjp
    def _solve(Q, p, G, h, A, b):
        return solve_qp(
            Q,
            p,
            G,
            h,
            A,
            b,
            eps=eps,
            verbose=verbose,
            notImprovedLim=notImprovedLim,
            maxIter=maxIter,
            check_Q_spd=check_Q_spd,
        )[0]

    def _solve_fwd(Q, p, G, h, A, b):
        expanded = expand_qp_params(Q, p, G, h, A, b)
        if check_Q_spd:
            _check_q_spd(expanded.Q)
        zhats, nus, lams, slacks = pdipm_forward(
            expanded.Q,
            expanded.p,
            expanded.G,
            expanded.h,
            expanded.A,
            expanded.b,
            eps=eps,
            verbose=verbose,
            notImprovedLim=notImprovedLim,
            maxIter=maxIter,
        )
        expanded_values = (expanded.Q, expanded.p, expanded.G, expanded.h, expanded.A, expanded.b)
        expanded_flags = (
            expanded.Q_expanded,
            expanded.p_expanded,
            expanded.G_expanded,
            expanded.h_expanded,
            expanded.A_expanded,
            expanded.b_expanded,
        )
        residual = (
            Q,
            p,
            G,
            h,
            A,
            b,
            expanded_values,
            expanded_flags,
            zhats,
            lams,
            slacks,
            nus,
        )
        return zhats, residual

    def _solve_bwd(residual, dl_dzhat):
        (
            Q_orig,
            p_orig,
            G_orig,
            h_orig,
            A_orig,
            b_orig,
            expanded_values,
            expanded_flags,
            zhats,
            lams,
            slacks,
            nus,
        ) = residual
        Q_exp, p_exp, G_exp, h_exp, A_exp, b_exp = expanded_values
        Q_e, p_e, G_e, h_e, A_e, b_e = expanded_flags
        dQs, dps, dGs, dhs, dAs, dbs = qpth_backward(
            Q_exp,
            p_exp,
            G_exp,
            h_exp,
            A_exp,
            b_exp,
            zhats,
            lams,
            slacks,
            nus,
            dl_dzhat,
        )
        return (
            reduce_grad(dQs, Q_orig, Q_e),
            reduce_grad(dps, p_orig, p_e),
            reduce_grad(dGs, G_orig, G_e),
            reduce_grad(dhs, h_orig, h_e),
            reduce_grad(dAs, A_orig, A_e),
            reduce_grad(dbs, b_orig, b_e),
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve
