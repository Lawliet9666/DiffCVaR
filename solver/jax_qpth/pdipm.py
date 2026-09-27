"""Dense batched primal-dual interior-point solver with qpth-style semantics."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.errors import TracerBoolConversionError
import jax
import numpy as np


INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solution.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the CVXPY solver to see if your problem is feasible
and you can use the verbose option to check the convergence status of
our solver while increasing the number of iterations.

Advanced users:
You can also try to enable iterative refinement in the solver:
https://github.com/locuslab/qpth/issues/6
--------
"""


def _lu_factor(mat):
    return jsp_linalg.lu_factor(mat, check_finite=False)


def _lu_solve_vec(lu_and_piv, vec):
    return jsp_linalg.lu_solve(lu_and_piv, vec[..., None], check_finite=False)[..., 0]


def _lu_solve_mat(lu_and_piv, rhs):
    return jsp_linalg.lu_solve(lu_and_piv, rhs, check_finite=False)


def _bmv(mat, vec):
    return jnp.einsum("bij,bj->bi", mat, vec)


def _btmv(mat, vec):
    return jnp.einsum("bij,bi->bj", mat, vec)


def _batch_diag(values):
    return jnp.einsum("bi,ij->bij", values, jnp.eye(values.shape[1], dtype=values.dtype))


def pre_factor_kkt(Q, G, A):
    """Precompute the qpth KKT factors that do not depend on z / s."""
    Q_LU = _lu_factor(Q)
    invQ_GT = _lu_solve_mat(Q_LU, jnp.swapaxes(G, 1, 2))
    G_invQ_GT = jnp.matmul(G, invQ_GT)
    neq = A.shape[1]
    if neq == 0:
        return Q_LU, G_invQ_GT

    invQ_AT = _lu_solve_mat(Q_LU, jnp.swapaxes(A, 1, 2))
    A_invQ_AT = jnp.matmul(A, invQ_AT)
    A_invQ_GT = jnp.matmul(A, invQ_GT)
    G_invQ_AT = jnp.matmul(G, invQ_AT)
    top = jnp.concatenate((A_invQ_AT, A_invQ_GT), axis=2)
    bottom = jnp.concatenate((G_invQ_AT, G_invQ_GT), axis=2)
    return Q_LU, jnp.concatenate((top, bottom), axis=1)


def factor_kkt(kkt_factors, d, neq):
    """Complete the qpth Schur complement factorization for the current d."""
    _Q_LU, schur_base = kkt_factors
    nineq = d.shape[1]
    diag_inv_d = _batch_diag(1.0 / d)
    if neq == 0:
        schur = schur_base + diag_inv_d
    else:
        update = jnp.zeros_like(schur_base)
        update = update.at[:, neq : neq + nineq, neq : neq + nineq].set(diag_inv_d)
        schur = schur_base + update
    return _lu_factor(schur)


def solve_kkt_factored(kkt_factors, d, G, A, S_LU, rx, rs, rz, ry):
    """Solve qpth's LU-partial KKT system from cached Q and Schur factors."""
    Q_LU, _schur_base = kkt_factors
    neq = A.shape[1]
    invQ_rx = _lu_solve_vec(Q_LU, rx)
    h_ineq = _bmv(G, invQ_rx) + rs / d - rz
    if neq > 0:
        h_eq = _bmv(A, invQ_rx) - ry
        rhs = jnp.concatenate((h_eq, h_ineq), axis=1)
    else:
        rhs = h_ineq

    w = _lu_solve_vec(S_LU, -rhs)
    w_eq = w[:, :neq] if neq > 0 else jnp.zeros((G.shape[0], 0), dtype=G.dtype)
    w_ineq = w[:, neq:]

    g1 = -rx - _btmv(G, w_ineq)
    if neq > 0:
        g1 = g1 - _btmv(A, w_eq)
    g2 = -rs - w_ineq

    dx = _lu_solve_vec(Q_LU, g1)
    ds = g2 / d
    dz = w_ineq
    dy = w_eq
    return dx, ds, dz, dy


def get_step(v, dv):
    a = -v / dv
    global_max = jnp.max(a)
    step_cap = jnp.where(global_max > 1.0, global_max, 1.0)
    a = jnp.where(dv > 0.0, step_cap, a)
    return jnp.min(a, axis=1)


def residuals(Q, p, G, h, A, b, x, s, z, y):
    neq = A.shape[1]
    rx = _btmv(G, z) + _bmv(jnp.swapaxes(Q, 1, 2), x) + p
    if neq > 0:
        rx = rx + _btmv(A, y)
    rz = _bmv(G, x) + s - h
    ry = _bmv(A, x) - b if neq > 0 else jnp.zeros((Q.shape[0], 0), dtype=Q.dtype)
    mu = jnp.abs(jnp.sum(s * z, axis=1) / G.shape[1])
    z_resid = jnp.linalg.norm(rz, axis=1)
    y_resid = jnp.linalg.norm(ry, axis=1) if neq > 0 else jnp.zeros_like(z_resid)
    pri_resid = y_resid + z_resid
    dual_resid = jnp.linalg.norm(rx, axis=1)
    resids = pri_resid + dual_resid + G.shape[1] * mu
    return rx, rz, ry, mu, pri_resid, dual_resid, resids


def forward(Q, p, G, h, A, b, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    if maxIter <= 0:
        raise ValueError("maxIter must be > 0")
    n_batch, nineq, _nz = G.shape
    neq = A.shape[1]
    dtype = Q.dtype
    kkt_factors = pre_factor_kkt(Q, G, A)

    d0 = jnp.ones((n_batch, nineq), dtype=dtype)
    S_LU0 = factor_kkt(kkt_factors, d0, neq)
    x0, s0, z0, y0 = solve_kkt_factored(
        kkt_factors,
        d0,
        G,
        A,
        S_LU0,
        p,
        jnp.zeros((n_batch, nineq), dtype=dtype),
        -h,
        -b if neq > 0 else jnp.zeros((n_batch, 0), dtype=dtype),
    )

    min_s = jnp.min(s0, axis=1, keepdims=True)
    s0 = jnp.where(min_s < 0.0, s0 - min_s + 1.0, s0)
    min_z = jnp.min(z0, axis=1, keepdims=True)
    z0 = jnp.where(min_z < 0.0, z0 - min_z + 1.0, z0)

    initial_carry = (
        x0,
        s0,
        z0,
        y0,
        x0,
        s0,
        z0,
        y0,
        jnp.full((n_batch,), jnp.inf, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
    )

    def iteration(_iteration, carry):
        (
            x,
            s,
            z,
            y,
            best_x,
            best_s,
            best_z,
            best_y,
            best_resids,
            n_not_improved,
            active,
        ) = carry
        rx, rz, ry, mu, _pri, _dual, resids = residuals(Q, p, G, h, A, b, x, s, z, y)
        d = z / s
        S_LU = factor_kkt(kkt_factors, d, neq)

        improved = active & (resids < best_resids)
        any_improved = jnp.sum(improved) > 0
        next_n_not_improved = jnp.where(any_improved, 0, n_not_improved + 1)

        next_best_resids = jnp.where(improved, resids, best_resids)
        next_best_x = jnp.where(improved[:, None], x, best_x)
        next_best_s = jnp.where(improved[:, None], s, best_s)
        next_best_z = jnp.where(improved[:, None], z, best_z)
        next_best_y = jnp.where(improved[:, None], y, best_y)

        reached_not_improved = next_n_not_improved == notImprovedLim
        reached_eps = jnp.max(next_best_resids) < eps
        reached_mu = jnp.min(mu) > 1e32
        should_step = active & ~(reached_not_improved | reached_eps | reached_mu)

        dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt_factored(kkt_factors, d, G, A, S_LU, rx, z, rz, ry)
        alpha_aff = jnp.minimum(
            jnp.minimum(get_step(z, dz_aff), get_step(s, ds_aff)),
            jnp.ones((n_batch,), dtype=dtype),
        )
        alpha_aff = alpha_aff[:, None]
        t1 = s + alpha_aff * ds_aff
        t2 = z + alpha_aff * dz_aff
        sig = (jnp.sum(t1 * t2, axis=1) / jnp.sum(s * z, axis=1)) ** 3

        rx_cor = jnp.zeros_like(x)
        rs_cor = ((-mu * sig)[:, None] + ds_aff * dz_aff) / s
        rz_cor = jnp.zeros_like(z)
        ry_cor = jnp.zeros_like(y)
        dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt_factored(kkt_factors, d, G, A, S_LU, rx_cor, rs_cor, rz_cor, ry_cor)
        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor
        alpha = jnp.minimum(
            0.999 * jnp.minimum(get_step(z, dz), get_step(s, ds)),
            jnp.ones((n_batch,), dtype=dtype),
        )
        alpha = alpha[:, None]
        x_candidate = x + alpha * dx
        s_candidate = s + alpha * ds
        z_candidate = z + alpha * dz
        y_candidate = y + alpha * dy

        return (
            jnp.where(should_step, x_candidate, x),
            jnp.where(should_step, s_candidate, s),
            jnp.where(should_step, z_candidate, z),
            jnp.where(should_step, y_candidate, y),
            jnp.where(active, next_best_x, best_x),
            jnp.where(active, next_best_s, best_s),
            jnp.where(active, next_best_z, best_z),
            jnp.where(active, next_best_y, best_y),
            jnp.where(active, next_best_resids, best_resids),
            jnp.where(active, next_n_not_improved, n_not_improved),
            should_step,
        )

    (
        _x,
        _s,
        _z,
        _y,
        best_x,
        best_s,
        best_z,
        best_y,
        best_resids,
        _n_not_improved,
        _active,
    ) = jax.lax.fori_loop(0, maxIter, iteration, initial_carry, unroll=False)

    if verbose >= 0:
        inaccurate = jnp.max(best_resids) > 1.0
        try:
            inaccurate_bool = bool(inaccurate)
        except TracerBoolConversionError:
            def _print_if_inaccurate(value):
                if bool(np.asarray(value)):
                    print(INACC_ERR)

            jax.debug.callback(_print_if_inaccurate, inaccurate)
        else:
            if inaccurate_bool:
                print(INACC_ERR)
    return best_x, best_y, best_z, best_s
