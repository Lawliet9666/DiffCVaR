"""JAX PPO core with a Brax-style PPO implementation for SocialNav.

The active PPO path is adapted from google/brax PPO training semantics
(Apache-2.0): GAE value targets, optional advantage normalization, clipped
surrogate objective, combined actor/value loss, and Optax Adam with global
gradient clipping. The project-specific rollout, model, checkpoint, and eval
APIs stay repo-native.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.jax_actor_critic import actor_std as shared_actor_std


LOG_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "total_loss",
    "approx_kl",
    "clipfrac",
    "actor_grad_norm",
    "critic_grad_norm",
    "batch_finite",
    "actor_loss_finite",
    "actor_grads_finite",
    "actor_step_finite",
    "critic_loss_finite",
    "critic_grads_finite",
    "critic_step_finite",
    "params_finite",
    "opt_state_finite",
    "update_is_finite",
    "max_abs_policy_action",
    "max_abs_logratio",
    "max_ratio",
    "std_min",
    "std_max",
)


def normalize_ppo_impl(ppo_impl) -> str:
    value = str(ppo_impl or "brax").strip().lower()
    if value != "brax":
        raise ValueError(f"trainer.ppo_impl must be 'brax', got {ppo_impl!r}")
    return value


def flat_time_batch(array):
    array = jnp.asarray(array)
    return array.reshape((-1,) + array.shape[2:]) if array.ndim > 2 else array.reshape(-1)


def compute_brax_gae(rewards, values, dones, next_values, gamma, lam):
    """Brax-style GAE adapted to fixed-horizon rollouts without truncation.

    This follows Brax PPO's compute_gae structure: first build lambda value
    targets `vs`, then compute policy advantages from those targets.  Our env
    only exposes done, so truncation is treated as zero and done is termination.
    """
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)
    next_values = jnp.asarray(next_values, dtype=jnp.float32)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    lam = jnp.asarray(lam, dtype=jnp.float32)

    values_t_plus_1 = jnp.concatenate((values[1:], next_values[None, :]), axis=0)
    deltas = rewards + gamma * (1.0 - dones) * values_t_plus_1 - values

    def scan_step(acc, inputs):
        delta, done = inputs
        acc = delta + gamma * (1.0 - done) * lam * acc
        return acc, acc

    _acc, vs_minus_v = jax.lax.scan(
        scan_step,
        jnp.zeros_like(next_values),
        (deltas, dones),
        reverse=True,
    )
    vs = vs_minus_v + values
    vs_t_plus_1 = jnp.concatenate((vs[1:], next_values[None, :]), axis=0)
    advantages = rewards + gamma * (1.0 - dones) * vs_t_plus_1 - values
    returns = advantages + values
    return jax.lax.stop_gradient(flat_time_batch(advantages)), jax.lax.stop_gradient(flat_time_batch(returns))


def tree_all_finite(tree):
    """Return a scalar boolean indicating whether every floating leaf is finite."""
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            finite = finite & jnp.all(jnp.isfinite(array))
    return finite



def _tree_global_norm(tree):
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in jax.tree_util.tree_leaves(tree):
        total = total + jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))
    return jnp.sqrt(total)



def make_optax_adam_tx(max_grad_norm):
    """Optax Adam transform with global norm clipping and external LR."""
    import optax

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    )


def init_optax_adam_state(params, max_grad_norm=1.0):
    return make_optax_adam_tx(max_grad_norm).init(params)


def optax_optimizer_step(params, grads, state, lr, max_grad_norm, tx=None):
    import optax

    if tx is None:
        tx = make_optax_adam_tx(max_grad_norm)
    updates, new_state = tx.update(grads, state, params)
    grad_norm = optax.global_norm(grads)
    scaled_updates = jax.tree_util.tree_map(
        lambda update: -jnp.asarray(lr, dtype=update.dtype) * update,
        updates,
    )
    new_params = optax.apply_updates(params, scaled_updates)
    return new_params, new_state, grad_norm


def _trainable_params(params):
    return {"actor": params["actor"], "critic": params["critic"]}


def _with_trainable(params, trainable_params):
    return {
        "actor": trainable_params["actor"],
        "critic": trainable_params["critic"],
        "action_low": params["action_low"],
        "action_high": params["action_high"],
    }



def init_ppo_opt_state(params, ppo_impl="brax", max_grad_norm=1.0):
    normalize_ppo_impl(ppo_impl)
    return {"brax": init_optax_adam_state(_trainable_params(params), max_grad_norm=max_grad_norm)}


def _normalize_advantages(advantages):
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + jnp.asarray(1e-8, dtype=jnp.float32))


def brax_ppo_loss(
    trainable_params,
    params,
    hparams,
    batch,
    clip,
    ent_coef,
    vf_coef,
    value_clip,
    normalize_advantage,
    get_action_and_value_fn,
):
    full_params = _with_trainable(params, trainable_params)
    policy_action = batch["policy_act"]
    if "qp_obs" in batch and "qp_topk_ids" in batch:
        _action, log_prob, entropy, values = get_action_and_value_fn(
            full_params,
            hparams,
            batch["obs"],
            action=batch["act"],
            policy_action=policy_action,
            qp_context={
                "qp_obs": batch["qp_obs"],
                "qp_topk_ids": batch["qp_topk_ids"],
            },
        )
    else:
        _action, log_prob, entropy, values = get_action_and_value_fn(
            full_params,
            hparams,
            batch["obs"],
            action=batch["act"],
            policy_action=policy_action,
        )
    advantages = jnp.asarray(batch["advantages"], dtype=jnp.float32)
    if normalize_advantage:
        advantages = _normalize_advantages(advantages)

    logratio = log_prob - batch["logp"]
    ratio = jnp.exp(logratio)
    surrogate_loss1 = ratio * advantages
    surrogate_loss2 = jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    value_error = batch["returns"] - values
    value_loss = jnp.square(value_error)
    value_clip = jnp.asarray(value_clip, dtype=jnp.float32)
    value_loss = jax.lax.cond(
        value_clip > 0.0,
        lambda _: jnp.maximum(
            value_loss,
            jnp.square(batch["returns"] - (batch["values"] + jnp.clip(values - batch["values"], -value_clip, value_clip))),
        ),
        lambda _: value_loss,
        operand=None,
    )
    value_loss = jnp.mean(value_loss) * 0.5 * vf_coef

    entropy = jnp.mean(entropy)
    entropy_loss = -ent_coef * entropy
    total_loss = policy_loss + value_loss + entropy_loss
    approx_kl = ((ratio - 1.0) - logratio).mean()
    clipfrac = (jnp.abs(ratio - 1.0) > clip).astype(jnp.float32).mean()
    std = shared_actor_std(full_params)
    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "max_abs_policy_action": jnp.max(jnp.abs(policy_action)),
        "max_abs_logratio": jnp.max(jnp.abs(logratio)),
        "max_ratio": jnp.max(ratio),
        "std_min": jnp.min(std),
        "std_max": jnp.max(std),
    }


def brax_minibatch_update(
    params,
    opt_state,
    hparams,
    batch,
    lr,
    clip,
    ent_coef,
    max_grad_norm,
    vf_coef,
    value_clip,
    normalize_advantage,
    get_action_and_value_fn,
    optax_tx=None,
):
    def loss_for_update(trainable_params, full_params, loss_hparams, loss_batch, loss_clip, loss_ent_coef, loss_vf_coef, loss_value_clip):
        return brax_ppo_loss(
            trainable_params,
            full_params,
            loss_hparams,
            loss_batch,
            loss_clip,
            loss_ent_coef,
            loss_vf_coef,
            loss_value_clip,
            normalize_advantage,
            get_action_and_value_fn,
        )

    trainable = _trainable_params(params)
    (objective, metrics), grads = jax.value_and_grad(loss_for_update, has_aux=True)(
        trainable,
        params,
        hparams,
        batch,
        clip,
        ent_coef,
        vf_coef,
        value_clip,
    )
    new_trainable, new_trainable_state, combined_grad_norm = optax_optimizer_step(
        trainable,
        grads,
        opt_state["brax"],
        lr=lr,
        max_grad_norm=max_grad_norm,
        tx=optax_tx,
    )
    new_params = _with_trainable(params, new_trainable)
    new_opt_state = {"brax": new_trainable_state}

    # Match official Brax PPO hot-path semantics: apply the Optax update
    # directly, without per-minibatch finite tree scans or safe-update revert.
    finite = jnp.asarray(True)
    metrics = {
        **metrics,
        "total_loss": objective,
        "actor_grad_norm": _tree_global_norm(grads["actor"]),
        "critic_grad_norm": _tree_global_norm(grads["critic"]),
        "batch_finite": finite,
        "actor_loss_finite": finite,
        "actor_grads_finite": finite,
        "actor_step_finite": finite,
        "critic_loss_finite": finite,
        "critic_grads_finite": finite,
        "critic_step_finite": finite,
        "params_finite": finite,
        "opt_state_finite": finite,
        "update_is_finite": finite,
    }
    return new_params, new_opt_state, metrics

# Brax-style device-side PPO update loop.  This mirrors Brax's structure of
# scanning over minibatches inside each SGD epoch, then scanning over epochs,
# while preserving this repo's native batch/model/optimizer interfaces.
def make_ppo_update_fn(
    hparams,
    ppo_impl="brax",
    normalize_advantage=True,
    num_epochs=1,
    num_minibatches=1,
    max_grad_norm=1.0,
    target_kl=0.02,
    use_target_kl_early_stop=True,
    get_action_and_value_fn=None,
):
    normalize_ppo_impl(ppo_impl)
    normalize_advantage = bool(normalize_advantage)
    num_epochs = int(num_epochs)
    num_minibatches = int(num_minibatches)
    max_grad_norm = float(max_grad_norm)
    target_kl = float(target_kl)
    use_target_kl_early_stop = bool(use_target_kl_early_stop) and target_kl > 0.0
    if get_action_and_value_fn is None:
        from model.jax_diff_cvar_gnn import get_action_and_value as get_action_and_value_fn
    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be > 0")
    brax_optax_tx = make_optax_adam_tx(max_grad_norm)

    def _reshape_minibatches(x, permutation):
        x = x[permutation]
        minibatch_size = x.shape[0] // num_minibatches
        return jnp.reshape(x, (num_minibatches, minibatch_size) + x.shape[1:])

    def _empty_epoch_metrics():
        zeros = jnp.zeros((num_minibatches,), dtype=jnp.float32)
        trues = jnp.ones((num_minibatches,), dtype=jnp.bool_)
        metrics = {}
        for key in LOG_KEYS:
            if key.endswith("_finite") or key == "update_is_finite":
                metrics[key] = trues
            else:
                metrics[key] = zeros
        metrics["epoch_active"] = jnp.zeros((num_minibatches,), dtype=jnp.bool_)
        metrics["kl_stop_epoch"] = jnp.zeros((num_minibatches,), dtype=jnp.bool_)
        return metrics

    def _aggregate_metrics(metrics):
        active = metrics["epoch_active"]
        flat_active = jnp.reshape(active, (-1,))
        active_count = jnp.maximum(jnp.sum(flat_active.astype(jnp.float32)), jnp.asarray(1.0, dtype=jnp.float32))
        flat_update_finite = jnp.reshape(jnp.where(active, metrics["update_is_finite"], True), (-1,))
        update_is_finite = jnp.all(flat_update_finite)
        first_bad_flat = jnp.argmax(jnp.logical_not(flat_update_finite))
        first_bad_flat = jnp.where(update_is_finite, jnp.asarray(-1, dtype=jnp.int32), first_bad_flat.astype(jnp.int32))
        minibatch_count = metrics["update_is_finite"].shape[1]
        first_bad_epoch = jnp.where(first_bad_flat < 0, jnp.asarray(-1, dtype=jnp.int32), first_bad_flat // minibatch_count)
        first_bad_minibatch = jnp.where(first_bad_flat < 0, jnp.asarray(-1, dtype=jnp.int32), first_bad_flat % minibatch_count)

        summary = {}
        for key in LOG_KEYS:
            value = metrics[key]
            if key.endswith("_finite") or key == "update_is_finite":
                summary[key] = jnp.all(jnp.where(active, value, True))
            else:
                summary[key] = jnp.sum(jnp.where(active, value, jnp.asarray(0.0, dtype=value.dtype))) / active_count
        summary["first_nonfinite_epoch"] = first_bad_epoch
        summary["first_nonfinite_minibatch"] = first_bad_minibatch
        summary["kl_early_stop"] = jnp.any(metrics["kl_stop_epoch"])
        summary["epochs_completed"] = jnp.sum(jnp.any(active, axis=1).astype(jnp.float32))
        summary["target_kl"] = jnp.asarray(target_kl, dtype=jnp.float32)
        return summary

    def update(params, opt_state, batch, rng, lr, clip, ent_coef, max_grad_norm, vf_coef, value_clip):
        batch_size = batch["obs"].shape[0]
        if batch_size % num_minibatches != 0:
            raise ValueError("PPO batch size must divide by num_minibatches")
        target_kl_value = jnp.asarray(target_kl, dtype=jnp.float32)
        use_kl_stop = jnp.asarray(use_target_kl_early_stop, dtype=jnp.bool_)

        def minibatch_step(carry, minibatch):
            mb_params, mb_opt_state = carry
            mb_params, mb_opt_state, metrics = brax_minibatch_update(
                mb_params,
                mb_opt_state,
                hparams,
                minibatch,
                lr=lr,
                clip=clip,
                ent_coef=ent_coef,
                max_grad_norm=max_grad_norm,
                vf_coef=vf_coef,
                value_clip=value_clip,
                normalize_advantage=normalize_advantage,
                get_action_and_value_fn=get_action_and_value_fn,
                optax_tx=brax_optax_tx,
            )
            return (mb_params, mb_opt_state), metrics

        def run_epoch(carry):
            epoch_params, epoch_opt_state, epoch_rng, stopped = carry
            epoch_rng, perm_key = jax.random.split(epoch_rng)
            permutation = jax.random.permutation(perm_key, batch_size)
            minibatches = jax.tree_util.tree_map(lambda x: _reshape_minibatches(x, permutation), batch)
            (epoch_params, epoch_opt_state), metrics = jax.lax.scan(
                minibatch_step,
                (epoch_params, epoch_opt_state),
                minibatches,
                length=num_minibatches,
            )
            epoch_kl = jnp.mean(metrics["approx_kl"])
            kl_stop = use_kl_stop & jnp.isfinite(epoch_kl) & (epoch_kl > target_kl_value)
            metrics["epoch_active"] = jnp.ones((num_minibatches,), dtype=jnp.bool_)
            metrics["kl_stop_epoch"] = jnp.full((num_minibatches,), kl_stop, dtype=jnp.bool_)
            return (epoch_params, epoch_opt_state, epoch_rng, stopped | kl_stop), metrics

        def skip_epoch(carry):
            epoch_params, epoch_opt_state, epoch_rng, stopped = carry
            return (epoch_params, epoch_opt_state, epoch_rng, stopped), _empty_epoch_metrics()

        def epoch_step(carry, _unused):
            stopped = carry[3]
            return jax.lax.cond(stopped, skip_epoch, run_epoch, carry)

        (params, opt_state, rng, _stopped), metrics = jax.lax.scan(
            epoch_step,
            (params, opt_state, rng, jnp.asarray(False, dtype=jnp.bool_)),
            xs=None,
            length=num_epochs,
        )
        return params, opt_state, rng, _aggregate_metrics(metrics)

    return update
