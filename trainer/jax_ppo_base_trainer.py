"""JAX PPO trainer using the Brax-style PPO training semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
import shutil
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb

from env.jax_robot import normalize_robot_type
from env.jax_social_nav import config_from_omegaconf, reset_env
from model import get_jax_model_module
from trainer.jax_checkpoint import save_jax_checkpoint
from trainer.jax_platform import (
    format_jax_device_runtime_info,
    jax_device_runtime_info,
    validate_jax_backend_for_device,
)
from trainer.jax_ppo_core import (
    LOG_KEYS,
    compute_brax_gae,
    init_ppo_opt_state,
    make_ppo_update_fn,
    normalize_ppo_impl,
    tree_all_finite,
)
from trainer.jax_ppo_eval import evaluate_jax_env
from trainer.jax_ppo_rollout import init_gnn_obs_history, make_rollout_collector


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))


def to_dict(cfg, name):
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must resolve to a dict")
    return {str(key): item for key, item in value.items()}


@dataclass
class TrainingState:
    """Small Brax-style state container for the active training loop."""

    params: Any
    opt_state: Any
    rng: Any
    env_state: Any
    global_step: int


class JaxTrainerBase:
    """JAX-only trainer shell for config, W&B, and batch sizing."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = str(getattr(config, "device", "cpu"))
        if self.config.run_name is None or not str(self.config.run_name):
            raise ValueError("run_name must be set")
        set_seed(int(config.seed))

        trainer = config.trainer
        self.total_timesteps = int(trainer.total_timesteps)
        self.timesteps_per_batch = int(trainer.timesteps_per_batch)
        self.num_envs = int(trainer.num_envs)
        self.num_minibatches = int(trainer.num_minibatches)
        if self.total_timesteps <= 0:
            raise ValueError("trainer.total_timesteps must be > 0")
        if self.timesteps_per_batch <= 0:
            raise ValueError("trainer.timesteps_per_batch must be > 0")
        if self.num_envs <= 0:
            raise ValueError("trainer.num_envs must be > 0")
        if self.num_minibatches <= 0:
            raise ValueError("trainer.num_minibatches must be > 0")

        self.steps_per_env = int(np.ceil(self.timesteps_per_batch / max(self.num_envs, 1)))
        self.rollout_batch_size = int(self.steps_per_env * self.num_envs)
        if self.rollout_batch_size % self.num_minibatches != 0:
            raise ValueError("ceil(timesteps_per_batch / num_envs) * num_envs must divide by trainer.num_minibatches")
        self.minibatch_size = int(self.rollout_batch_size // self.num_minibatches)

        self.setup_env()
        self.setup_model_and_optimizer()
        self.setup_wandb()

    def setup_wandb(self):
        self.run_name = self.build_run_name()
        self.save_dir = self.resolve_path(self.config.save_dir) / self.run_name
        if self.save_dir.exists():
            if not bool(self.config.overwrite):
                raise ValueError(f"Save directory {self.save_dir} already exists; set overwrite=true to replace it")
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(config=self.config, f=str(self.save_dir / "config.yaml"), resolve=True)
        entity_value = str(getattr(self.config, "wandb_entity", "")).strip()
        wandb_config = to_dict(self.config, "config")
        wandb_config["runtime_jax_backend"] = self.jax_backend
        run = wandb.init(
            project=str(self.config.wandb_project),
            entity=entity_value or None,
            name=self.run_name,
            config=wandb_config,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run

    def build_run_name(self):
        trainer = self.config.trainer
        run_name = (
            f"{self.config.run_name}-{self.config.model.type}"
            f"-bs{int(trainer.timesteps_per_batch)}"
            f"-ep{int(trainer.n_updates_per_iteration)}"
            f"-lr{float(trainer.lr):.1e}"
        )
        if float(trainer.ent_coef) > 0.0:
            run_name += f"-ent{trainer.ent_coef}"
        normalize_ppo_impl(getattr(trainer, "ppo_impl", "brax"))
        run_name += "-braxppo"
        lr_schedule = str(getattr(trainer, "learning_rate_schedule", "linear")).strip().lower()
        if lr_schedule == "adaptive_kl":
            run_name += "-adapkl"
        elif lr_schedule in {"none", "constant"}:
            run_name += "-constlr"
        return run_name

    def resolve_path(self, path):
        path = Path(str(path))
        if path.is_absolute():
            return path
        return PACKAGE_ROOT / path

    def setup_env(self):
        self.obs_dim = int(self.config.model.obs_dim)
        self.env_act_dim = int(self.config.env.act_dim)
        if self.env_act_dim != 2:
            raise ValueError("JAX env expects env.act_dim=2")

        robot_type = normalize_robot_type(getattr(self.config.robot, "type", "unicycle"))
        v_max = float(self.config.robot.v_max)
        if robot_type == "single_integrator":
            self.action_low = np.asarray([-v_max, -v_max], dtype=np.float32)
            self.action_high = np.asarray([v_max, v_max], dtype=np.float32)
        else:
            omega_max = float(self.config.robot.omega_max)
            self.action_low = np.asarray([-v_max, -omega_max], dtype=np.float32)
            self.action_high = np.asarray([v_max, omega_max], dtype=np.float32)

        self.history_len = int(self.config.model.history_len)
        self.obs_frame_dim = int(self.config.model.obs_frame_dim)
        expected_frame_dim = 6 + int(self.config.env.humans.num_humans) * 6
        if self.obs_frame_dim != expected_frame_dim:
            raise ValueError("model.obs_frame_dim must equal 6 + env.humans.num_humans * 6")
        if self.obs_dim != self.history_len * self.obs_frame_dim:
            raise ValueError("model.obs_dim must equal model.history_len * model.obs_frame_dim")
    def setup_model_and_optimizer(self):
        self.jax_backend = validate_jax_backend_for_device(jax, getattr(self.config, "device", "cpu"))
        self.jax_device_info = jax_device_runtime_info(jax)
        device_suffix = format_jax_device_runtime_info(self.jax_device_info)
        devices = ", ".join(str(device) for device in jax.devices())
        print(f"JAX backend: {self.jax_backend}; devices: {devices}{device_suffix}", flush=True)
        self.ppo_impl = normalize_ppo_impl(getattr(self.config.trainer, "ppo_impl", "brax"))
        self.policy_action_dim = self.env_act_dim

        self.jax_hparams = self.model_module.hparams_from_config(
            self.config.model,
            self.config.model.critic,
            act_dim=self.policy_action_dim,
        )
        rng = jax.random.PRNGKey(int(self.config.seed))
        rng, init_key = jax.random.split(rng)
        params = self.model_module.init_params_from_config(
            init_key,
            self.config.model,
            self.config.model.critic,
            act_dim=self.policy_action_dim,
            action_low=self.action_low,
            action_high=self.action_high,
            use_init_weights=bool(getattr(self.config.trainer, "use_init_weights", True)),
        )
        opt_state = init_ppo_opt_state(
            params,
            self.ppo_impl,
            max_grad_norm=float(self.config.trainer.max_grad_norm),
        )
        self._set_train_state(
            TrainingState(
                params=params,
                opt_state=opt_state,
                rng=rng,
                env_state=None,
                global_step=0,
            )
        )
        self.current_lr = float(self.config.trainer.lr)
        self.jax_env_cfg = config_from_omegaconf(self.config)
        self.jitted_train_step = self._make_jitted_train_step()
        self.lr_schedule = self._normalize_lr_schedule(getattr(self.config.trainer, "learning_rate_schedule", "linear"))
        self.lr_min = float(getattr(self.config.trainer, "learning_rate_schedule_min_lr", 0.0))
        configured_lr_max = float(getattr(self.config.trainer, "learning_rate_schedule_max_lr", 0.0))
        self.lr_max = configured_lr_max if configured_lr_max > 0.0 else float(self.config.trainer.lr)
        self.adaptive_kl_factor = float(getattr(self.config.trainer, "adaptive_kl_factor", 1.5))
        self.desired_kl = float(getattr(self.config.trainer, "desired_kl", 0.01))
        if self.lr_min < 0.0:
            raise ValueError("trainer.learning_rate_schedule_min_lr must be >= 0")
        if self.lr_max <= 0.0:
            raise ValueError("trainer.learning_rate_schedule_max_lr must be > 0, or 0 to use trainer.lr")
        if self.lr_min > self.lr_max:
            raise ValueError("trainer.learning_rate_schedule_min_lr must be <= trainer.learning_rate_schedule_max_lr")
        if self.adaptive_kl_factor <= 1.0:
            raise ValueError("trainer.adaptive_kl_factor must be > 1")
        if self.desired_kl <= 0.0:
            raise ValueError("trainer.desired_kl must be > 0")
        self.finite_check_interval = int(getattr(self.config.trainer, "finite_check_interval", 1))
        if self.finite_check_interval <= 0:
            raise ValueError("trainer.finite_check_interval must be > 0")

    def _make_jitted_train_step(self):
        ppo_update_fn = make_ppo_update_fn(
            self.jax_hparams,
            ppo_impl=self.ppo_impl,
            normalize_advantage=bool(getattr(self.config.trainer, "normalize_advantages", True)),
            num_epochs=int(self.config.trainer.n_updates_per_iteration),
            num_minibatches=int(self.num_minibatches),
            max_grad_norm=float(self.config.trainer.max_grad_norm),
            target_kl=float(getattr(self.config.trainer, "target_kl", 0.02)),
            use_target_kl_early_stop=bool(getattr(self.config.trainer, "use_target_kl_early_stop", True)),
            get_action_and_value_fn=self.model_module.get_action_and_value,
        )
        steps_per_env = int(self.steps_per_env)
        gamma = float(self.config.trainer.gamma)
        lam = float(self.config.trainer.lam)
        rollout_collector = self.rollout_collector

        def train_step(
            params,
            opt_state,
            env_state,
            rng,
            global_step,
            lr,
            clip,
            ent_coef,
            max_grad_norm,
            vf_coef,
            value_clip,
            run_finite_check,
        ):
            rng, rollout_key = jax.random.split(rng)
            batch, next_env_state, rollout_rng = rollout_collector(
                params,
                self.jax_hparams,
                self.jax_env_cfg,
                env_state,
                rollout_key,
                steps_per_env,
                global_step=global_step,
            )
            advantages, returns = compute_brax_gae(
                batch["rew"],
                batch["val"],
                batch["done"],
                batch["next_value"],
                gamma=gamma,
                lam=lam,
            )
            update_batch = JaxPPOBaseTrainer._update_batch(batch, advantages, returns)
            pre_summary = JaxPPOBaseTrainer._pre_update_summary(
                batch,
                advantages,
                returns,
                update_batch,
                run_finite_check,
            )
            new_params, new_opt_state, new_rng, metrics = ppo_update_fn(
                params,
                opt_state,
                update_batch,
                rollout_rng,
                lr,
                clip,
                ent_coef,
                max_grad_norm,
                vf_coef,
                value_clip,
            )
            post_summary = JaxPPOBaseTrainer._post_update_summary(
                metrics,
                new_params,
                new_opt_state,
                run_finite_check,
            )
            return (
                new_params,
                new_opt_state,
                next_env_state,
                new_rng,
                pre_summary["global_step"],
                pre_summary,
                post_summary,
            )

        return jax.jit(train_step, donate_argnums=(0, 1, 2, 3))

    def _set_train_state(self, train_state):
        self.train_state = train_state

    def _replace_train_state(self, **updates):
        self._set_train_state(replace(self.train_state, **updates))

    @staticmethod
    def _normalize_lr_schedule(schedule):
        value = str(schedule or "linear").strip().lower()
        if value == "none":
            value = "constant"
        if value not in {"linear", "constant", "adaptive_kl"}:
            raise ValueError("trainer.learning_rate_schedule must be one of: linear, constant/none, adaptive_kl")
        return value

    def _linear_lr(self, global_step):
        frac = (float(global_step) - 1.0) / float(self.total_timesteps)
        return float(max(float(self.config.trainer.lr) * (1.0 - frac), 0.0))

    def _lr_for_update(self, global_step):
        if self.lr_schedule == "linear":
            return self._linear_lr(global_step)
        if self.lr_schedule == "constant":
            return float(self.config.trainer.lr)
        return float(self.current_lr)

    def _update_adaptive_kl_lr(self, approx_kl):
        if self.lr_schedule != "adaptive_kl":
            return
        if approx_kl > self.desired_kl * 2.0:
            self.current_lr = max(self.current_lr / self.adaptive_kl_factor, self.lr_min)
        elif approx_kl < self.desired_kl * 0.5:
            self.current_lr = min(self.current_lr * self.adaptive_kl_factor, self.lr_max)

    def _should_eval(self, update, global_step):
        trainer = self.config.trainer
        eval_interval = int(trainer.eval_interval)
        if eval_interval > 0:
            return update % eval_interval == 0 or update == 1 or global_step >= self.total_timesteps
        eval_freq = int(trainer.eval_freq_timesteps)
        if eval_freq > 0 and (global_step - self.last_eval_timestep) >= eval_freq:
            self.last_eval_timestep = global_step
            return True
        return update >= self.total_updates

    @staticmethod
    def _summary_mean(summary, name):
        return float(summary.get(f"{name}_sum", 0.0) / max(summary.get(f"{name}_count", 1.0), 1.0))

    @staticmethod
    def _summary_value(summary, name, default=0.0):
        return float(summary.get(name, default))

    @staticmethod
    def _update_batch(batch, advantages, returns):
        update_batch = {
            "obs": jnp.asarray(batch["obs"], dtype=jnp.float32),
            "act": jnp.asarray(batch["act"], dtype=jnp.float32),
            "policy_act": jnp.asarray(batch["policy_act"], dtype=jnp.float32),
            "logp": jnp.asarray(batch["logp"], dtype=jnp.float32),
            "values": jnp.asarray(batch["val"], dtype=jnp.float32).reshape(-1),
            "advantages": jnp.asarray(advantages, dtype=jnp.float32),
            "returns": jnp.asarray(returns, dtype=jnp.float32),
        }
        if "qp_obs" in batch and "qp_topk_ids" in batch:
            update_batch["qp_obs"] = jnp.asarray(batch["qp_obs"], dtype=jnp.float32)
            update_batch["qp_topk_ids"] = jnp.asarray(batch["qp_topk_ids"], dtype=jnp.int32)
        return update_batch

    @staticmethod
    def _finite_if_enabled(run_finite_check, tree):
        return jax.lax.cond(
            run_finite_check,
            lambda _: tree_all_finite(tree),
            lambda _: jnp.asarray(True),
            operand=None,
        )

    @staticmethod
    def _pre_update_summary(batch, advantages, returns, update_batch, run_finite_check=True):
        diagnostic_keys = tuple(key for key in (
            "qp_slack_sum",
            "qp_slack_count",
            "qp_slack_max",
            "qp_correction_sum",
            "qp_correction_count",
            "beta_sum",
            "beta_count",
            "r_safe_sum",
            "r_safe_count",
        ) if key in batch)
        if "nominal_cbf_residual_sum" in batch:
            diagnostic_keys += (
                "nominal_cbf_residual_sum",
                "nominal_cbf_residual_count",
                "nominal_cbf_residual_max",
                "safe_cbf_residual_sum",
                "safe_cbf_residual_count",
                "safe_cbf_residual_max",
            )
        return {
            "global_step": batch["global_step"],
            "completed_count": batch["train_completed_count"],
            "return_sum": batch["train_return_sum"],
            "ppo_return_sum": batch["train_ppo_return_sum"],
            "length_sum": batch["train_length_sum"],
            "success_count": batch["train_success_count"],
            "collision_count": batch["train_collision_count"],
            "timeout_count": batch["train_timeout_count"],
            "min_clearance": batch["jax_env_min_clearance"],
            "boundary_violation_mean": batch["boundary_violation_mean"],
            "boundary_violation_max": batch["boundary_violation_max"],
            **{key: batch[key] for key in diagnostic_keys},
            "rollout_batch_finite": JaxPPOBaseTrainer._finite_if_enabled(run_finite_check, batch),
            "gae_finite": JaxPPOBaseTrainer._finite_if_enabled(
                run_finite_check,
                {"advantages": advantages, "returns": returns},
            ),
            "update_batch_finite": JaxPPOBaseTrainer._finite_if_enabled(run_finite_check, update_batch),
        }

    @staticmethod
    def _train_metrics_from_host_summary(summary):
        metrics = {
            "completed_count": int(summary["completed_count"]),
            "return_sum": float(summary["return_sum"]),
            "ppo_return_sum": float(summary["ppo_return_sum"]),
            "length_sum": float(summary["length_sum"]),
            "success_count": int(summary["success_count"]),
            "collision_count": int(summary["collision_count"]),
            "timeout_count": int(summary["timeout_count"]),
            "min_clearance": float(summary["min_clearance"]),
            "boundary_violation_mean": float(summary["boundary_violation_mean"]),
            "boundary_violation_max": float(summary["boundary_violation_max"]),
        }
        if "qp_slack_sum" in summary:
            metrics.update({
                "qp_slack_mean": JaxPPOBaseTrainer._summary_mean(summary, "qp_slack"),
                "qp_slack_max": JaxPPOBaseTrainer._summary_value(summary, "qp_slack_max"),
                "qp_correction_mean": JaxPPOBaseTrainer._summary_mean(summary, "qp_correction"),
                "beta_mean": JaxPPOBaseTrainer._summary_mean(summary, "beta"),
                "r_safe_mean": JaxPPOBaseTrainer._summary_mean(summary, "r_safe"),
            })
        if "nominal_cbf_residual_sum" in summary:
            metrics.update({
                "nominal_cbf_residual_mean": JaxPPOBaseTrainer._summary_mean(summary, "nominal_cbf_residual"),
                "nominal_cbf_residual_max": JaxPPOBaseTrainer._summary_value(summary, "nominal_cbf_residual_max"),
                "safe_cbf_residual_mean": JaxPPOBaseTrainer._summary_mean(summary, "safe_cbf_residual"),
                "safe_cbf_residual_max": JaxPPOBaseTrainer._summary_value(summary, "safe_cbf_residual_max"),
            })
        return metrics

    @staticmethod
    def _empty_logs():
        return {key: [] for key in LOG_KEYS}

    @staticmethod
    def _post_update_summary(metrics, params, opt_state, run_finite_check=True):
        return {
            "metrics": metrics,
            "params_finite_after_update": JaxPPOBaseTrainer._finite_if_enabled(run_finite_check, params),
            "opt_state_finite_after_update": JaxPPOBaseTrainer._finite_if_enabled(run_finite_check, opt_state),
        }

    @staticmethod
    def _nonfinite_update_details_from_summary(metrics, post_summary):
        finite_keys = (
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
        )
        details = {key: bool(metrics[key]) for key in finite_keys if key in metrics}
        details["params_finite_after_update"] = bool(post_summary["params_finite_after_update"])
        details["opt_state_finite_after_update"] = bool(post_summary["opt_state_finite_after_update"])
        return details

    @staticmethod
    def _append_host_logs(logs, summary):
        for key in LOG_KEYS:
            logs[key].append(float(summary[key]))
        return summary

    @staticmethod
    def _mean_log(logs, key):
        values = logs.get(key, [])
        return float(np.mean(values)) if values else float("nan")

    def _log_update(self, global_step, update, logs, eval_metrics, train_metrics, sps):
        wandb_interval = int(self.config.wandb_interval)
        if wandb_interval <= 0:
            raise ValueError("wandb_interval must be > 0")
        periodic_log = update % wandb_interval == 0
        log_dict = {
            "global_step": global_step,
            "charts/learning_rate": self.current_lr,
            "charts/ent_coef": float(self.config.trainer.ent_coef),
            "charts/sps": int(sps),
            "loss/policy": self._mean_log(logs, "policy_loss"),
            "loss/value": self._mean_log(logs, "value_loss"),
            "loss/entropy": self._mean_log(logs, "entropy_loss"),
            "loss/total": self._mean_log(logs, "total_loss"),
            "diagnostics/approx_kl": self._mean_log(logs, "approx_kl"),
            "diagnostics/clipfrac": self._mean_log(logs, "clipfrac"),
            "diagnostics/max_abs_logratio": self._mean_log(logs, "max_abs_logratio"),
            "safety/min_clearance": train_metrics["min_clearance"],
            "diagnostics/boundary_violation_mean": train_metrics["boundary_violation_mean"],
            "diagnostics/boundary_violation_max": train_metrics["boundary_violation_max"],
        }
        if "qp_slack_mean" in train_metrics:
            log_dict.update({
                "diagnostics/qp_slack_mean": train_metrics["qp_slack_mean"],
                "diagnostics/qp_slack_max": train_metrics["qp_slack_max"],
                "diagnostics/qp_correction_mean": train_metrics["qp_correction_mean"],
            })
        if periodic_log:
            log_dict.update({
                "diagnostics/max_abs_policy_action": self._mean_log(logs, "max_abs_policy_action"),
                "diagnostics/std_min": self._mean_log(logs, "std_min"),
                "diagnostics/std_max": self._mean_log(logs, "std_max"),
                "diagnostics/actor_grad_norm": self._mean_log(logs, "actor_grad_norm"),
                "diagnostics/critic_grad_norm": self._mean_log(logs, "critic_grad_norm"),
            })
            if "beta_mean" in train_metrics:
                log_dict.update({
                    "diagnostics/beta_mean": train_metrics["beta_mean"],
                    "diagnostics/r_safe_mean": train_metrics["r_safe_mean"],
                })
        if bool(getattr(self.jax_hparams, "qp_full_diagnostics", False)):
            log_dict.update({
                "diagnostics/nominal_cbf_residual_mean": train_metrics["nominal_cbf_residual_mean"],
                "diagnostics/nominal_cbf_residual_max": train_metrics["nominal_cbf_residual_max"],
                "diagnostics/safe_cbf_residual_mean": train_metrics["safe_cbf_residual_mean"],
                "diagnostics/safe_cbf_residual_max": train_metrics["safe_cbf_residual_max"],
            })
        if self._mean_log(logs, "kl_early_stop") > 0.5:
            log_dict.update({
                "diagnostics/kl_early_stop": 1.0,
                "diagnostics/epochs_completed": self._mean_log(logs, "epochs_completed"),
            })
        completed = int(train_metrics["completed_count"])
        if completed > 0:
            log_dict["train/episodic_env_return"] = float(train_metrics["return_sum"] / completed)
            log_dict["train/episodic_ppo_return"] = float(train_metrics["ppo_return_sum"] / completed)
            log_dict["train/episodic_length"] = float(train_metrics["length_sum"] / completed)
            log_dict["train/success_rate"] = float(train_metrics["success_count"] / completed)
            log_dict["train/collision_rate"] = float(train_metrics["collision_count"] / completed)
            log_dict["train/timeout_rate"] = float(train_metrics["timeout_count"] / completed)
        if eval_metrics is not None:
            log_dict["charts/eval_return_mean"] = eval_metrics["mean_return"]
            log_dict["charts/eval_return_std"] = eval_metrics["std_return"]
            log_dict["charts/eval_success_rate"] = eval_metrics["success_rate"]
            log_dict["charts/eval_collision_rate"] = eval_metrics["collision_rate"]
            log_dict["charts/eval_timeout_rate"] = eval_metrics["timeout_rate"]

        wandb_log = {key: value for key, value in log_dict.items() if not key.startswith("train/")}
        if completed > 0 and periodic_log:
            wandb_log.update({key: value for key, value in log_dict.items() if key.startswith("train/")})
        wandb.log(wandb_log, step=global_step)
        return log_dict

    def eval(self, episodes=None):
        episodes = int(self.config.trainer.eval_episodes if episodes is None else episodes)
        return evaluate_jax_env(
            self.train_state.params,
            self.jax_hparams,
            self.jax_env_cfg,
            episodes=episodes,
            policy_action_mean_fn=self.model_module.policy_action_mean,
            policy_action_to_env_action_fn=self.model_module.policy_action_to_env_action,
        )

    @staticmethod
    def _nonfinite_update_message(update, global_step, epoch, minibatch_start, details):
        return (
            "non-finite PPO minibatch update detected "
            f"at update={update}, global_step={global_step}, epoch={epoch}, "
            f"minibatch_start={minibatch_start}, finite={details}"
        )

    def _assert_finite_eval_metrics(self, eval_metrics, update, global_step):
        bad = {key: value for key, value in eval_metrics.items() if not np.isfinite(float(value))}
        if bad:
            raise RuntimeError(
                "non-finite eval metric detected "
                f"at update={update}, global_step={global_step}, metrics={bad}"
            )

    def train(self):
        reset_key = self._next_rng_key()
        env_state = reset_env(self.jax_env_cfg, reset_key, self.num_envs)
        env_history = init_gnn_obs_history(self.jax_env_cfg, env_state, self.history_len)
        ppo_episode_returns = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        env_state = (env_state, env_history, ppo_episode_returns)
        self._replace_train_state(env_state=env_state, global_step=0)
        start_time = time.time()
        global_step = int(self.train_state.global_step)
        update = 0

        try:
            while global_step < self.total_timesteps:
                update += 1
                state = self.train_state
                expected_global_step = int(state.global_step) + int(self.rollout_batch_size)
                self.current_lr = self._lr_for_update(expected_global_step)
                logs = self._empty_logs()
                (
                    new_params,
                    new_opt_state,
                    env_state,
                    new_rng,
                    new_global_step,
                    pre_summary,
                    post_summary,
                ) = self.jitted_train_step(
                    state.params,
                    state.opt_state,
                    state.env_state,
                    state.rng,
                    jnp.asarray(state.global_step, dtype=jnp.int32),
                    jnp.asarray(self.current_lr, dtype=jnp.float32),
                    jnp.asarray(float(self.config.trainer.clip), dtype=jnp.float32),
                    jnp.asarray(float(self.config.trainer.ent_coef), dtype=jnp.float32),
                    jnp.asarray(float(self.config.trainer.max_grad_norm), dtype=jnp.float32),
                    jnp.asarray(float(getattr(self.config.trainer, "vf_coef", 0.5)), dtype=jnp.float32),
                    jnp.asarray(float(getattr(self.config.trainer, "value_clip", 0.0)), dtype=jnp.float32),
                    jnp.asarray(update % self.finite_check_interval == 0, dtype=jnp.bool_),
                )
                summary = jax.device_get({
                    "global_step": new_global_step,
                    "pre": pre_summary,
                    "post": post_summary,
                })
                global_step = int(summary["global_step"])
                pre_summary = summary["pre"]
                post_summary = summary["post"]
                train_metrics = self._train_metrics_from_host_summary(pre_summary)
                run_finite_check = update % self.finite_check_interval == 0
                if run_finite_check and not bool(pre_summary["rollout_batch_finite"]):
                    raise RuntimeError(f"non-finite JAX value detected in rollout batch at update={update}, global_step={global_step}")
                if run_finite_check and not bool(pre_summary["gae_finite"]):
                    raise RuntimeError(f"non-finite JAX value detected in GAE outputs at update={update}, global_step={global_step}")
                if run_finite_check and not bool(pre_summary["update_batch_finite"]):
                    raise RuntimeError(f"non-finite JAX value detected in PPO update batch at update={update}, global_step={global_step}")
                metric_values = post_summary["metrics"]
                if run_finite_check and (
                    not bool(metric_values["update_is_finite"])
                    or not bool(post_summary["params_finite_after_update"])
                    or not bool(post_summary["opt_state_finite_after_update"])
                ):
                    details = self._nonfinite_update_details_from_summary(metric_values, post_summary)
                    bad_epoch = int(metric_values.get("first_nonfinite_epoch", -1))
                    bad_minibatch = int(metric_values.get("first_nonfinite_minibatch", -1))
                    bad_start = bad_minibatch * int(self.minibatch_size) if bad_minibatch >= 0 else -1
                    raise RuntimeError(self._nonfinite_update_message(update, global_step, bad_epoch, bad_start, details))

                metric_values = self._append_host_logs(logs, metric_values)
                for extra_key in ("target_kl", "kl_early_stop", "epochs_completed"):
                    logs.setdefault(extra_key, []).append(float(metric_values[extra_key]))
                approx_kl = metric_values.get("approx_kl", float("nan"))
                if not np.isfinite(approx_kl):
                    raise FloatingPointError(
                        f"non-finite approx_kl at update={update}, step={global_step}: {approx_kl}"
                    )

                lr_used = float(self.current_lr)
                self._replace_train_state(params=new_params, opt_state=new_opt_state, rng=new_rng, env_state=env_state, global_step=global_step)

                if self._should_eval(update, global_step):
                    eval_metrics = self.eval()
                    self._assert_finite_eval_metrics(eval_metrics, update, global_step)
                    save_jax_checkpoint(
                        self.save_dir,
                        self.train_state.params,
                        step=global_step,
                        performance=eval_metrics["mean_return"],
                        max_keep=int(self.config.trainer.max_checkpoints),
                    )
                else:
                    eval_metrics = None

                sps = int(global_step / max(time.time() - start_time, 1e-6))
                self.current_lr = lr_used
                self._log_update(global_step, update, logs, eval_metrics, train_metrics, sps)
                self._update_adaptive_kl_lr(approx_kl)
                eval_return = eval_metrics["mean_return"] if eval_metrics is not None else float("nan")
                eval_success = eval_metrics["success_rate"] if eval_metrics is not None else float("nan")
                print(
                    f"update {update:4d}/{self.total_updates} | steps {global_step:8d} | "
                    f"eval {eval_return:8.2f} | success {eval_success:.3f} | sps {sps}",
                    flush=True,
                )
        finally:
            if wandb.run is not None:
                wandb.finish()

    def _next_rng_key(self):
        rng, key = jax.random.split(self.train_state.rng)
        self._replace_train_state(rng=rng)
        return key

class JaxPPOBaseTrainer(JaxTrainerBase):
    """JAX trainer entrypoint for supported PPO-style robot dynamics."""

    def __init__(self, config):
        self.model_module = get_jax_model_module(config.model)
        self.rollout_collector = make_rollout_collector(
            self.model_module,
            use_qp=hasattr(self.model_module, "build_qp_context"),
        )
        super().__init__(config)
        self.total_updates = max(1, int(np.ceil(self.total_timesteps / max(self.timesteps_per_batch, 1))))
        self.last_eval_timestep = 0
