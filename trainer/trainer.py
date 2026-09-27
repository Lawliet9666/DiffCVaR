"""Shared training framework utilities."""

import os
import shutil
from datetime import datetime

import wandb
from gymnasium.vector import AsyncVectorEnv
from omegaconf import OmegaConf

from crowd_sim.utils import build_env, resolve_env_name
from trainer.utils import get_policy_kwargs, resolve_device, set_global_seeds


def _action_std_init(cfg):
    actor_cfg = cfg.model.get("actor", {}) or {}
    return float(actor_cfg.get("action_std_init", cfg.model.get("action_std_init", 0.5)))


def _actor_cfg(cfg):
    return cfg.model.get("actor", {}) or {}


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config_dict = OmegaConf.to_container(cfg, resolve=True)
        self.config = cfg
        self.device = resolve_device(cfg.device, default="cuda")
        self.env_name = resolve_env_name(self.config)
        self.train_envs = None
        self.eval_env = None

        set_global_seeds(int(cfg.seed))
        self.run_name = self.build_run_name()
        self.save_dir = self.prepare_save_dir()
        self.hyperparameters = self.build_hyperparameters()

        os.makedirs(self.save_dir, exist_ok=True)
        OmegaConf.save(config=self.cfg, f=os.path.join(self.save_dir, "config.yaml"), resolve=True)
        print(f"Models will be saved to: {self.save_dir}", flush=True)
        self.setup_wandb()

    def build_run_name(self):
        trainer_cfg = self.cfg.trainer
        method = str(self.cfg.model.type)
        run_prefix = str(self.cfg.run_name) if self.cfg.run_name is not None and str(self.cfg.run_name) else None
        if run_prefix is None:
            run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cfg.run_name is not None and str(self.cfg.run_name):
            run_prefix = str(self.cfg.run_name)
        return (
            f"{run_prefix}-{method}"
            f"-bs{int(trainer_cfg.timesteps_per_batch)}"
            f"-ep{int(trainer_cfg.n_updates_per_iteration)}"
            f"-lr{float(trainer_cfg.lr):.1e}"
            + (f"-ent{trainer_cfg.ent_coef}" if float(trainer_cfg.ent_coef) > 0.0 else "")
        )

    def train_root(self):
        return str(self.cfg.save_dir)

    def prepare_save_dir(self):
        save_dir = os.path.join(self.train_root(), self.run_name)
        if os.path.exists(save_dir):
            if bool(self.cfg.get("overwrite", False)):
                shutil.rmtree(save_dir)
            else:
                raise ValueError(f"Save directory {save_dir} already exists; set overwrite=true to replace it")
        return save_dir

    def build_hyperparameters(self):
        trainer_cfg = self.cfg.trainer
        method = str(self.cfg.model.type)
        actor_cfg = _actor_cfg(self.cfg)
        safety_margin = float(actor_cfg.get("safety_margin", 0.0))
        alpha = float(actor_cfg.get("alpha", 2.0))
        beta_min = float(actor_cfg.get("beta_min", 0.05))
        beta = float(actor_cfg.get("beta", 0.5))
        safe_dist = (
            safety_margin
            + self.cfg.env.humans["radius"]
            + self.cfg.robot["radius"]
        )
        return {
            "model_config": self.cfg,
            "max_timesteps_per_episode": self.cfg.env.max_steps,
            "env_name": self.env_name,
            "seed": self.cfg.seed,
            "save_dir": self.save_dir,
            "device": self.device,
            "safe_dist": safe_dist,
            "safety_margin": safety_margin,
            "alpha": alpha,
            "beta_min": beta_min,
            "beta": beta,
            "robot_type": self.cfg.robot["type"],
            "vmax": self.cfg.robot["vmax"],
            "omega_max": self.cfg.robot["omega_max"],
            "obs_top_k": int(
                self.cfg.env.get("obs_top_k", self.cfg.env.get("max_obstacles_obs", 1))
            ),
            "policy_kwargs": get_policy_kwargs(self.cfg, method),
            "n_updates_per_iteration": trainer_cfg.n_updates_per_iteration,
            "timesteps_per_batch": trainer_cfg.timesteps_per_batch,
            "num_minibatches": trainer_cfg.num_minibatches,
            "clip": trainer_cfg.clip,
            "lr": trainer_cfg.lr,
            "gamma": trainer_cfg.gamma,
            "lam": trainer_cfg.lam,
            "ent_coef": trainer_cfg.ent_coef,
            "target_kl": trainer_cfg.target_kl,
            "max_grad_norm": trainer_cfg.max_grad_norm,
            "action_std_init": _action_std_init(self.cfg),
            "eval_interval": trainer_cfg.eval_interval,
            "eval_freq_timesteps": trainer_cfg.eval_freq_timesteps,
            "eval_episodes": trainer_cfg.eval_episodes,
            "max_checkpoints": trainer_cfg.max_checkpoints,
            "wandb_interval": self.cfg.wandb_interval,
        }

    def setup_wandb(self):
        wandb_entity = str(self.cfg.wandb_entity).strip() or None
        run = wandb.init(
            project=str(self.cfg.wandb_project),
            entity=wandb_entity,
            name=self.run_name,
            config=self.config_dict,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run

    def make_env_fn(self, seed_offset):
        config = self.config
        env_name = self.env_name
        seed = int(self.cfg.seed) + int(seed_offset)

        def _init():
            env = build_env(env_name, render_mode=None, config=config)
            env.reset(seed=seed)
            return env

        return _init

    def setup_env(self):
        trainer_cfg = self.cfg.trainer
        num_envs = int(trainer_cfg.num_envs)
        if num_envs <= 0:
            raise ValueError("trainer.num_envs must be > 0")
        self.train_envs = AsyncVectorEnv([self.make_env_fn(i) for i in range(num_envs)])
        if trainer_cfg.eval_interval > 0 or trainer_cfg.eval_freq_timesteps > 0:
            self.eval_env = build_env(self.env_name, render_mode=None, config=self.config)
            self.hyperparameters["eval_env"] = self.eval_env

    def close(self):
        if self.train_envs is not None:
            self.train_envs.close()
            self.train_envs = None
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None
        if wandb.run is not None:
            wandb.finish()
