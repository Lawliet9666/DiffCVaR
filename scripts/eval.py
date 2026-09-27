"""Evaluate one or all RL checkpoints in a training run."""

from pathlib import Path
import argparse
import glob
import json
import re
import sys

import imageio
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.utils import absolute_obs_to_relative, build_env, resolve_env_name, select_top_k_obs
from model.factory import build_model
from scripts.override_validation import validate_existing_leaf_overrides
from trainer.utils import resolve_device


FIXED_EVAL_SEEDS = list(range(100, 1000 + 1, 100))


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


def policy_obs_from_env_obs(obs, obs_top_k):
    return select_top_k_obs(absolute_obs_to_relative(obs), obs_top_k)


def render_step(env, frames):
    if env.render_mode is None:
        return
    if env.render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    else:
        env.render()


def _safe_scalar(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.flatten()[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def set_render_safe_distance(env, actor):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if base_env is None:
        return

    safe_dist = _safe_scalar(getattr(actor, "last_r_safe", None))
    base_env.render_safe_dist = float(safe_dist) if np.isfinite(safe_dist) else None


def override_suffix(overrides):
    if not overrides:
        return ""
    raw = "__".join(str(item) for item in overrides)
    token = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return f"_{token[:120]}" if token else "_overrides"


class Evaluator:
    def __init__(self, save_dir, episodes_per_seed=None, visualize=False, overrides=None):
        self.save_dir = Path(save_dir)
        self.visualize = bool(visualize)
        self.video_save_dir = None
        self.overrides = list(overrides or [])
        self.output_suffix = override_suffix(self.overrides)
        self.config = OmegaConf.load(self.save_dir / "config.yaml")
        if self.overrides:
            validate_existing_leaf_overrides(self.config, self.overrides, context="eval")
            OmegaConf.set_struct(self.config, True)
            self.config = OmegaConf.merge(self.config, OmegaConf.from_dotlist(self.overrides))
            OmegaConf.resolve(self.config)
            print(f"eval overrides: {' '.join(self.overrides)}", flush=True)
        self.env_name = resolve_env_name(self.config)
        self.device = resolve_device(self.config.get("device", "auto"), default="cuda")

        env = build_env(self.env_name, render_mode=None, config=self.config)
        try:
            self.obs_top_k = int(
                self.config.env.get(
                    "obs_top_k",
                    self.config.env.get("max_obstacles_obs", 1),
                )
            )
            self.obs_dim = int(self.config.model.obs_dim)
            self.act_dim = int(self.config.model.act_dim)
            env_act_dim = int(env.action_space.shape[0])
            if self.obs_dim != 6 + self.obs_top_k * 6:
                raise ValueError("model.obs_dim must equal 6 + env.obs_top_k * 6")
            if env_act_dim != int(self.config.env.act_dim):
                raise ValueError("env.act_dim does not match env.action_space")
            if self.act_dim != env_act_dim:
                raise ValueError("model.act_dim must match env.action_space")
            self.action_low = env.action_space.low.astype(np.float32)
            self.action_high = env.action_space.high.astype(np.float32)
        finally:
            env.close()

        self.model = build_model(
            self.config,
            self.obs_dim,
            self.act_dim,
            action_low=self.action_low,
            action_high=self.action_high,
        ).to(self.device)

        default_episodes = int(self.config.trainer.eval_episodes)
        self.episodes_per_seed = int(episodes_per_seed if episodes_per_seed is not None else default_episodes)
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be > 0")

    def evaluate_one_ckpt(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        if self.visualize:
            self.video_save_dir = self.save_dir / f"visualize_{ckpt_path.stem}{self.output_suffix}"
            self.video_save_dir.mkdir(parents=True, exist_ok=True)
        return self.evaluate(ckpt_path)

    def eval_all_ckpts(self, checkpoint=None):
        ckpt_paths = [Path(checkpoint)] if checkpoint else self._checkpoint_paths()
        if not ckpt_paths:
            raise FileNotFoundError(f"no ckpt_*.pt files found under {self.save_dir}")

        results = {}
        for ckpt_path in ckpt_paths:
            print(f"Evaluating {ckpt_path}...")
            metrics = self.evaluate_one_ckpt(ckpt_path)
            results[ckpt_path.name] = metrics
            print(
                f"\treturn: {metrics['mean_return']:.2f} +/- {metrics['std_return']:.2f}, "
                f"success: {metrics['success_rate']:.2f}, collision: {metrics['collision_rate']:.2f}, "
                f"timeout: {metrics['timeout_rate']:.2f}"
            )

        results["aggregate"] = self._aggregate(results)
        result_path = self.save_dir / f"eval_results{self.output_suffix}.json"
        result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"results: {result_path}")
        return results

    def evaluate(self, ckpt_path):
        env = build_env(
            self.env_name,
            render_mode="rgb_array" if self.visualize else None,
            config=self.config,
        )
        returns = []
        lengths = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        infeasible_count = 0
        total_episodes = 0

        try:
            for seed in tqdm(FIXED_EVAL_SEEDS, desc="Evaluating"):
                for ep in range(self.episodes_per_seed):
                    episode_seed = seed + ep
                    obs, _info = env.reset(seed=episode_seed)
                    self.model.reset_episode_cache()
                    done = False
                    total = 0.0
                    length = 0
                    info = {}
                    frames = [] if self.visualize else None

                    while not done:
                        policy_obs = policy_obs_from_env_obs(obs, self.obs_top_k)
                        obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        with torch.no_grad():
                            policy_action = self.model.get_action_deterministic(obs_t)
                            set_render_safe_distance(env, self.model.actor)
                            action = self.model.policy_action_to_env_action(obs_t, policy_action)
                            action = action.detach().cpu().numpy().astype(np.float32).squeeze(0)

                        obs, reward, terminated, truncated, info = env.step(action)
                        done = bool(terminated or truncated)
                        total += float(reward)
                        length += 1
                        if frames is not None:
                            render_step(env, frames)

                    if frames:
                        success = bool(info.get("is_success", info.get("reached", False)))
                        video_path = self.video_save_dir / f"seed_{episode_seed}_{success}.gif"
                        imageio.mimsave(video_path, frames, fps=10)

                    returns.append(total)
                    lengths.append(length)
                    total_episodes += 1
                    success_count += int(info.get("is_success", info.get("reached", False)))
                    collision_count += int(info.get("is_collision", info.get("collision", False)))
                    timeout_count += int(info.get("is_timeout", info.get("timeout", False)))
                    infeasible_count += int(bool(getattr(self.model.actor, "infeasible", False)))
        finally:
            env.close()

        total = max(total_episodes, 1)
        return {
            "checkpoint": str(ckpt_path),
            "step": self._checkpoint_step(ckpt_path),
            "eval_seeds": FIXED_EVAL_SEEDS,
            "episodes_per_seed": self.episodes_per_seed,
            "total_episodes": total_episodes,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": success_count / total,
            "collision_rate": collision_count / total,
            "timeout_rate": timeout_count / total,
            "infeasible_rate": infeasible_count / total,
        }

    def _checkpoint_paths(self):
        paths = [Path(path) for path in glob.glob(str(self.save_dir / "ckpt_*.pt"))]
        paths.sort(key=lambda path: self._checkpoint_step(path))
        return paths

    @staticmethod
    def _checkpoint_step(path):
        match = re.search(r"ckpt_(\d+)\.pt$", Path(path).name)
        if not match:
            raise ValueError(f"Expected checkpoint filename format ckpt_<step>.pt: {path}")
        return int(match.group(1))

    @staticmethod
    def _aggregate(results):
        metrics = [value for key, value in results.items() if key != "aggregate"]
        return {
            "num_checkpoints": len(metrics),
            "min_return": float(min(metrics, key=lambda item: item["mean_return"])["mean_return"]),
            "max_return": float(max(metrics, key=lambda item: item["mean_return"])["mean_return"]),
            "mean_return": float(np.mean([m["mean_return"] for m in metrics])),
            "std_return": float(np.std([m["mean_return"] for m in metrics])),
            "mean_success_rate": float(np.mean([m["success_rate"] for m in metrics])),
            "mean_collision_rate": float(np.mean([m["collision_rate"] for m in metrics])),
            "mean_timeout_rate": float(np.mean([m["timeout_rate"] for m in metrics])),
            "mean_infeasible_rate": float(np.mean([m["infeasible_rate"] for m in metrics])),
            "max_success_rate": float(max(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "max_collision_rate": float(max(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "max_timeout_rate": float(max(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
            "min_success_rate": float(min(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "min_collision_rate": float(min(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "min_timeout_rate": float(min(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, help="Run directory containing config.yaml and ckpt_*.pt")
    parser.add_argument("--checkpoint", default="", help="Evaluate one checkpoint instead of all ckpt_*.pt files")
    parser.add_argument("--episodes-per-seed", type=int, default=None, help="Episodes evaluated for each fixed seed")
    parser.add_argument("--visualize", action="store_true", help="Save GIF rollout videos")
    parser.add_argument("overrides", nargs="*", help="Eval-only OmegaConf overrides, e.g. env.humans.use_gmm=false")
    args = parser.parse_args()

    try:
        Evaluator(
            args.save_dir,
            episodes_per_seed=args.episodes_per_seed,
            visualize=args.visualize,
            overrides=args.overrides,
        ).eval_all_ckpts(checkpoint=args.checkpoint or None)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
