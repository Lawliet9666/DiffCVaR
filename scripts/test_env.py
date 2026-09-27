from pathlib import Path
import argparse
import json
import os
import sys

import hydra
import imageio
import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CACHE_DIR = REPO_ROOT / "outputs" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from crowd_sim.utils import build_env, resolve_env_name


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


def load_config(seed, env_name, overrides):
    config_dir = str(REPO_ROOT / "config")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(
            config_name="config",
            overrides=[f"seed={seed}", f"env={env_name}", *overrides],
        )


def next_save_dir(base_dir, name):
    path = base_dir / name
    if not path.exists():
        return path

    index = 1
    while True:
        path = base_dir / f"{name}_{index}"
        if not path.exists():
            return path
        index += 1


def env_count_token(cfg):
    humans = cfg.env.humans
    count = int(humans["num_humans"])
    human_num_range = int(humans.get("human_num_range", 0))
    if str(cfg.env.name) == "social_nav_var_num" and human_num_range > 0:
        return f"h{count}r{human_num_range}"
    return f"h{count}"


def build_env_test_name(cfg, args):
    tokens = [
        str(cfg.env.name),
        env_count_token(cfg),
        f"s{args.seed}",
        f"ep{args.episodes}",
    ]
    return "-".join(tokens)


def result_from_info(info, step, steps):
    if info.get("is_collision", False):
        return "collision"
    if info.get("is_success", False):
        return "success"
    if info.get("is_timeout", False) or step >= steps:
        return "timeout"
    return "running"


def distance_metrics(env):
    goal_distance = float(np.linalg.norm(env.robot_pos - env.goal_pos))
    if env.num_humans <= 0:
        return goal_distance, float("inf")

    center_dists = np.linalg.norm(env.human_positions - env.robot_pos, axis=1)
    clearances = center_dists - env.human_radii - float(env.robot_radius)
    return goal_distance, float(np.min(clearances))


def save_frames(frames, save_dir, seed, result):
    if not frames:
        return None
    gif_path = save_dir / f"seed_{seed}_{result}.gif"
    imageio.mimsave(gif_path, frames, fps=10)
    return gif_path


def run_env_test(env_name, seed, steps, save_dir, overrides):
    cfg = load_config(seed, env_name, overrides)
    resolved_env_name = resolve_env_name(cfg)
    env = build_env(resolved_env_name, render_mode="rgb_array", config=cfg)

    obs, _info = env.reset(seed=seed)
    if obs.shape[0] != env.obs_dim:
        raise RuntimeError(f"obs shape {obs.shape} does not match env.obs_dim={env.obs_dim}")

    total_reward = 0.0
    min_clearance = float("inf")
    frames = [env.render()]
    info = {}

    for step in range(1, steps + 1):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if obs.shape[0] != env.obs_dim:
            raise RuntimeError(f"obs shape {obs.shape} does not match env.obs_dim={env.obs_dim}")

        total_reward += float(reward)
        _goal_distance, clearance = distance_metrics(env)
        min_clearance = min(min_clearance, clearance)
        frames.append(env.render())

        if bool(terminated or truncated):
            break

    result = result_from_info(info, step, steps)
    final_goal_distance, final_clearance = distance_metrics(env)
    if min_clearance == float("inf"):
        min_clearance = final_clearance

    gif_path = save_frames(frames, save_dir, seed, result)
    env.close()

    return {
        "seed": int(seed),
        "steps": int(step),
        "total_reward": float(total_reward),
        "result": result,
        "gif_path": str(gif_path) if gif_path is not None else None,
        "final_goal_distance": float(final_goal_distance),
        "min_clearance": float(min_clearance),
        "collision": bool(info.get("is_collision", False)),
        "success": bool(info.get("is_success", False)),
        "timeout": bool(info.get("is_timeout", False)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-name",
        choices=["social_nav", "social_nav_var_num"],
        default="social_nav_var_num",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. env.humans.num_humans=15")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")

    cfg = load_config(args.seed, args.env_name, args.overrides)
    if args.save_dir is None:
        test_name = build_env_test_name(cfg, args)
        save_dir = next_save_dir(REPO_ROOT / "outputs" / str(cfg.env.name) / "env_test", test_name)
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving test results to: {save_dir}", flush=True)

    summaries = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        summary = run_env_test(args.env_name, seed, args.steps, save_dir, args.overrides)
        print(
            f"Episode {episode}: {summary['result']} | steps: {summary['steps']} | "
            f"return: {summary['total_reward']:.2f} | final goal: {summary['final_goal_distance']:.3f}",
            flush=True,
        )
        summaries.append(summary)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"env test passed: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
