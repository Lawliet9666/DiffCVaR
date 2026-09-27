from pathlib import Path
import argparse
import glob
import json
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


def _frame_to_rgb_array(frame):
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def save_video(path, frames, fps=30):
    import imageio

    frames = [_frame_to_rgb_array(frame) for frame in frames]
    imageio.mimsave(path, frames, fps=fps)


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def render_jax_social_nav_frame(
    env_cfg,
    state,
    total_return=0.0,
    status="",
    size=740,
    robot_traj=None,
    qp_viz=None,
):
    """Render one JAX SocialNav state in the legacy matplotlib env.render style."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    robot_state = np.asarray(state.robot_state[0], dtype=float)
    robot_pos = robot_state[:2]
    robot_theta = float(robot_state[2]) if robot_state.shape[0] > 2 else 0.0
    robot_radius = float(env_cfg.robot_radius)
    human_positions = np.asarray(state.human_positions[0], dtype=float)
    human_radii = np.asarray(state.human_radii[0], dtype=float)
    human_mask = np.asarray(state.human_mask[0], dtype=float) > 0.5
    goal_pos = np.asarray(state.goal_pos[0], dtype=float)

    robot_traj_arr = np.asarray(robot_traj if robot_traj is not None else [robot_pos], dtype=float)
    start_pos = np.asarray(robot_traj_arr[0] if robot_traj_arr.size else robot_pos, dtype=float)

    low = np.asarray(getattr(env_cfg, "workspace_low", [-6.0, -6.0]), dtype=float)
    high = np.asarray(getattr(env_cfg, "workspace_high", [6.0, 6.0]), dtype=float)
    boundary_buffer = max(float(getattr(env_cfg, "boundary_buffer", 0.0)), 0.0)
    plot_low = low - boundary_buffer
    plot_high = high + boundary_buffer
    plot_span = np.maximum(plot_high - plot_low, 1e-6)
    plot_aspect = float(plot_span[0] / plot_span[1])

    frame_width = max(16, int(np.ceil(float(size) / 16.0)) * 16)
    frame_height = max(16, int(np.ceil((frame_width / plot_aspect) / 16.0)) * 16)
    dpi = max(80, int(size / 7.4))
    fig = Figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.90)

    ax.clear()
    ax.set_xlim(float(plot_low[0]), float(plot_high[0]))
    ax.set_ylim(float(plot_low[1]), float(plot_high[1]))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    robot_color = "#3b82f6"
    traj_cmap = LinearSegmentedColormap.from_list("robot_traj", ["#dbeafe", robot_color])
    qp_viz = qp_viz or {}
    robot_type = str(getattr(env_cfg, "robot_type", "unicycle"))
    lookahead_distance = float(qp_viz.get("lookahead_distance", 0.0) or 0.0)
    if robot_type != "unicycle":
        lookahead_distance = 0.0

    topk_ids = np.asarray(qp_viz.get("topk_ids", []), dtype=int).reshape(-1)
    r_safe = np.asarray(qp_viz.get("r_safe", []), dtype=float).reshape(-1)
    beta = np.asarray(qp_viz.get("beta", []), dtype=float).reshape(-1)
    beta_min = float(qp_viz.get("beta_min", 0.05) or 0.05)
    beta_total = float(
        qp_viz.get("beta_total", qp_viz.get("beta_max", 0.5)) or 0.5
    )
    active_topk_count = sum(
        0 <= human_id < human_positions.shape[0]
        and (human_id >= human_mask.shape[0] or bool(human_mask[human_id]))
        for human_id in topk_ids
    )
    remaining_beta_budget = max(beta_total - active_topk_count * beta_min, 1e-6)
    beta_individual_max = beta_min + remaining_beta_budget
    base_human_color = np.asarray(to_rgba("#8c8c8c"), dtype=float)
    risky_human_color = np.asarray(to_rgba("#dc2626"), dtype=float)
    human_colors = np.repeat(base_human_color[None, :], human_positions.shape[0], axis=0)
    for rank, human_id in enumerate(topk_ids):
        if human_id < 0 or human_id >= human_positions.shape[0] or rank >= r_safe.shape[0]:
            continue
        if human_id < human_mask.shape[0] and not human_mask[human_id]:
            continue
        beta_i = (
            float(beta[rank])
            if rank < beta.shape[0] and np.isfinite(beta[rank])
            else beta_individual_max
        )
        beta_t = np.clip(
            (beta_i - beta_min) / remaining_beta_budget, 0.0, 1.0
        )
        human_colors[human_id] = risky_human_color * (1.0 - beta_t) + base_human_color * beta_t
        adaptive_radius = float(r_safe[rank]) - robot_radius
        if not np.isfinite(adaptive_radius):
            continue
        adaptive_radius = max(adaptive_radius, float(human_radii[human_id]))
        ax.add_artist(
            Circle(
                human_positions[human_id],
                adaptive_radius,
                facecolor=to_rgba("#9ca3af", 0.16),
                edgecolor="none",
                linewidth=0.0,
                zorder=1.7,
            )
        )

    for i, pos in enumerate(human_positions):
        if i < human_mask.shape[0] and not human_mask[i]:
            continue
        ax.add_artist(Circle(pos, human_radii[i], facecolor=human_colors[i], edgecolor="none", linewidth=0.0, alpha=0.95, zorder=2))

    if robot_traj_arr.shape[0] > 1:
        points = robot_traj_arr.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_values = np.linspace(0.0, 1.0, len(segments), dtype=float)
        collection = LineCollection(
            segments,
            cmap=traj_cmap,
            norm=Normalize(vmin=0.0, vmax=1.0),
            linewidths=4.4,
            alpha=0.98,
            zorder=4,
        )
        collection.set_array(seg_values)
        collection.set_capstyle("round")
        ax.add_collection(collection)

    ax.add_artist(Circle(robot_pos, robot_radius, facecolor=robot_color, edgecolor="none", linewidth=0.0, alpha=0.82, zorder=6))
    if lookahead_distance > 0.0:
        lookahead_point = robot_pos + lookahead_distance * np.asarray([np.cos(robot_theta), np.sin(robot_theta)], dtype=float)
        ax.add_artist(
            Circle(
                lookahead_point,
                robot_radius + lookahead_distance,
                facecolor="none",
                edgecolor="#1d4ed8",
                linewidth=1.4,
                linestyle="--",
                alpha=0.7,
                zorder=7,
            )
        )
        ax.plot([robot_pos[0], lookahead_point[0]], [robot_pos[1], lookahead_point[1]], color="#1d4ed8", linewidth=1.2, alpha=0.85, zorder=8)
        ax.scatter([lookahead_point[0]], [lookahead_point[1]], s=56, c="#1d4ed8", edgecolors="white", linewidths=0.8, zorder=9)

    ax.scatter([start_pos[0]], [start_pos[1]], s=420, marker="^", c="#ead94c", edgecolors="none", linewidths=0.0, zorder=8)
    ax.add_artist(Circle(goal_pos, 0.3, facecolor="#ead94c", edgecolor="#d2bc38", linewidth=1.2, alpha=0.65, zorder=7))

    start_handle = Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="none", markersize=9, label="Start")
    goal_handle = Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#ead94c", markeredgecolor="#d2bc38", markeredgewidth=1.0, markersize=9, label="Goal")
    traj_handle = Line2D([0], [0], color=robot_color, linewidth=4.4, alpha=0.98, label="Trajectory")
    handles = [start_handle, goal_handle, traj_handle]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=13, borderpad=0.3, handletextpad=0.4)

    ax.grid(False)
    ax.tick_params(labelsize=18, colors="black")
    current_time = float(_scalar(state.step_count[0])) * float(env_cfg.dt)
    title = f"Time: {current_time:.1f}s"
    if status:
        title += f" | {status}"
    ax.set_title(title, fontsize=20, pad=7, color="black")

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


class Evaluator:
    def __init__(self, save_dir, visualize=False, device=None, seeds=None, episodes=None):
        self.save_dir = Path(save_dir)
        self.visualize = bool(visualize)
        self.seeds = list(seeds) if seeds is not None else None
        self.episodes = int(episodes) if episodes is not None else None
        self.video_save_dir = None
        self.config = OmegaConf.load(self.save_dir / "config.yaml")
        if device:
            self.config.device = str(device)
        self.jax_params = None
        self._jax_configured = False

    def _configure_jax(self):
        if self._jax_configured:
            return
        from trainer.jax_platform import configure_jax_platform_for_device

        configure_jax_platform_for_device(getattr(self.config, "device", "cpu"))
        self._jax_configured = True

    def _model_type(self):
        return str(getattr(self.config.model, "type", "diff_cvar")).strip().lower()

    def _model_module(self):
        from model import get_jax_model_module

        return get_jax_model_module(self.config.model)

    def _hparams(self):
        self._configure_jax()
        model_module = self._model_module()
        return model_module.hparams_from_config(
            self.config.model,
            self.config.model.critic,
            act_dim=int(self.config.env.act_dim),
        )

    def evaluate_one_ckpt(self, ckpt_path):
        self._configure_jax()
        from trainer.jax_checkpoint import load_jax_checkpoint

        ckpt_path = Path(ckpt_path)
        payload = load_jax_checkpoint(ckpt_path)
        self.jax_params = payload["params"]
        if self.visualize:
            self.video_save_dir = self.save_dir / f"visualize_{ckpt_path.stem}"
            self.video_save_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.evaluate()
        return metrics

    def eval_all_ckpts(self, checkpoint=None):
        ckpt_paths = [Path(checkpoint)] if checkpoint else self._checkpoint_paths()
        if not ckpt_paths:
            raise FileNotFoundError(f"no ckpt_*.pkl files found under {self.save_dir}")

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
        result_path = self.save_dir / "eval_results.json"
        result_path.write_text(json.dumps(results, indent=2))
        print(f"results: {result_path}")
        return results

    def evaluate(self):
        if self.jax_params is None:
            raise RuntimeError("evaluate_one_ckpt must load a JAX checkpoint before evaluate")
        if self.visualize:
            return self.evaluate_visualize()
        return self.evaluate_jax()

    def evaluate_jax(self):
        self._configure_jax()
        from trainer.jax_ppo_eval import evaluate_jax_env

        model_module = self._model_module()
        episodes = int(self.episodes) if self.episodes is not None else int(self.config.trainer.eval_episodes)
        return evaluate_jax_env(
            self.jax_params,
            self._hparams(),
            self.config,
            episodes=episodes,
            policy_action_mean_fn=model_module.policy_action_mean,
            policy_action_to_env_action_fn=model_module.policy_action_to_env_action,
            seeds=self.seeds,
        )

    def evaluate_visualize(self):
        self._configure_jax()
        import jax
        import jax.numpy as jnp

        from env.jax_social_nav import (
            config_from_omegaconf,
            fixed_order_policy_obs_from_state,
            reset_env,
            step_env,
        )

        env_cfg = config_from_omegaconf(self.config)
        model_module = self._model_module()
        hparams = self._hparams()
        history_len = int(self.config.model.history_len)
        seeds = self.seeds if self.seeds is not None else list(range(100, 1000 + 1, 100))
        episodes = int(self.episodes) if self.episodes is not None else int(self.config.trainer.eval_episodes)
        returns = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_episodes = 0
        has_qp_viz = hasattr(model_module, "build_qp_context") and hasattr(model_module, "safe_action_mean")

        def compute_action_and_qp_viz(policy_obs):
            if not has_qp_viz:
                policy_action = model_module.policy_action_mean(self.jax_params, hparams, policy_obs)
                action = model_module.policy_action_to_env_action(self.jax_params, policy_action)
                return action, None

            qp_context = model_module.build_qp_context(policy_obs, hparams)
            policy_action = model_module.safe_action_mean(
                self.jax_params,
                hparams,
                policy_obs,
                qp_context=qp_context,
            )
            action, _logprob, _entropy, _value, diagnostics = model_module.get_action_and_value(
                self.jax_params,
                hparams,
                policy_obs,
                policy_action=policy_action,
                return_diagnostics=True,
                qp_context=qp_context,
            )
            ctx_cpu = jax.device_get(qp_context)
            diag_cpu = jax.device_get(diagnostics)

            def first_array(name):
                value = np.asarray(diag_cpu[name])
                if value.ndim == 0:
                    return value.reshape(1)
                return value[0]

            def first_scalar(name):
                value = np.asarray(diag_cpu[name]).reshape(-1)
                return float(value[0]) if value.size else 0.0

            qp_viz = {
                "topk_ids": np.asarray(ctx_cpu["qp_topk_ids"][0], dtype=int),
                "r_safe": np.asarray(first_array("r_safe"), dtype=float),
                "beta": np.asarray(first_array("beta"), dtype=float),
                "slack_mean": first_scalar("qp_slack"),
                "slack_max": first_scalar("qp_slack_max"),
                "qp_correction": first_scalar("qp_correction"),
                "lookahead_distance": (
                    float(hparams.lookahead_distance)
                    if hparams.robot_type == "unicycle"
                    else 0.0
                ),
                "beta_min": float(getattr(hparams, "beta_min", 0.05)),
                "beta_total": float(getattr(hparams, "beta_max", 0.5)),
            }
            return action, qp_viz

        for seed in tqdm(seeds, desc="Visualizing"):
            for ep in range(episodes):
                rng = jax.random.PRNGKey(seed + ep)
                rng, reset_key = jax.random.split(rng)
                state = reset_env(env_cfg, reset_key, 1)
                state_cpu = jax.device_get(state)
                robot_traj = [np.asarray(state_cpu.robot_state[0][:2], dtype=float)]
                frame = fixed_order_policy_obs_from_state(env_cfg, state)
                history = jnp.repeat(frame[:, None, :], history_len, axis=1)
                total = 0.0
                trace = []
                status = "running"
                success = False
                collision = False
                timeout = False
                last_qp_viz = None

                for _step in range(int(env_cfg.max_steps)):
                    policy_obs = history.reshape((1, -1))
                    action, qp_viz = compute_action_and_qp_viz(policy_obs)
                    last_qp_viz = qp_viz
                    trace.append(
                        {
                            "state": jax.device_get(state),
                            "total": total,
                            "status": status,
                            "robot_traj": np.asarray(robot_traj, dtype=float).copy(),
                            "qp_viz": qp_viz,
                        }
                    )
                    rng, step_key = jax.random.split(rng)
                    state, reward, done, metrics = step_env(env_cfg, state, action, step_key)
                    state_cpu = jax.device_get(state)
                    robot_traj.append(np.asarray(state_cpu.robot_state[0][:2], dtype=float))
                    next_frame = fixed_order_policy_obs_from_state(env_cfg, state)
                    history = jnp.concatenate([history[:, 1:, :], next_frame[:, None, :]], axis=1)
                    total += float(jax.device_get(reward[0]))
                    done_bool = bool(jax.device_get(done[0] > 0.5))
                    if done_bool:
                        success = bool(jax.device_get(metrics["success"][0] > 0.5))
                        collision = bool(jax.device_get(metrics["collision"][0] > 0.5))
                        timeout = bool(jax.device_get(metrics["timeout"][0] > 0.5))
                        if success:
                            status = "success"
                        elif collision:
                            status = "collision"
                        elif timeout:
                            status = "timeout"
                        else:
                            status = "done"
                        trace.append(
                            {
                                "state": state_cpu,
                                "total": total,
                                "status": status,
                                "robot_traj": np.asarray(robot_traj, dtype=float).copy(),
                                "qp_viz": last_qp_viz,
                            }
                        )
                        break

                if not trace:
                    trace.append(
                        {
                            "state": jax.device_get(state),
                            "total": total,
                            "status": status,
                            "robot_traj": np.asarray(robot_traj, dtype=float).copy(),
                            "qp_viz": last_qp_viz,
                        }
                    )
                frames = [
                    render_jax_social_nav_frame(
                        env_cfg,
                        item["state"],
                        item["total"],
                        status=item["status"],
                        robot_traj=item["robot_traj"],
                        qp_viz=item["qp_viz"],
                    )
                    for item in trace
                ]
                video_path = self.video_save_dir / f"seed_{seed + ep}_{status}.mp4"
                save_video(video_path, frames, fps=int(round(1.0 / float(env_cfg.dt))))
                returns.append(total)
                total_episodes += 1
                success_count += int(success)
                collision_count += int(collision)
                timeout_count += int(timeout or status == "running")

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": success_count / total_episodes,
            "collision_rate": collision_count / total_episodes,
            "timeout_rate": timeout_count / total_episodes,
        }

    def _checkpoint_paths(self):
        paths = [Path(path) for path in glob.glob(str(self.save_dir / "ckpt_*.pkl"))]
        paths.sort(key=lambda path: int(path.stem.split("_")[-1]))
        return paths

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
            "max_success_rate": float(max(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "max_collision_rate": float(max(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "max_timeout_rate": float(max(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
            "min_success_rate": float(min(metrics, key=lambda item: item["success_rate"])["success_rate"]),
            "min_collision_rate": float(min(metrics, key=lambda item: item["collision_rate"])["collision_rate"]),
            "min_timeout_rate": float(min(metrics, key=lambda item: item["timeout_rate"])["timeout_rate"]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, help="Run directory containing config.yaml and ckpt_*.pkl")
    parser.add_argument("--checkpoint", default="", help="Evaluate one checkpoint instead of all ckpt_*.pkl files")
    parser.add_argument("--visualize", action="store_true", help="Save MP4 rollout videos with the JAX env renderer")
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "auto"], help="Override device from config for eval/visualize")
    parser.add_argument("--seeds", default="", help="Comma-separated eval seeds; default is 100,200,...,1000")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed; default uses config.trainer.eval_episodes")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()] if args.seeds else None
    Evaluator(
        args.save_dir,
        visualize=args.visualize,
        device=args.device or None,
        seeds=seeds,
        episodes=args.episodes,
    ).eval_all_ckpts(checkpoint=args.checkpoint or None)
