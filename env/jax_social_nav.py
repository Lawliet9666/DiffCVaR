"""JAX SocialNav dynamics for fixed-shape training rollouts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from env.jax_obstacles import (
    apply_gmm_noise,
    ellipse_points_from_angles,
    human_step,
    outer_motion_bounds,
    social_force_actions,
    update_obstacle_goals,
    workspace_center_radii,
)
from env.jax_robot import angle_normalize, normalize_robot_type, robot_step


class JaxSocialNavConfig(NamedTuple):
    dt: float
    max_steps: int
    workspace_low: jnp.ndarray
    workspace_high: jnp.ndarray
    robot_type: str
    goal_tolerance: float
    robot_radius: float
    robot_vmax: float
    robot_omega_max: float
    robot_goal_min_distance: float
    robot_goal_max_tries: int
    num_humans: int
    min_num_humans: int
    random_num_humans: bool
    human_vmax_min: float
    human_vmax_max: float
    human_radius: float
    human_init_noise_range: float
    human_max_tries: int
    human_use_gmm: bool
    gmm_weights: jnp.ndarray
    gmm_stds: jnp.ndarray
    gmm_lateral_ratio: float
    random_goal_changing: bool
    goal_change_chance: float
    end_goal_changing: bool
    end_goal_change_chance: float
    social_force_ki: float
    social_force_a: float
    social_force_b: float
    social_force_avoid_robot: bool
    social_force_avoid_robot_ratio: float
    success_reward: float
    collision_penalty: float
    discomfort_dist: float
    discomfort_penalty_factor: float
    potential_factor: float
    back_factor: float
    spin_factor: float
    constant_penalty: float
    boundary_penalty_factor: float
    boundary_buffer: float


class JaxSocialNavState(NamedTuple):
    robot_state: jnp.ndarray
    robot_vel: jnp.ndarray
    robot_action: jnp.ndarray
    goal_pos: jnp.ndarray
    human_positions: jnp.ndarray
    human_vels: jnp.ndarray
    human_goals: jnp.ndarray
    human_vmaxs: jnp.ndarray
    human_radii: jnp.ndarray
    human_mask: jnp.ndarray
    human_avoid_robot_mask: jnp.ndarray
    step_count: jnp.ndarray
    prev_dist_to_goal: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray


def _cfg_get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    get_fn = getattr(cfg, "get", None)
    if callable(get_fn):
        return get_fn(name, default)
    return getattr(cfg, name, default)


def config_from_omegaconf(config) -> JaxSocialNavConfig:
    env_cfg = config.env
    robot_cfg = config.robot
    scenario = str(_cfg_get(env_cfg, "scenario"))
    supported_scenarios = {"crowd_dyn_var_num_env"}
    if scenario not in supported_scenarios:
        raise ValueError(
            f"unsupported env.scenario={scenario!r}; expected one of {tuple(sorted(supported_scenarios))}"
        )
    human_cfg = env_cfg.humans
    if int(_cfg_get(human_cfg, "human_num_range", 0)) != 0:
        raise ValueError("env.humans.human_num_range has been removed; use num_humans")
    if str(_cfg_get(human_cfg, "human_policy", "social_force")) != "social_force":
        raise ValueError("JAX env v1 requires env.humans.human_policy=social_force")
    reward = env_cfg.reward
    social_force = env_cfg.social_force
    gmm = human_cfg.gmm
    workspace = env_cfg.workspace
    workspace_low = np.asarray([float(workspace.x[0]), float(workspace.y[0])], dtype=np.float32)
    workspace_high = np.asarray([float(workspace.x[1]), float(workspace.y[1])], dtype=np.float32)
    workspace_size = workspace_high - workspace_low
    if np.any(workspace_size <= 0.0):
        raise ValueError("env.workspace high bounds must be greater than low bounds")
    vmax = list(human_cfg.vmax)
    num_humans = int(human_cfg.num_humans)
    min_num_humans = int(_cfg_get(human_cfg, "min_num_humans", num_humans))
    if num_humans <= 0:
        raise ValueError("env.humans.num_humans must be > 0")
    if min_num_humans < 0 or min_num_humans > num_humans:
        raise ValueError("env.humans.min_num_humans must satisfy 0 <= min_num_humans <= num_humans")
    avoid_robot_ratio = float(_cfg_get(social_force, "avoid_robot_ratio", 1.0))
    if avoid_robot_ratio < 0.0 or avoid_robot_ratio > 1.0:
        raise ValueError("env.social_force.avoid_robot_ratio must satisfy 0 <= ratio <= 1")
    robot_type = normalize_robot_type(_cfg_get(robot_cfg, "type", "unicycle"))
    return JaxSocialNavConfig(
        dt=float(env_cfg.dt),
        max_steps=int(env_cfg.max_steps),
        workspace_low=jnp.asarray(workspace_low, dtype=jnp.float32),
        workspace_high=jnp.asarray(workspace_high, dtype=jnp.float32),
        robot_type=robot_type,
        goal_tolerance=float(robot_cfg.goal_tolerance),
        robot_radius=float(robot_cfg.radius),
        robot_vmax=float(robot_cfg.v_max),
        robot_omega_max=float(_cfg_get(robot_cfg, "omega_max", 0.0)),
        robot_goal_min_distance=float(env_cfg.robot_goal.min_distance),
        robot_goal_max_tries=int(env_cfg.robot_goal.max_tries),
        num_humans=num_humans,
        min_num_humans=min_num_humans,
        random_num_humans=bool(_cfg_get(human_cfg, "random_num_humans", False)),
        human_vmax_min=float(vmax[0]),
        human_vmax_max=float(vmax[1]),
        human_radius=float(human_cfg.radius),
        human_init_noise_range=float(human_cfg.init_noise_range),
        human_max_tries=int(human_cfg.max_tries),
        human_use_gmm=bool(human_cfg.use_gmm),
        gmm_weights=jnp.asarray(gmm.weights, dtype=jnp.float32),
        gmm_stds=jnp.asarray(gmm.stds, dtype=jnp.float32),
        gmm_lateral_ratio=float(gmm.lateral_ratio),
        random_goal_changing=bool(_cfg_get(human_cfg, "random_goal_changing", False)),
        goal_change_chance=float(np.clip(_cfg_get(human_cfg, "goal_change_chance", 0.0), 0.0, 1.0)),
        end_goal_changing=bool(human_cfg.end_goal_changing),
        end_goal_change_chance=float(np.clip(_cfg_get(human_cfg, "end_goal_change_chance", 1.0), 0.0, 1.0)),
        social_force_ki=float(social_force.KI),
        social_force_a=float(social_force.A),
        social_force_b=float(social_force.B),
        social_force_avoid_robot=bool(social_force.avoid_robot),
        social_force_avoid_robot_ratio=avoid_robot_ratio,
        success_reward=float(reward.success_reward),
        collision_penalty=float(reward.collision_penalty),
        discomfort_dist=float(reward.discomfort_dist),
        discomfort_penalty_factor=float(reward.discomfort_penalty_factor),
        potential_factor=float(reward.potential_factor),
        back_factor=float(_cfg_get(reward, "back_factor", 0.0)),
        spin_factor=float(_cfg_get(reward, "spin_factor", 0.0)),
        constant_penalty=float(reward.constant_penalty),
        boundary_penalty_factor=float(_cfg_get(reward, "boundary_penalty_factor", 0.0)),
        boundary_buffer=float(_cfg_get(reward, "boundary_buffer", 0.0)),
    )


def compute_distances(cfg: JaxSocialNavConfig, state: JaxSocialNavState):
    dist_to_goal = jnp.linalg.norm(state.robot_state[:, 0:2] - state.goal_pos, axis=-1)
    clearances = (
        jnp.linalg.norm(state.human_positions - state.robot_state[:, None, 0:2], axis=-1)
        - cfg.robot_radius
        - state.human_radii
    )
    clearances = jnp.where(state.human_mask > 0.5, clearances, jnp.inf)
    return dist_to_goal, jnp.min(clearances, axis=-1)


def robot_boundary_violation(cfg: JaxSocialNavConfig, state: JaxSocialNavState):
    pos = state.robot_state[:, 0:2]
    low_violation = cfg.workspace_low[None, :] - cfg.boundary_buffer - pos
    high_violation = pos - (cfg.workspace_high[None, :] + cfg.boundary_buffer)
    violations = jnp.concatenate([low_violation, high_violation], axis=1)
    return jnp.maximum(jnp.max(violations, axis=1), 0.0).astype(jnp.float32)


def reward_done(cfg: JaxSocialNavConfig, state: JaxSocialNavState, dist_to_goal, min_clearance, action):
    next_step = state.step_count + 1
    collision = min_clearance < 0.0
    success = (dist_to_goal < cfg.goal_tolerance) & ~collision
    timeout = (next_step >= cfg.max_steps) & ~(collision | success)
    discomfort = (min_clearance < cfg.discomfort_dist) & ~(collision | success | timeout)
    potential = ~(collision | success | timeout | discomfort)

    potential_reward = cfg.potential_factor * (state.prev_dist_to_goal - dist_to_goal)
    potential_reward = jnp.clip(
        potential_reward,
        -cfg.dt * cfg.robot_vmax * cfg.potential_factor,
        cfg.dt * cfg.robot_vmax * cfg.potential_factor,
    )
    reward = jnp.where(collision, cfg.collision_penalty, 0.0)
    reward = jnp.where(success, cfg.success_reward, reward)
    reward = jnp.where(timeout, 0.0, reward)
    reward = jnp.where(discomfort, (min_clearance - cfg.discomfort_dist) * cfg.discomfort_penalty_factor * cfg.dt, reward)
    reward = jnp.where(potential, potential_reward, reward)

    spin = -cfg.spin_factor * jnp.square(action[:, 1] * cfg.dt)
    back = jnp.where(action[:, 0] < 0.0, -cfg.back_factor * jnp.abs(action[:, 0]), 0.0)
    boundary_violation = robot_boundary_violation(cfg, state)
    boundary_penalty = cfg.boundary_penalty_factor * boundary_violation
    reward = (reward + spin + back + cfg.constant_penalty - boundary_penalty) / 10.0
    done = collision | success | timeout
    return reward, done, success, collision, timeout, next_step, boundary_violation


def _workspace_rectangle_samples(cfg: JaxSocialNavConfig, rng, shape):
    unit = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    return cfg.workspace_low + unit * (cfg.workspace_high - cfg.workspace_low)


def _clip_robot_to_outer_bounds(cfg: JaxSocialNavConfig, state: JaxSocialNavState):
    outer_low, outer_high = outer_motion_bounds(cfg)
    robot_xy = jnp.clip(state.robot_state[:, 0:2], outer_low[None, :], outer_high[None, :])
    robot_state = state.robot_state.at[:, 0:2].set(robot_xy)
    return state._replace(robot_state=robot_state)


def fixed_order_policy_obs_from_state(cfg: JaxSocialNavConfig, state: JaxSocialNavState):
    batch_size = state.robot_state.shape[0]
    robot = jnp.concatenate(
        [
            state.robot_state[:, 0:2] - state.goal_pos,
            state.robot_vel,
            state.robot_state[:, 2:3],
            jnp.full((batch_size, 1), cfg.robot_radius, dtype=state.robot_state.dtype),
        ],
        axis=1,
    )
    blocks = jnp.concatenate(
        [
            state.robot_state[:, None, 0:2] - state.human_positions,
            state.human_vels,
            state.human_radii[..., None],
            state.human_mask[..., None],
        ],
        axis=-1,
    )
    return jnp.concatenate([robot, blocks.reshape((batch_size, -1))], axis=1).astype(jnp.float32)


def _reset_one(cfg: JaxSocialNavConfig, rng):
    key_robot, key_theta, key_goal, key_humans, key_avoid_robot = jax.random.split(rng, 5)
    key_active = jax.random.fold_in(rng, 17)
    angle = jax.random.uniform(key_robot, (), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32)
    robot_pos = ellipse_points_from_angles(cfg, angle)
    if cfg.robot_type == "unicycle":
        robot_theta = jax.random.uniform(
            key_theta,
            (),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float32,
        )
    else:
        robot_theta = jnp.asarray(0.0, dtype=jnp.float32)

    center, radii = workspace_center_radii(cfg)
    goal_candidates = _workspace_rectangle_samples(cfg, key_goal, (cfg.robot_goal_max_tries, 2))
    goal_valid = jnp.linalg.norm(goal_candidates - robot_pos[None, :], axis=1) >= cfg.robot_goal_min_distance
    goal_idx = jnp.argmax(goal_valid)
    fallback_goal = 2.0 * center - robot_pos
    goal_pos = jnp.where(jnp.any(goal_valid), goal_candidates[goal_idx], fallback_goal)

    init_positions = jnp.zeros((cfg.num_humans, 2), dtype=jnp.float32)
    init_goals = jnp.zeros((cfg.num_humans, 2), dtype=jnp.float32)
    init_vmaxs = jnp.zeros((cfg.num_humans,), dtype=jnp.float32)
    human_radii = jnp.full((cfg.num_humans,), cfg.human_radius, dtype=jnp.float32)
    safe_pair_dist = 2.0 * cfg.human_radius + cfg.discomfort_dist
    robot_safe_dist = 0.5 * jnp.linalg.norm(radii)
    human_outer_scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))

    def sample_human(carry, idx):
        positions, goals, vmaxs, key = carry
        key, key_angle, key_noise, key_vmax = jax.random.split(key, 4)
        thetas = jax.random.uniform(
            key_angle,
            (cfg.human_max_tries,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float32,
        )
        noise = jax.random.uniform(
            key_noise,
            (cfg.human_max_tries, 2),
            minval=-0.5 * cfg.human_init_noise_range,
            maxval=0.5 * cfg.human_init_noise_range,
            dtype=jnp.float32,
        )
        candidates = ellipse_points_from_angles(cfg, thetas, human_outer_scale) + noise
        candidate_goals = 2.0 * center - candidates
        previous = jnp.arange(cfg.num_humans) < idx
        dist_pos = jnp.linalg.norm(positions[None, :, :] - candidates[:, None, :], axis=2)
        dist_goals = jnp.linalg.norm(goals[None, :, :] - candidates[:, None, :], axis=2)
        clear_previous = jnp.all(
            jnp.where(
                previous[None, :],
                (dist_pos >= safe_pair_dist) & (dist_goals >= safe_pair_dist),
                True,
            ),
            axis=1,
        )
        clear_robot = jnp.linalg.norm(candidates - robot_pos[None, :], axis=1) >= robot_safe_dist
        clear_goal = jnp.linalg.norm(candidates - goal_pos[None, :], axis=1) >= robot_safe_dist
        valid = clear_previous & clear_robot & clear_goal
        choice = jnp.argmax(valid)
        pos = candidates[choice]
        goal = candidate_goals[choice]
        vmax = jax.random.uniform(
            key_vmax,
            (),
            minval=cfg.human_vmax_min,
            maxval=cfg.human_vmax_max,
            dtype=jnp.float32,
        )
        positions = positions.at[idx].set(pos)
        goals = goals.at[idx].set(goal)
        vmaxs = vmaxs.at[idx].set(vmax)
        return (positions, goals, vmaxs, key), None

    (human_positions, human_goals, human_vmaxs, _key), _ = jax.lax.scan(
        sample_human,
        (init_positions, init_goals, init_vmaxs, key_humans),
        jnp.arange(cfg.num_humans),
    )
    if cfg.random_num_humans:
        active_num_humans = jax.random.randint(
            key_active,
            (),
            minval=cfg.min_num_humans,
            maxval=cfg.num_humans + 1,
            dtype=jnp.int32,
        )
    else:
        active_num_humans = jnp.asarray(cfg.num_humans, dtype=jnp.int32)
    human_mask = (jnp.arange(cfg.num_humans) < active_num_humans).astype(jnp.float32)
    human_positions = jnp.where(human_mask[:, None] > 0.5, human_positions, 0.0)
    human_goals = jnp.where(human_mask[:, None] > 0.5, human_goals, 0.0)
    human_vmaxs = jnp.where(human_mask > 0.5, human_vmaxs, 0.0)
    if cfg.social_force_avoid_robot:
        human_avoid_robot_mask = jax.random.bernoulli(
            key_avoid_robot,
            p=cfg.social_force_avoid_robot_ratio,
            shape=(cfg.num_humans,),
        ).astype(jnp.float32)
        human_avoid_robot_mask = jnp.where(human_mask > 0.5, human_avoid_robot_mask, 0.0)
    else:
        human_avoid_robot_mask = jnp.zeros((cfg.num_humans,), dtype=jnp.float32)
    robot_state = jnp.asarray([robot_pos[0], robot_pos[1], angle_normalize(robot_theta)], dtype=jnp.float32)
    robot_action = jnp.zeros((2,), dtype=jnp.float32)
    robot_vel = jnp.zeros((2,), dtype=jnp.float32)
    prev_dist = jnp.linalg.norm(robot_pos - goal_pos)
    return JaxSocialNavState(
        robot_state=robot_state,
        robot_vel=robot_vel,
        robot_action=robot_action,
        goal_pos=goal_pos,
        human_positions=human_positions,
        human_vels=jnp.zeros_like(human_positions),
        human_goals=human_goals,
        human_vmaxs=human_vmaxs,
        human_radii=human_radii,
        human_mask=human_mask,
        human_avoid_robot_mask=human_avoid_robot_mask,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        prev_dist_to_goal=prev_dist.astype(jnp.float32),
        episode_returns=jnp.asarray(0.0, dtype=jnp.float32),
        episode_lengths=jnp.asarray(0, dtype=jnp.int32),
    )


def reset_env(cfg: JaxSocialNavConfig, rng, batch_size):
    keys = jax.random.split(rng, int(batch_size))
    return reset_env_with_keys(cfg, keys)


def reset_env_with_keys(cfg: JaxSocialNavConfig, keys):
    return jax.vmap(lambda key: _reset_one(cfg, key))(keys)


def step_env(cfg: JaxSocialNavConfig, state: JaxSocialNavState, action, rng):
    next_robot_state, robot_vel, robot_action = robot_step(cfg, state.robot_state, action)
    moved_state = state._replace(robot_state=next_robot_state, robot_vel=robot_vel, robot_action=robot_action)

    gmm_rng = rng
    if cfg.end_goal_changing or cfg.random_goal_changing:
        goal_rng, gmm_rng = jax.random.split(rng)
        moved_state = update_obstacle_goals(cfg, moved_state, goal_rng)

    human_actions = social_force_actions(cfg, moved_state)
    if cfg.human_use_gmm:
        human_actions = apply_gmm_noise(cfg, human_actions, moved_state, gmm_rng)
    next_positions, next_vels = human_step(cfg, moved_state, human_actions)
    moved_state = moved_state._replace(human_positions=next_positions, human_vels=next_vels)

    bounded_state = _clip_robot_to_outer_bounds(cfg, moved_state)
    dist_to_goal, min_clearance = compute_distances(cfg, bounded_state)
    reward, done, success, collision, timeout, next_step, boundary_violation = reward_done(
        cfg,
        bounded_state,
        dist_to_goal,
        min_clearance,
        robot_action,
    )
    completed_returns = bounded_state.episode_returns + reward
    completed_lengths = bounded_state.episode_lengths + 1
    next_state = bounded_state._replace(
        step_count=next_step,
        prev_dist_to_goal=dist_to_goal.astype(jnp.float32),
        episode_returns=jnp.where(done, 0.0, completed_returns).astype(jnp.float32),
        episode_lengths=jnp.where(done, 0, completed_lengths).astype(jnp.int32),
    )
    metrics = {
        "success": success.astype(jnp.float32),
        "collision": collision.astype(jnp.float32),
        "timeout": timeout.astype(jnp.float32),
        "min_clearance": min_clearance.astype(jnp.float32),
        "goal_distance": dist_to_goal.astype(jnp.float32),
        "boundary_violation": boundary_violation.astype(jnp.float32),
        "episode_return": completed_returns.astype(jnp.float32),
        "episode_length": completed_lengths.astype(jnp.float32),
    }
    return next_state, reward.astype(jnp.float32), done.astype(jnp.float32), metrics
