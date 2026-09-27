"""JAX human obstacle and social-force helpers for SocialNav."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from env.jax_social_nav import JaxSocialNavConfig, JaxSocialNavState


def workspace_center_radii(cfg: JaxSocialNavConfig):
    center = 0.5 * (cfg.workspace_low + cfg.workspace_high)
    radii = jnp.maximum(0.5 * (cfg.workspace_high - cfg.workspace_low), 1e-3)
    return center, radii


def outer_motion_bounds(cfg: JaxSocialNavConfig):
    center, radii = workspace_center_radii(cfg)
    half_extents = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32)) * radii
    half_extents = half_extents + cfg.human_init_noise_range
    return center - half_extents, center + half_extents


def ellipse_points_from_angles(cfg: JaxSocialNavConfig, angles, radial_scale=1.0):
    center, radii = workspace_center_radii(cfg)
    unit = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    return center + radii * jnp.asarray(radial_scale, dtype=jnp.float32) * unit


def _gmm_component_means(actions, lateral_ratio):
    speed_sq = jnp.sum(actions * actions, axis=-1, keepdims=True)
    zeros = jnp.zeros_like(actions)
    vx = actions[..., 0]
    vy = actions[..., 1]
    scale = 1.0 / jnp.sqrt(1.0 + lateral_ratio * lateral_ratio)
    means = jnp.stack(
        [
            actions,
            scale * jnp.stack([vx - lateral_ratio * vy, vy + lateral_ratio * vx], axis=-1),
            scale * jnp.stack([vx + lateral_ratio * vy, vy - lateral_ratio * vx], axis=-1),
        ],
        axis=-2,
    )
    return jnp.where(speed_sq[..., None] < 1e-12, zeros[..., None, :], means)

def gmm_noise_from_component_noise(cfg: JaxSocialNavConfig, nominal_actions, component_idx, normal_noise):
    means = _gmm_component_means(nominal_actions, cfg.gmm_lateral_ratio)
    one_hot = jax.nn.one_hot(component_idx, cfg.gmm_weights.shape[0], dtype=nominal_actions.dtype)
    mean = jnp.sum(means * one_hot[..., None], axis=-2)
    std = jnp.sum(cfg.gmm_stds * one_hot, axis=-1)
    return mean + normal_noise * std[..., None]

def apply_gmm_noise(cfg: JaxSocialNavConfig, nominal_actions, state: JaxSocialNavState, rng):
    comp_key, noise_key = jax.random.split(rng)
    logits = jnp.log(cfg.gmm_weights)
    component_idx = jax.random.categorical(comp_key, logits, shape=nominal_actions.shape[:-1])
    normal_noise = jax.random.normal(noise_key, nominal_actions.shape, dtype=nominal_actions.dtype)
    noisy = gmm_noise_from_component_noise(cfg, nominal_actions, component_idx, normal_noise)
    dist_to_goal = jnp.linalg.norm(state.human_goals - state.human_positions, axis=-1)
    noise_mask = (dist_to_goal >= 2.0 * state.human_radii) & (state.human_mask > 0.5)
    return jnp.where(noise_mask[..., None], noisy, nominal_actions)

def _sample_new_human_goal(
    cfg: JaxSocialNavConfig,
    robot_pos,
    robot_goal,
    human_positions,
    human_goals,
    human_radii,
    human_mask,
    human_vmaxs,
    idx,
    rng,
):
    old_goal = human_goals[idx]
    v_pref = human_vmaxs[idx]
    self_radius = human_radii[idx]

    key_angle, key_noise = jax.random.split(rng)
    angles = jax.random.uniform(
        key_angle,
        (cfg.human_max_tries,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=jnp.float32,
    )
    noise = jax.random.uniform(
        key_noise,
        (cfg.human_max_tries, 2),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ) * v_pref
    human_outer_scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
    candidates = ellipse_points_from_angles(cfg, angles, human_outer_scale) + noise

    min_robot_dist = self_radius + cfg.robot_radius + cfg.discomfort_dist
    clear_robot = jnp.linalg.norm(candidates - robot_pos[None, :], axis=1) >= min_robot_dist
    clear_robot_goal = jnp.linalg.norm(candidates - robot_goal[None, :], axis=1) >= min_robot_dist

    other_mask = (jnp.arange(cfg.num_humans) != idx) & (human_mask > 0.5)
    safe_dists = self_radius + human_radii + cfg.discomfort_dist
    dist_to_positions = jnp.linalg.norm(candidates[:, None, :] - human_positions[None, :, :], axis=2)
    dist_to_goals = jnp.linalg.norm(candidates[:, None, :] - human_goals[None, :, :], axis=2)
    clear_humans = jnp.all(
        jnp.where(
            other_mask[None, :],
            (dist_to_positions >= safe_dists[None, :]) & (dist_to_goals >= safe_dists[None, :]),
            True,
        ),
        axis=1,
    )

    valid = clear_robot & clear_robot_goal & clear_humans
    selected = candidates[jnp.argmax(valid)]
    keep_old = (v_pref <= 1e-8) | (~jnp.any(valid))
    return jnp.where(keep_old, old_goal, selected)


def update_obstacle_goals(cfg: JaxSocialNavConfig, state: JaxSocialNavState, rng):
    if not (cfg.end_goal_changing or cfg.random_goal_changing):
        return state
    if cfg.num_humans <= 0:
        return state

    batch_size = state.robot_state.shape[0]
    key_trigger, key_sample = jax.random.split(rng)
    trigger_draws = jax.random.uniform(key_trigger, state.human_vmaxs.shape, dtype=jnp.float32)
    sample_keys = jax.random.split(key_sample, batch_size * cfg.num_humans).reshape((batch_size, cfg.num_humans, 2))
    human_indices = jnp.arange(cfg.num_humans)

    def update_one(robot_state, robot_goal, positions, goals, vmaxs, radii, mask, triggers, keys):
        robot_pos = robot_state[0:2]

        def update_goal(goals_in, idx):
            active = (vmaxs[idx] > 1e-8) & (mask[idx] > 0.5)
            dist_to_goal = jnp.linalg.norm(positions[idx] - goals_in[idx])
            near_goal = dist_to_goal <= radii[idx] + 0.1
            random_trigger = cfg.random_goal_changing & (triggers[idx] <= cfg.goal_change_chance)
            end_goal_trigger = cfg.end_goal_changing & near_goal & (triggers[idx] <= cfg.end_goal_change_chance)
            should_change = active & (random_trigger | end_goal_trigger)

            def change_goal(current_goals):
                new_goal = _sample_new_human_goal(
                    cfg,
                    robot_pos,
                    robot_goal,
                    positions,
                    current_goals,
                    radii,
                    mask,
                    vmaxs,
                    idx,
                    keys[idx],
                )
                return current_goals.at[idx].set(new_goal)

            return jax.lax.cond(should_change, change_goal, lambda current_goals: current_goals, goals_in), None

        new_goals, _ = jax.lax.scan(update_goal, goals, human_indices)
        return new_goals

    new_goals = jax.vmap(update_one)(
        state.robot_state,
        state.goal_pos,
        state.human_positions,
        state.human_goals,
        state.human_vmaxs,
        state.human_radii,
        state.human_mask,
        trigger_draws,
        sample_keys,
    )
    return state._replace(human_goals=new_goals)

def social_force_actions(cfg: JaxSocialNavConfig, state: JaxSocialNavState):
    pos = state.human_positions
    vel = state.human_vels
    goals = state.human_goals
    radii = state.human_radii
    vpref = jnp.maximum(state.human_vmaxs, 0.0)
    mask = state.human_mask > 0.5

    goal_delta = goals - pos
    dist_goal = jnp.linalg.norm(goal_delta, axis=-1)
    valid_goal = (dist_goal > 1e-5) & mask
    desired_vel = jnp.where(
        valid_goal[..., None],
        goal_delta / jnp.maximum(dist_goal[..., None], 1e-8) * vpref[..., None],
        0.0,
    )
    curr_delta = cfg.social_force_ki * (desired_vel - vel)

    if cfg.social_force_avoid_robot:
        all_pos = jnp.concatenate([pos, state.robot_state[:, None, 0:2]], axis=1)
        all_radii = jnp.concatenate(
            [radii, jnp.full((pos.shape[0], 1), cfg.robot_radius, dtype=pos.dtype)],
            axis=1,
        )
        all_mask = jnp.concatenate([mask, jnp.ones((pos.shape[0], 1), dtype=bool)], axis=1)
        robot_neighbor_mask = jnp.asarray(state.human_avoid_robot_mask > 0.5, dtype=bool)
    else:
        all_pos = pos
        all_radii = radii
        all_mask = mask
        robot_neighbor_mask = jnp.zeros_like(mask, dtype=bool)

    delta = pos[:, :, None, :] - all_pos[:, None, :, :]
    dist = jnp.linalg.norm(delta, axis=-1)
    valid_neighbor = mask[:, :, None] & all_mask[:, None, :]
    eye = jnp.eye(cfg.num_humans, dtype=bool)[None, :, :]
    valid_neighbor = valid_neighbor.at[:, :, : cfg.num_humans].set(valid_neighbor[:, :, : cfg.num_humans] & ~eye)
    if cfg.social_force_avoid_robot:
        valid_neighbor = valid_neighbor.at[:, :, cfg.num_humans].set(
            valid_neighbor[:, :, cfg.num_humans] & robot_neighbor_mask
        )

    b = max(float(cfg.social_force_b), 1e-5)
    sum_r = radii[:, :, None] + all_radii[:, None, :]
    non_overlap = valid_neighbor & (dist > 1e-5)
    force_mag = cfg.social_force_a * jnp.exp((sum_r - dist) / b)
    unit = jnp.where(non_overlap[..., None], delta / jnp.maximum(dist[..., None], 1e-8), 0.0)
    interaction = jnp.sum(force_mag[..., None] * unit * non_overlap[..., None], axis=2)

    new_vel = vel + (curr_delta + interaction) * cfg.dt
    speed = jnp.linalg.norm(new_vel, axis=-1)
    clip_mask = (speed > vpref) & (vpref > 0.0)
    new_vel = jnp.where(clip_mask[..., None], new_vel / jnp.maximum(speed[..., None], 1e-8) * vpref[..., None], new_vel)
    return jnp.where(mask[..., None], new_vel, 0.0)

def human_step(cfg: JaxSocialNavConfig, state: JaxSocialNavState, actions):
    speed = jnp.linalg.norm(actions, axis=-1)
    clipped = jnp.where(
        (state.human_vmaxs > 0.0)[..., None] & (speed > state.human_vmaxs)[..., None],
        actions / jnp.maximum(speed[..., None], 1e-8) * state.human_vmaxs[..., None],
        actions,
    )
    clipped = jnp.where((state.human_vmaxs > 0.0)[..., None], clipped, 0.0)
    clipped = jnp.where((state.human_mask > 0.5)[..., None], clipped, 0.0)
    next_positions = state.human_positions + clipped * cfg.dt
    outer_low, outer_high = outer_motion_bounds(cfg)
    clipped_positions = jnp.clip(
        next_positions,
        outer_low[None, None, :],
        outer_high[None, None, :],
    )
    next_positions = jnp.where((state.human_mask > 0.5)[..., None], clipped_positions, state.human_positions)
    return next_positions, clipped
