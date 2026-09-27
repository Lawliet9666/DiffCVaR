"""JAX robot dynamics helpers for SocialNav."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp


if TYPE_CHECKING:
    from env.jax_social_nav import JaxSocialNavConfig


def angle_normalize(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def normalize_robot_type(robot_type):
    robot_type = str(robot_type).strip().lower()
    if robot_type not in {"single_integrator", "unicycle"}:
        raise ValueError(
            f"unsupported robot.type={robot_type!r}; expected 'single_integrator' or 'unicycle'"
        )
    return robot_type


def single_integrator_step(cfg: JaxSocialNavConfig, robot_state, action):
    action = jnp.asarray(action, dtype=jnp.float32)
    speed = jnp.linalg.norm(action, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, cfg.robot_vmax / jnp.maximum(speed, 1e-8))
    robot_action = action * scale
    next_xy = robot_state[..., 0:2] + robot_action * cfg.dt
    next_theta = jnp.zeros_like(robot_state[..., 2:3])
    next_state = jnp.concatenate([next_xy, next_theta], axis=-1)
    return next_state, robot_action, robot_action


def unicycle_step(cfg: JaxSocialNavConfig, robot_state, action):
    action = jnp.asarray(action, dtype=jnp.float32)
    v = jnp.clip(action[..., 0], -cfg.robot_vmax, cfg.robot_vmax)
    omega = jnp.clip(action[..., 1], -cfg.robot_omega_max, cfg.robot_omega_max)
    theta = robot_state[..., 2]
    next_theta = angle_normalize(theta + omega * cfg.dt)
    next_state = jnp.stack(
        [
            robot_state[..., 0] + v * jnp.cos(theta) * cfg.dt,
            robot_state[..., 1] + v * jnp.sin(theta) * cfg.dt,
            next_theta,
        ],
        axis=-1,
    )
    robot_action = jnp.stack([v, omega], axis=-1)
    robot_vel = jnp.stack([v * jnp.cos(next_theta), v * jnp.sin(next_theta)], axis=-1)
    return next_state, robot_vel, robot_action


def robot_step(cfg: JaxSocialNavConfig, robot_state, action):
    if cfg.robot_type == "single_integrator":
        return single_integrator_step(cfg, robot_state, action)
    return unicycle_step(cfg, robot_state, action)
