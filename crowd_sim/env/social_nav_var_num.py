import numpy as np

from crowd_sim.env.social_nav import SocialNav
from crowd_sim.env.robot.obstacle import SingleIntegrator as HumanIntegrator


class SocialNavVarNum(SocialNav):
    """
    Variant of SocialNav that initializes robot/humans using
    the sample_var_num_scene logic, but encapsulated as an instance method
    to avoid passing many self parameters.
    """
    def __init__(self, render_mode=None, config_file=None):
        super().__init__(render_mode=render_mode, config_file=config_file)

        human_cfg = config_file.env.humans if config_file is not None else {}
        robot_cfg = config_file.robot if config_file is not None else {}
        self.human_num_range = int(human_cfg.get("human_num_range", 0))
        # Matches initialization sampling below: px_noise/py_noise in [0, human_init_noise_range).
        self.human_init_noise_range = float(human_cfg.get("init_noise_range", 1.0))
        self.robot_ini_goal_dist = float(robot_cfg.get("ini_goal_dist", 4.0))

    def _init_robot_humans(self, options=None):
        rng = self.np_random

        if options and options.get("scenario") == "crossing":
            robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii = \
            self.crossing_init_robot_humans()
        else:
            # --- Sample robot state ---
            if self.robot_type == "unicycle":
                angle = rng.uniform(0, 2 * np.pi)
                px = self.arena_size * np.cos(angle)
                py = self.arena_size * np.sin(angle)
                while True:
                    gx, gy = rng.uniform(-self.arena_size, self.arena_size, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= self.robot_ini_goal_dist:
                        break
                theta = rng.uniform(0, 2 * np.pi)
            else:
                while True:
                    px, py, gx, gy = rng.uniform(-self.arena_size, self.arena_size, 4)
                    if np.linalg.norm([px - gx, py - gy]) >= self.robot_ini_goal_dist:
                        break
                theta = np.pi / 2

            robot_pos = np.array([px, py], dtype=float)
            goal_pos = np.array([gx, gy], dtype=float)
            robot_theta = float(theta)

            # --- Sample humans ---
            num_humans = self._sample_human_count()
            self._ensure_human_buffers(num_humans)

            human_states = []
            for _ in range(self.num_humans):
                human_vmax = rng.uniform(self.human_vmax_min, self.human_vmax)
                human_radius = rng.uniform(self.human_radius_min, self.human_radius_max)

                while True:
                    angle = rng.random() * np.pi * 2
                    noise_range = self.human_init_noise_range
                    px_noise = rng.uniform(-0.5 * noise_range, 0.5 * noise_range)
                    py_noise = rng.uniform(-0.5 * noise_range, 0.5 * noise_range)
                    hx = self.human_circle_radius * np.cos(angle) + px_noise
                    hy = self.human_circle_radius * np.sin(angle) + py_noise
                    collide = False

                    for i, agent in enumerate([{"px": px, "py": py, "gx": gx, "gy": gy, "radius": self.robot_radius}] + human_states):
                        if self.robot_type == "unicycle" and i == 0:
                            safe_dist_init = self.human_circle_radius / 2
                            agent_radius = self.robot_radius
                        else:
                            agent_radius = agent["radius"]
                            safe_dist_init = human_radius + agent_radius + self.discomfort_dist

                        if np.linalg.norm([hx - agent["px"], hy - agent["py"]]) < safe_dist_init or \
                                np.linalg.norm([hx - agent["gx"], hy - agent["gy"]]) < safe_dist_init:
                            collide = True
                            break

                    if not collide:
                        break

                human_states.append({
                    "px": hx,
                    "py": hy,
                    "gx": -hx,
                    "gy": -hy,
                    "radius": human_radius,
                    "v_pref": human_vmax,
                })

            human_positions = np.array([[h["px"], h["py"]] for h in human_states], dtype=float)
            human_goals = np.array([[h["gx"], h["gy"]] for h in human_states], dtype=float)
            human_vmaxs = np.array([h["v_pref"] for h in human_states], dtype=float)
            human_radii = np.array([h["radius"] for h in human_states], dtype=float)
        
            robot_vel = np.zeros(2, dtype=float)
            human_vels = np.zeros((self.num_humans, 2), dtype=float)

        return (
            robot_pos,
            goal_pos,
            robot_theta,
            robot_vel,
            human_positions,
            human_goals,
            human_vels,
            human_vmaxs,
            human_radii,
        )
    
    def _ensure_human_buffers(self, num_humans):
        if int(num_humans) == int(self.num_humans):
            return

        self.num_humans = int(num_humans)
        self.humans = [
            HumanIntegrator(self.dt, gmm_params=self.human_gmm_params)
            for _ in range(self.num_humans)
        ]
        self.human_positions = np.zeros((self.num_humans, 2), dtype=float)
        self.human_vels = np.zeros((self.num_humans, 2), dtype=float)
        self.human_goals = np.zeros((self.num_humans, 2), dtype=float)
        self.human_trajs = [[] for _ in range(self.num_humans)]
        self.human_traj_steps = []

    def _sample_human_count(self):
        if self.human_num_range <= 0:
            return self.num_humans

        if self.robot_type == "unicycle":
            low = 1
            high = self.num_humans + self.human_num_range + 1
        else:
            low = max(1, self.num_humans - self.human_num_range)
            high = self.num_humans + self.human_num_range + 1

        rng = self.np_random
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))

    def _sample_new_human_goal(self, idx):
        """
        Sample a new goal for one human following circle-based sampling with velocity-scaled noise.
        Collision checks are done against robot and other humans' current/goal positions.
        """
        v_pref = float(getattr(self.humans[idx], "vmax", 1.0))
        if v_pref <= 1e-8:
            return np.asarray(self.human_goals[idx], dtype=float).copy()
        noise_scale = 1.0 if v_pref <= 1e-8 else v_pref
        self_r = float(self.human_radii[idx])

        other_mask = np.ones((self.num_humans,), dtype=bool)
        other_mask[idx] = False
        other_positions = self.human_positions[other_mask]
        other_goals = self.human_goals[other_mask]
        other_radii = self.human_radii[other_mask]
        safe_dists = self_r + other_radii + float(self.discomfort_dist)

        for _ in range(100):
            angle = float(self.np_random.random() * np.pi * 2.0)
            gx_noise = (float(self.np_random.random()) - 0.5) * noise_scale
            gy_noise = (float(self.np_random.random()) - 0.5) * noise_scale
            gx = self.human_circle_radius * np.cos(angle) + gx_noise
            gy = self.human_circle_radius * np.sin(angle) + gy_noise
            candidate = np.array([gx, gy], dtype=float)

            collide = False
            min_dist_robot = self_r + self.robot_radius + self.discomfort_dist
            if np.linalg.norm(candidate - self.robot_pos) < min_dist_robot or \
               np.linalg.norm(candidate - self.goal_pos) < min_dist_robot:
                collide = True
            if collide:
                continue

            if other_positions.shape[0] > 0:
                dist_to_positions = np.linalg.norm(other_positions - candidate, axis=1)
                dist_to_goals = np.linalg.norm(other_goals - candidate, axis=1)
                collide = bool(np.any((dist_to_positions < safe_dists) | (dist_to_goals < safe_dists)))

            if not collide:
                return candidate

        # Fallback: keep old goal to avoid invalid updates.
        return np.asarray(self.human_goals[idx], dtype=float).copy()
