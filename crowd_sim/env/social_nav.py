import gymnasium as gym
from gymnasium import spaces
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from crowd_sim.env.robot.robot import SingleIntegrator, Unicycle
from crowd_sim.env.robot.obstacle import SingleIntegrator as HumanIntegrator
from crowd_nav.policy.social_force_helper import SocialForceHelper
from crowd_sim.utils import sample_point_in_disk


class SocialNav(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, config_file=None):
        super(SocialNav, self).__init__()
        self.render_mode = render_mode
        self.show_human_traj = False
        
        # Load params directly from the composed Hydra config.
        env_cfg = config_file.env
        self.dt = env_cfg['dt']
        self.max_steps = env_cfg['max_steps']
        self.sensing_radius = float(env_cfg.get('sensing_radius', 5.0))

        # --- 0. Initialize Robot Model ---
        robot_cfg = config_file.robot
        self.robot_radius = robot_cfg['radius']
        self.robot_pos = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.robot_theta = 0.0
        self.goal_pos = np.zeros(2)
        self.robot_traj = []
        self.render_safe_dist = None
        self.robot_ini_goal_dist = float(env_cfg.get("robot_ini_goal_dist", robot_cfg.get("ini_goal_dist", 6.0)))

        self.robot_type = robot_cfg['type']
        if self.robot_type == 'single_integrator':
            robot_u_max = robot_cfg['vmax']
            self.robot = SingleIntegrator(self.dt, self.robot_radius, umax=robot_u_max)
        elif self.robot_type == 'unicycle':
            # input is v and omega, so robot_u_max is 2 dimensions: [v_max, w_max]
            robot_u_max = [robot_cfg['vmax'], robot_cfg['omega_max']]
            self.robot = Unicycle(self.dt, self.robot_radius, umax=robot_u_max)
        elif self.robot_type == 'unicycle_dynamic':
            raise NotImplementedError("unicycle_dynamic is not supported")
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")
            

        # --- 1. Initialize Humans ---
        human_cfg = config_file.env.humans
        self.num_humans = int(human_cfg.get('num_humans', 1))
        # Frozen at init: max possible humans across episodes (accounts for var-num
        # environments where self.num_humans mutates per reset). The env always emits
        # _obs_human_cap blocks; unused slots are zero-padded with mask=0.
        _hnr = int(human_cfg.get('human_num_range', 0))
        self._obs_human_cap = self.num_humans + max(0, _hnr)

        self.human_radii = np.zeros(self.num_humans, dtype=float)  
        self.human_positions = np.zeros((self.num_humans, 2), dtype=float)
        self.human_vels = np.zeros((self.num_humans, 2), dtype=float)
        self.human_goals = np.zeros((self.num_humans, 2), dtype=float)
        self.human_trajs = [[] for _ in range(self.num_humans)]
        self.human_traj_steps = []
        self.human_vmaxs = np.zeros(self.num_humans, dtype=float) 

        self.human_vmax_min, self.human_vmax = map(float, human_cfg["vmax"])
        self.human_radius_min = float(human_cfg["radius"])
        self.human_radius_max = float(human_cfg["radius"])

        self.human_gmm_params = dict(human_cfg.get("gmm", {}))
        self.humans = [
            HumanIntegrator(self.dt, gmm_params=self.human_gmm_params)
            for _ in range(self.num_humans)
        ]
    
        self.arena_size = human_cfg.get('arena_size', 6.0)
        self.human_circle_radius = self.arena_size * np.sqrt(2) 

        self.human_policy_name = human_cfg.get('policy', 'nominal')
        self.human_use_gmm = bool(human_cfg.get('use_gmm', True))
        self.random_goal_changing = bool(human_cfg.get('random_goal_changing', False))
        self.goal_change_chance = float(np.clip(human_cfg.get('goal_change_chance', 0.0), 0.0, 1.0))
        self.end_goal_changing = bool(human_cfg.get('end_goal_changing', False))
        self.end_goal_change_chance = float(np.clip(human_cfg.get('end_goal_change_chance', 1.0), 0.0, 1.0))
        self.current_scenario = None

        self.sf_params = human_cfg.get('sf', {})
        self.sf_helper = None
        if self.human_policy_name == 'social_force':
            self.sf_helper = SocialForceHelper(
                dt=self.dt,
                sf_params=self.sf_params,
                max_humans=self.num_humans,
            )

        # --- Base Observation (absolute protocol) ---
        # Robot+goal block: [rx, ry, gx, gy, rvx, rvy, theta, r_radius] -> 8 dims
        # Per-human block:  [hx, hy, hvx, hvy, hradius, mask]            -> 6 dims
        # Total: 8 + _obs_human_cap * 6
        # The env exposes one slot per possible human; downstream code (network
        # or controller) selects how many to actually attend to via obs_top_k.
        self.obs_dim = 8 + self._obs_human_cap * 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = self.robot.action_space

        # --- Reward Parameters (CrowdNav++) ---
        self.current_step = 0
        reward_cfg = config_file.env.reward
        self.success_reward = reward_cfg['success_reward']
        self.collision_penalty = reward_cfg['collision_penalty']
        self.discomfort_dist = reward_cfg['discomfort_dist']
        self.discomfort_penalty_factor = reward_cfg['discomfort_penalty_factor']
        self.potential_factor = reward_cfg['potential_factor']
        self.back_factor = reward_cfg['back_factor']
        self.spin_factor = reward_cfg['spin_factor']
        self.constant_penalty = reward_cfg['constant_penalty']
        # Optional smooth safety shaping near humans (disabled by default).
        self.safe_shaping_weight = reward_cfg.get('safe_shaping_weight', 0.0)
        self.safe_shaping_band = reward_cfg.get('safe_shaping_band', 0.6)

        self.max_abs_pot_reward = self.dt * self.robot.vmax * self.potential_factor
        # maximum absolute value of rotation penalty (only meaningful for unicycle robots)
        if self.robot_type == 'unicycle':
            self.max_abs_rot_penalty = self.spin_factor * (max(abs(self.robot.w_min), self.robot.w_max) * self.dt) ** 2
        else:
            self.max_abs_rot_penalty = 0.0
        self.max_abs_back_penalty = self.back_factor * max(abs(self.robot.vmin), self.robot.vmax)

    def reset(self, seed=None, options=None):
        # 1. Basic Environment Reset & Seeding
        super().reset(seed=seed)
        self.current_step = 0
        self.current_scenario = options.get("scenario") if isinstance(options, dict) else options
        
        # 2. Initialize Trajectory Storage
        self.robot_traj = []
        self.human_trajs = [[] for _ in range(self.num_humans)]
        self.human_traj_steps = []
        self._render_limits = None
        self.render_safe_dist = None
        
        # 3. Initialize Robot State (Position, Goal, Theta)
        (
            self.robot_pos,
            self.goal_pos,
            self.robot_theta,
            self.robot_vel,
            self.human_positions,
            self.human_goals,
            self.human_vels,
            self.human_vmaxs,
            self.human_radii,
        ) = self._init_robot_humans(options=options)
        self._seed_social_force_model()
        if self.human_use_gmm:
            self._seed_human_noise_models()

        # 4. Reset Internal Dynamics Models
        # Robot: [x, y, theta, v]
        self.robot.reset([self.robot_pos[0], self.robot_pos[1], self.robot_theta, 0.0])
        # Humans
        for i, human in enumerate(self.humans):
            human.vmax = self.human_vmaxs[i]
            human.radius = self.human_radii[i]
            human.reset(self.human_positions[i])
        
        # 6. Initialize Metrics & History
        self.prev_dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        self.robot_traj.append(self.robot_pos.copy())
        self.human_traj_steps.append(self.human_positions.copy())
            
        return self._get_obs(), {}
    
    def _seed_human_noise_models(self, seed=None):
        max_seed = np.iinfo(np.uint32).max
        if seed is not None:
            base_seed = int(seed) % max_seed
        else:
            base_seed = int(self.np_random.integers(0, max_seed))
        for i, human in enumerate(self.humans):
            human.gmm.set_seed((base_seed + i) % max_seed)

    def _seed_social_force_model(self, seed=None):
        if self.sf_helper is None:
            return

        max_seed = np.iinfo(np.uint32).max
        if seed is not None:
            sf_seed = int(seed) % max_seed
        else:
            sf_seed = int(self.np_random.integers(0, max_seed))
        self.sf_helper.set_seed(sf_seed)

    def step(self, action):
        # 1. robot position update
        self._update_robot_state(action)

        # 2. human position update (with optional goal changing)
        self._update_human_states()

        # 3. Hard wall clamp: keep robot/humans inside [-half_extent, half_extent].
        self._clip_positions_to_wall()
        
        # 4. calculate distances
        dist_to_goal, min_dist = self._compute_distances(self.human_positions)
        
        # 5. calculate reward and done
        reward, done, info = self._compute_reward_and_done(dist_to_goal, min_dist, self.robot.u)
            
        return self._get_obs(), reward, done, False, info

    def _get_wall_half_extent(self):
        return float(self.human_circle_radius + getattr(self, "human_init_noise_range", 0.0))

    def _clip_positions_to_wall(self):
        half_extent = self._get_wall_half_extent()
        lo, hi = -half_extent, half_extent

        np.clip(self.robot_pos, lo, hi, out=self.robot_pos)
        np.clip(self.human_positions, lo, hi, out=self.human_positions)

        # Keep internal robot state and trajectory buffers aligned with clipped positions.
        if hasattr(self, "robot_state") and self.robot_state is not None:
            self.robot_state[0:2] = self.robot_pos
        if self.robot_traj:
            self.robot_traj[-1] = self.robot_pos.copy()
        if self.human_traj_steps:
            self.human_traj_steps[-1] = self.human_positions.copy()

    def _init_robot_humans(self, options=None):
        if options and options.get("scenario") == "crossing":
            robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii = \
            self.crossing_init_robot_humans()
        else:
            def sample_arena_point():
                return self.np_random.uniform(-self.arena_size, self.arena_size, 2)

            for _ in range(500):
                robot_pos = sample_arena_point()
                goal_pos = sample_arena_point()
                if np.linalg.norm(robot_pos - goal_pos) >= self.robot_ini_goal_dist:
                    break
            else:
                raise RuntimeError(
                    "Failed to sample a valid robot start/goal pair in SocialNav"
                )
            robot_theta = self.np_random.uniform(-np.pi, np.pi)

            min_pair_dist = 6.0
            human_positions = np.zeros((self.num_humans, 2), dtype=float)
            human_goals = np.zeros((self.num_humans, 2), dtype=float)

            safe_dist_init = self.robot_radius + self.human_radius_max + self.discomfort_dist

            def valid_human_pair(p, g):
                return (
                    np.linalg.norm(p - g) >= min_pair_dist
                    and np.linalg.norm(robot_pos - p) >= safe_dist_init
                    and np.linalg.norm(goal_pos - g) >= safe_dist_init
                )

            for i in range(self.num_humans):
                found = False
                for _ in range(50):
                    p = sample_point_in_disk(
                        self.np_random, center=[0.0, 0.0], radius=self.human_circle_radius, arena_size=self.arena_size
                    )
                    g = sample_point_in_disk(
                        self.np_random, center=[0.0, 0.0], radius=self.human_circle_radius, arena_size=self.arena_size
                    )
                    if valid_human_pair(p, g):
                        human_positions[i] = p
                        human_goals[i] = g
                        found = True
                        break

                if not found:
                    for _ in range(500):
                        p = sample_arena_point()
                        g = sample_arena_point()
                        if valid_human_pair(p, g):
                            human_positions[i] = p
                            human_goals[i] = g
                            found = True
                            break
                    if not found:
                        raise RuntimeError(
                            f"Failed to sample a valid human start/goal pair in SocialNav for human {i}"
                        )

            human_vmaxs = self.np_random.uniform(self.human_vmax_min, self.human_vmax, size=self.num_humans)
            human_radii = self.np_random.uniform(self.human_radius_min, self.human_radius_max, size=self.num_humans)
            human_vels = np.zeros((self.num_humans, 2), dtype=float)
            robot_vel = np.zeros(2)

        return robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii

    def _update_human_states(self):
        # Optional human goal update.
        self._update_obstacle_goals()

        if self.human_policy_name == 'social_force':
            nominal_actions = self.sf_helper.action_for_humans(
                human_positions=self.human_positions,
                human_vels=self.human_vels,
                human_goals=self.human_goals,
                human_radii=self.human_radii,
                human_vprefs=self.human_vmaxs,
                robot_pos=self.robot_pos,
                robot_vel=self.robot_vel,
                robot_radius=self.robot_radius,
                robot_vpref=float(getattr(self.robot, "vmax", 1.0)),
            )
        else:
            nominal_actions = np.zeros((self.num_humans, 2), dtype=np.float32)
            for i, human in enumerate(self.humans):
                nominal_actions[i] = human.nominal_controller(self.human_positions[i], self.human_goals[i])

        if self.human_use_gmm:
            exec_actions = HumanIntegrator.apply_gmm_batch(
                humans=self.humans,
                nominal_actions=nominal_actions,
                states=self.human_positions,
                goals=self.human_goals,
                radii=self.human_radii,
            )
        else:
            exec_actions = np.asarray(nominal_actions, dtype=np.float32)

        # 3. Human state update (vectorized).
        self.human_vels, self.human_positions = HumanIntegrator.step_batch(
            actions=exec_actions,
            vmaxs=self.human_vmaxs,
            positions=self.human_positions,
            dt=self.dt,
        )
        self.human_traj_steps.append(self.human_positions.copy())

    def _update_robot_state(self, action):
        # action is whatever the robot model expects natively:
        # single_integrator -> [vx, vy], unicycle -> [v, omega].
        self.robot.step(action)

        self.robot_state = self.robot.get_state()
        if self.robot_type == 'single_integrator':
            self.robot_pos = self.robot_state
            self.robot_vel = self.robot.u
            self.robot_theta = 0.0
        elif self.robot_type == 'unicycle':
            self.robot_pos = self.robot_state[0:2]
            self.robot_vel = np.array([
                self.robot.u[0] * np.cos(self.robot_state[2]),
                self.robot.u[0] * np.sin(self.robot_state[2])
            ])
            self.robot_theta = self.robot_state[2]
        elif self.robot_type == 'unicycle_dynamic':
            self.robot_pos = self.robot_state[0:2]
            v = self.robot_state[3]
            self.robot_vel = np.array([
                v * np.cos(self.robot_state[2]),
                v * np.sin(self.robot_state[2])
            ])
            self.robot_theta = self.robot_state[2]
        self.robot_traj.append(self.robot_pos.copy())

    def _compute_distances(self, human_positions=None):
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        targets = human_positions if human_positions is not None else self.human_positions
        if len(targets) == 0:
            min_clearance = float('inf')
        else:
            rel = targets - self.robot_pos
            dists = np.linalg.norm(rel, axis=1)
            radii = self.human_radii if self.human_radii.size == dists.size else np.full(dists.size, self.human_radius_max)
            clearances = dists - self.robot_radius - radii
            min_clearance = float(np.min(clearances))

        return dist_to_goal, min_clearance

    def _compute_reward_and_done(self, dist_to_goal, min_dist, u=None):
        reward = 0
        done = False
        info = {"is_success": False, "is_collision": False, "is_timeout": False}
        goal_threshold = self._goal_success_threshold()

        self.current_step += 1
        if self.current_step >= self.max_steps and not done:
            done = True
            info["is_timeout"] = True
            reward = 0
        elif min_dist < 0:
            reward = self.collision_penalty
            done = True
            info["is_collision"] = True
        elif dist_to_goal < goal_threshold:
            reward = self.success_reward
            done = True
            info["is_success"] = True

        elif min_dist < self.discomfort_dist:
            reward = (min_dist - self.discomfort_dist) * self.discomfort_penalty_factor * self.dt
            done = False
        else:
            potential_reward = self.potential_factor * (self.prev_dist_to_goal - dist_to_goal)
            reward = potential_reward
            reward = np.clip(reward, -self.max_abs_pot_reward, self.max_abs_pot_reward)
            done = False
            
        self.prev_dist_to_goal = dist_to_goal

        # Add rotational and backward-motion penalties.
        r_spin = -self.spin_factor * (u[1] * self.dt) ** 2
        r_spin = np.clip(r_spin, -self.max_abs_rot_penalty, self.max_abs_rot_penalty)

        if u[0] < 0:
            r_back = -self.back_factor * abs(u[0])
        else:
            r_back = 0.
        r_back = np.clip(r_back, -self.max_abs_back_penalty, self.max_abs_back_penalty)

        reward = reward + r_spin + r_back + self.constant_penalty
        
        # Scale reward to keep magnitudes stable during RL training.
        reward = reward / 10.0

        return reward, done, info

    def _goal_success_threshold(self):
        if self.robot_type == 'unicycle':
            return 0.6
        return float(self.robot.radius)

    def _goal_render_radius(self):
        if self.robot_type == 'single_integrator':
            return float(self.robot.radius)
        return max(self._goal_success_threshold() - float(self.robot.radius), 0.0)

    def _render_inflated_robot_radius(self):
        render_safe_dist = getattr(self, "render_safe_dist", None)
        if render_safe_dist is not None:
            try:
                total_safe_dist = float(render_safe_dist)
            except (TypeError, ValueError):
                total_safe_dist = np.nan
            if np.isfinite(total_safe_dist):
                human_ref_radius = float(self.human_radius_max) if self.num_humans > 0 else 0.0
                return max(total_safe_dist - human_ref_radius, float(self.robot_radius))

        return float(self.robot_radius)

    def _compute_safe_shaping_reward(self, min_dist):
        """
        Smooth penalty when clearance is close to the safety boundary.
        Returns 0 when far enough; negative value as clearance shrinks.
        """
        if self.safe_shaping_weight <= 0.0:
            return 0.0

        band = max(float(self.safe_shaping_band), 1e-6)
        start = float(self.discomfort_dist)

        if min_dist >= start + band:
            return 0.0

        x = np.clip((start + band - float(min_dist)) / band, 0.0, 1.0)
        smooth = x * x * (3.0 - 2.0 * x)  # smoothstep in [0, 1]
        return -float(self.safe_shaping_weight) * float(smooth)

    def _get_obs(self):
        # --- Robot+goal absolute block ---
        robot_state = np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.robot_vel[0], self.robot_vel[1],
            self.robot_theta,
            self.robot_radius
        ], dtype=float)

        k = self._obs_human_cap
        obs_blocks = np.zeros((k, 6), dtype=float)  # dummy blocks: [0, 0, 0, 0, 0, 0]

        # Use robot-relative distances for visibility/ranking only.
        rel = self.robot_pos - self.human_positions
        dists = np.linalg.norm(rel, axis=1)
        visible_idx = np.where(dists <= self.sensing_radius)[0]
        if visible_idx.size > 0:
            order = np.argsort(dists[visible_idx])
            selected = visible_idx[order[:k]]
            rows = np.arange(selected.size)
            obs_blocks[rows, 0:2] = self.human_positions[selected]
            obs_blocks[rows, 2:4] = self.human_vels[selected]
            obs_blocks[rows, 4] = self.human_radii[selected]
            obs_blocks[rows, 5] = 1.0  # mask

        obs = np.concatenate([robot_state, obs_blocks.reshape(-1)]).astype(np.float32)
        return obs

    def render(self):
        if self.render_mode is None:
            return
        plt = self._get_plt()

        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7.4, 7.4), facecolor="white")
            self.fig.subplots_adjust(left=0.022, right=0.997, bottom=0.065, top=0.935)
            if self.render_mode == "human":
                plt.ion()
                plt.show()

        self.ax.clear()
        start_pos = np.asarray(self.robot_traj[0] if len(self.robot_traj) > 0 else self.robot_pos, dtype=float)
        robot_traj_arr = np.asarray(self.robot_traj, dtype=float) if len(self.robot_traj) > 0 else np.empty((0, 2), dtype=float)
        human_traj_arr = np.asarray(self.human_traj_steps, dtype=float) if len(self.human_traj_steps) > 0 else np.empty((0, self.num_humans, 2), dtype=float)
        self.ax.set_xlim(-6.0, 6.0)
        self.ax.set_ylim(-6.0, 6.0)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("white")
        robot_color = "#3b82f6"
        inflated_robot_color = "#60a5fa"
        traj_cmap = LinearSegmentedColormap.from_list("robot_traj", ["#dbeafe", robot_color])
        human_traj_cmap = LinearSegmentedColormap.from_list("human_traj", ["#ffffff", "#8c8c8c"])
        inflated_robot_radius = self._render_inflated_robot_radius()

        closest_idx = None
        if self.num_humans > 0:
            center_dists = np.linalg.norm(self.human_positions - self.robot_pos, axis=1)
            clearances = center_dists - self.human_radii - float(self.robot_radius)
            closest_idx = int(np.argmin(clearances))

        # Draw obstacle trajectories first so current positions stay legible.
        if self.show_human_traj and human_traj_arr.ndim == 3 and human_traj_arr.shape[0] > 1:
            traj_norm = Normalize(vmin=0.0, vmax=1.0)
            seg_values = np.linspace(0.0, 1.0, human_traj_arr.shape[0] - 1, dtype=float)
            for i in range(self.num_humans):
                human_points = human_traj_arr[:, i, :]
                if human_points.shape[0] <= 1:
                    continue
                points = human_points.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                human_line_collection = LineCollection(
                    segments,
                    cmap=human_traj_cmap,
                    norm=traj_norm,
                    linewidths=2.4,
                    alpha=0.9,
                    zorder=1,
                )
                human_line_collection.set_array(seg_values)
                human_line_collection.set_capstyle("round")
                self.ax.add_collection(human_line_collection)

        # Draw obstacles first so the robot path stays readable on top.
        for i in range(self.num_humans):
            is_closest = (closest_idx is not None and i == closest_idx)
            face_color = "#a64040" if is_closest else "#8c8c8c"
            human = plt.Circle(
                self.human_positions[i],
                self.human_radii[i],
                facecolor=face_color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.95,
                zorder=2,
            )
            self.ax.add_artist(human)

        # Draw robot trajectory in a compact trajectory-plot style.
        if robot_traj_arr.shape[0] > 1:
            points = robot_traj_arr.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            traj_norm = Normalize(vmin=0.0, vmax=1.0)
            seg_values = np.linspace(0.0, 1.0, len(segments), dtype=float)
            line_collection = LineCollection(
                segments,
                cmap=traj_cmap,
                norm=traj_norm,
                linewidths=4.4,
                alpha=0.98,
                zorder=4,
            )
            line_collection.set_array(seg_values)
            line_collection.set_capstyle("round")
            self.ax.add_collection(line_collection)

        if inflated_robot_radius > float(self.robot_radius):
            inflated_robot = plt.Circle(
                self.robot_pos,
                inflated_robot_radius,
                facecolor=inflated_robot_color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.18,
                zorder=5.5,
            )
            self.ax.add_artist(inflated_robot)

        # Current robot pose.
        robot = plt.Circle(
            self.robot_pos,
            self.robot_radius,
            facecolor=robot_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.82,
            zorder=6,
        )
        self.ax.add_artist(robot)

        goal_radius = self._goal_render_radius()

        # Start / goal markers and legend.
        self.ax.scatter(
            [start_pos[0]],
            [start_pos[1]],
            s=420,
            marker="^",
            c="#ead94c",
            edgecolors="none",
            linewidths=0.0,
            zorder=8,
        )
        goal = plt.Circle(
            self.goal_pos,
            goal_radius,
            facecolor="#ead94c",
            edgecolor="#d2bc38",
            linewidth=1.2,
            alpha=0.65,
            zorder=7,
        )
        self.ax.add_artist(goal)

        start_handle = plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="None",
            markerfacecolor="#ead94c",
            markeredgecolor="none",
            markersize=9,
            label="Start",
        )
        goal_handle = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#ead94c",
            markeredgecolor="#d2bc38",
            markeredgewidth=1.0,
            markersize=9,
            label="Goal",
        )
        traj_handle = plt.Line2D(
            [0], [0],
            color=robot_color,
            linewidth=4.4,
            alpha=0.98,
            label="Trajectory",
        )

        self.ax.legend(
            handles=[start_handle, goal_handle, traj_handle],
            loc='upper right',
            frameon=True,
            fontsize=16,
            borderpad=0.3,
            handletextpad=0.4,
        )
        self.ax.grid(False)
        self.ax.tick_params(labelsize=18, colors="black")
        current_time = float(self.current_step) * float(self.dt)
        self.ax.set_title(f"Time: {current_time:.1f}s", fontsize=20, pad=7, color="black")

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.01) # Slow down visualization
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            # Prefer buffer_rgba as it handles high-DPI (Retina) scaling properly by preserving shape
            if hasattr(self.fig.canvas, "buffer_rgba"):
                rgba = np.asarray(self.fig.canvas.buffer_rgba())
                return rgba[:, :, :3].copy()
            # Fallback for older backends
            if hasattr(self.fig.canvas, "tostring_rgb"):
                data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return data

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt = self._get_plt()
            plt.close(self.fig)
            self.fig = None

    @staticmethod
    def _get_plt():
        import matplotlib.pyplot as plt
        return plt

    def _update_obstacle_goals(self):
        """
        Optional stochastic goal update for humans (obstacles).
        Two independent events:
        1) random_goal_changing: can trigger regardless of goal reaching status.
        2) end_goal_changing: can trigger when the human reaches its current goal.
        """
        if self.current_scenario == "crossing":
            return
        if not (self.random_goal_changing or self.end_goal_changing):
            return
        n = int(self.num_humans)
        if n <= 0:
            return

        # Humans with effectively zero preferred speed do not change goals.
        active = np.asarray(self.human_vmaxs, dtype=float).reshape(n) > 1e-8
        if not np.any(active):
            return

        random_trigger = np.zeros((n,), dtype=bool)
        if self.random_goal_changing:
            random_vals = self.np_random.random(n)
            random_trigger = random_vals <= float(self.goal_change_chance)

        end_goal_trigger = np.zeros((n,), dtype=bool)
        if self.end_goal_changing:
            dist_to_goal = np.linalg.norm(self.human_positions - self.human_goals, axis=1)
            near_goal = dist_to_goal <= (self.human_radii + 0.1)
            end_vals = self.np_random.random(n)
            end_goal_trigger = near_goal & (end_vals <= float(self.end_goal_change_chance))

        trigger = active & (random_trigger | end_goal_trigger)
        triggered_idx = np.flatnonzero(trigger)
        for i in triggered_idx:
            self.human_goals[i] = self._sample_new_human_goal(int(i))

    def _sample_new_human_goal(self, idx):
        # Sample goal uniformly in disk with only one hard constraint:
        # ||current_position - new_goal|| >= 6.0
        p = np.asarray(self.human_positions[idx], dtype=float)
        min_pair_dist = 6.0
        r = float(self.human_circle_radius)

        for _ in range(50):
            candidate = sample_point_in_disk(
                self.np_random, center=[0.0, 0.0], radius=r, arena_size=self.arena_size
            )
            if np.linalg.norm(p - candidate) >= min_pair_dist:
                return candidate

        # Fallback: put goal on circle boundary opposite to current position.
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-8:
            candidate = -(p / p_norm) * r
            if np.linalg.norm(p - candidate) >= min_pair_dist:
                return candidate
        return np.asarray(self.human_goals[idx], dtype=float).copy()
    
    def crossing_init_robot_humans(self):
        robot_pos = np.array([-5.5, 0.0])
        goal_pos = np.array([5.5, 0.0])
        robot_theta = 0.0

        base_y = 0.0
        offsets = np.linspace(-1.0, 1.0, self.num_humans) if self.num_humans > 1 else np.array([0.0])
        human_positions = np.zeros((self.num_humans, 2), dtype=float)
        human_goals = np.zeros((self.num_humans, 2), dtype=float)
        for i in range(self.num_humans):
            human_positions[i] = np.array([5.0, base_y + offsets[i]])
            human_goals[i] = np.array([-5.0, base_y + offsets[i]])

        human_vmaxs = np.full((self.num_humans,), 1.8, dtype=float)
        human_radii = np.full((self.num_humans,), 1.0, dtype=float)
        human_vels = np.zeros((self.num_humans, 2), dtype=float)
        robot_vel = np.zeros(2)
        return robot_pos, goal_pos, robot_theta, robot_vel, human_positions, human_goals, human_vels, human_vmaxs, human_radii
