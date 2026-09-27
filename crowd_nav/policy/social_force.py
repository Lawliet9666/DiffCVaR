import numpy as np
from crowd_nav.policy.policy import Policy


class SocialForce(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'social_force'
        self.KI = self.config.sf.KI
        self.A = self.config.sf.A
        self.B = self.config.sf.B
        self.rng = np.random.default_rng(0)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(int(seed))

    def predict(self, state):
        return self.predict_from_states(state.self_state, state.human_states)

    def predict_from_states(self, self_state, human_states):
        """
        Produce action for agent with circular specification of social force model.
        """
        # Pull force to goal
        delta_x = self_state.gx - self_state.px
        delta_y = self_state.gy - self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * self_state.v_pref if dist_to_goal > 1e-5 else 0.0
        desired_vy = (delta_y / dist_to_goal) * self_state.v_pref if dist_to_goal > 1e-5 else 0.0

            
        B = max(self.B, 1e-5) # Prevent division by zero
            
        time_step = self.config.env.dt

        curr_delta_vx = self.KI * (desired_vx - self_state.vx)
        curr_delta_vy = self.KI * (desired_vy - self_state.vy)
        
        # Push force(s) from other agents
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in human_states:
            delta_x = self_state.px - other_human_state.px
            delta_y = self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            
            if dist_to_human > 1e-5:
                force_magnitude = self.A * np.exp((self_state.radius + other_human_state.radius - dist_to_human) / B)
                interaction_vx += force_magnitude * (delta_x / dist_to_human)
                interaction_vy += force_magnitude * (delta_y / dist_to_human)
            else:
                # If agents are exactly overlapping, apply a random repulsive force to separate them
                force_magnitude = self.A * np.exp((self_state.radius + other_human_state.radius) / B)
                angle = self.rng.uniform(-np.pi, np.pi)
                interaction_vx += force_magnitude * np.cos(angle)
                interaction_vy += force_magnitude * np.sin(angle)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = self_state.vx + total_delta_vx
        new_vy = self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > self_state.v_pref:
            scale = self_state.v_pref / act_norm
            return np.array([new_vx * scale, new_vy * scale], dtype=np.float32)
        return np.array([new_vx, new_vy], dtype=np.float32)
