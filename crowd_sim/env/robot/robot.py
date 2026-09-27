
import numpy as np
from gymnasium import spaces
from crowd_sim.env.robot.agents import RobotModel

def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)) or np.isscalar(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    
class SingleIntegrator(RobotModel):
    def __init__(self, dt, radius=0.3, umax=1.0):
        super().__init__(dt)
        self.radius = radius
        self.vmax = umax
        self.vmin = -self.vmax  # Assuming symmetric speed limits
        self.type = 'single_integrator'
        self.u = None

    def reset(self, initial_pos):
        # state: [x, y]
        self.state = np.array(initial_pos[0:2], dtype=np.float32)
        self.u = None
        self.pos = self.state[0:2]
    

    def nominal_input(self, relative_goal, theta, d_min=0.05, k_v=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        relative_goal = np.copy(relative_goal.reshape(-1, 1))
        pos_errors = -relative_goal[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > self.vmax:
            v_des = v_des * self.vmax / v_mag

        return v_des.reshape(-1, 1)
    

    
    def step(self, action):
        # action: [vx, vy]
        
        # Clip action magnitude.
        speed = np.linalg.norm(action)
        if speed > self.vmax:
            action = action / speed * self.vmax

        self.u = np.asarray(action, dtype=np.float32).reshape(-1)

        self.state += action * self.dt
        return self.state

    def get_state(self):
        return self.state
    
    def get_pos(self):
        return self.state[0:2]
    
    @property
    def action_space(self):
        # [vx, vy]
        return spaces.Box(low=self.vmin, high=self.vmax, shape=(2,), dtype=np.float32)

class Unicycle(RobotModel):
    def __init__(self, dt, radius=0.3, umax=[1.0, 1.68]):
        # umax: [vmax, w_max]
        super().__init__(dt)
        self.radius = radius
        self.type = 'unicycle'
    
        self.vmax = umax[0]
        self.vmin = -self.vmax  # Assuming symmetric speed limits
        self.w_max = umax[1]
        self.w_min = -self.w_max
        self.u = None


    def nominal_input(self, relative_goal, theta, d_min = 0.05, k_omega = 2.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        relative_goal = np.copy(relative_goal.reshape(-1,1))
        distance = max(np.linalg.norm(relative_goal[0:2,0]) - d_min, 0.05)
        theta_d = np.arctan2(-relative_goal[1,0],-relative_goal[0,0])
        error_theta = angle_normalize(theta_d - theta)

        omega = k_omega * error_theta   
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = k_v*( distance )*np.cos( error_theta )

        return np.array([v, omega]).reshape(-1,1)
    
    def nominal_input_SI(self, goal_rel, theta, d_min=0.05, k_v=1.0):
        # 1. Nominal Control (PD towards goal)
        u_nom = -k_v * goal_rel
        
        # Clip nominal control to vmax
        speed = np.linalg.norm(u_nom)
        # Handle scalar vs list umax
        v_limit = self.vmax if np.isscalar(self.vmax) else self.vmax[0]
        
        if speed > v_limit:
            u_nom = u_nom / speed * v_limit
        return u_nom
    
    def reset(self, initial_pos):
        # state: [x, y, theta]
        self.state = np.array(initial_pos[0:3], dtype=np.float32)
        self.u = np.zeros(2, dtype=np.float32)
        self.pos = self.state[0:2]

    
    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        v = float(a[0])
        omega = float(a[1])

        # Clip actions to the configured bounds.
        v = np.clip(v, self.vmin, self.vmax)
        omega = np.clip(omega, self.w_min, self.w_max)

        self.u = np.array([v, omega], dtype=np.float32)
        
        theta = self.state[2]
        
        # Update using kinematics
        self.state[0] += v * np.cos(theta) * self.dt
        self.state[1] += v * np.sin(theta) * self.dt
        self.state[2] += omega * self.dt
        
        # Normalize theta to [-pi, pi]
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return self.state

    def get_state(self):
        return self.state
    
    def get_pos(self):
        return self.state[0:2]
    
    def get_vel(self):
        # Velocity requires previous state and is not stored directly.
        raise NotImplementedError("Velocity computation requires previous state.")

    @property
    def action_space(self):
        # [v, omega] 
        # v in [-vmax, vmax], omega in [-wmax, wmax]
        return spaces.Box(
            low=np.array([self.vmin, self.w_min], dtype=np.float32),
            high=np.array([ self.vmax,  self.w_max], dtype=np.float32),
            dtype=np.float32
        )


class UnicycleDynamic(RobotModel):
    def __init__(self, dt, radius=0.3, umax=[1.0, 1.68, 3.0]):
        # umax: [vmax, w_max, acc_max]
        super().__init__(dt)
        self.radius = radius
        self.type = 'unicycle_dynamic'

        if np.isscalar(umax):
            self.vmax = umax
            self.acc_max = umax
            self.w_max = umax
        else:
            self.vmax = umax[0]
            self.w_max = umax[1]
            self.acc_max = umax[2]
        self.u = None

    def reset(self, initial_pos):
        # state: [x, y, theta, v]
        # initial_pos is [x, y, theta, v]
        self.state = np.array(initial_pos, dtype=np.float32)
        self.u = np.zeros(2, dtype=np.float32)
        self.pos = self.state[0:2]
        
        return self.state

    def step(self, action):
        # state is [x, y, theta, v], action is [omega, a], then:
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = omega
        # v_dot = a
        
        omega, a = action
        
        # Clip actions to dynamic limits.
        a = np.clip(a, -self.acc_max, self.acc_max)
        omega = np.clip(omega, -self.w_max, self.w_max)

        self.u = np.array([omega, a], dtype=np.float32)

        x, y, theta, v = self.state
        
        # Update State (Euler integration)
        # 1. Update Position
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        
        # 2. Update Velocity
        v_new = v + a * self.dt
        # Preserve current forward-speed convention.
        v_new = np.clip(v_new, 0.0, self.vmax)
        v_new = np.clip(v_new, -self.vmax, self.vmax)
        
        # 3. Update Heading
        theta_new = theta + omega * self.dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # Normalize
        
        self.state = np.array([x_new, y_new, theta_new, v_new], dtype=np.float32)
        return self.state

    def get_state(self):
        return self.state
    
    def get_pos(self):
        return self.state[0:2]
    
    def get_vel(self):
        theta, v = self.state[2], self.state[3]
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        return np.array([vx, vy])

    @property
    def action_space(self):
        # [angular_velocity, acceleration]
        return spaces.Box(
            low=np.array([-self.w_max, -self.acc_max], dtype=np.float32), 
            high=np.array([self.w_max, self.acc_max], dtype=np.float32), 
            dtype=np.float32
        )
