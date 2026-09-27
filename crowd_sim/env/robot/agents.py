

class RobotModel:
    def __init__(self, dt, radius=0.3, v_max=None, w_max=None, **kwargs):
        self.dt = dt
        self.state = None
        self.pos = None
        self.theta = None

        self.vel = None
        
        self.goal = None
        self.traj = None
        
        self.radius = None

        self.v_max = None
        self.w_max = None
        self.v_min = None
        self.w_min = None   
        self.a_max = None
        self.a_min = None

        self.type = None



    def reset(self, initial_pos):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError
    
    def get_pos(self):
        raise NotImplementedError
    
    def get_vel(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError
