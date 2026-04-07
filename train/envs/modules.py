import numpy as np

class Obstacle():
    def __init__(self,
                 pos,
                 radius,
                 ) -> None:
        self.pos = pos
        self.r = radius

class WAMV:
    def __init__(self, agility=1.0):
        self.dt = 0.05 # discretized time step (second)
        self.N = 4 # number of time step per action

        # WAM-V 16 simulation model
        self.length = 5.0
        self.width = 2.5
        self.detect_r = 0.5*np.sqrt(self.length**2+self.width**2) # detection range
        self.r = self.detect_r # collision range
        self.hull_width = 0.2 * self.width
        self.hull_tip_length = 0.25 * self.length
        self.hull_tip_width = 0.5 * self.hull_width
        self.hull_rear_length = 0.2 * self.length
        self.hull_rear_width = 0.8 * self.hull_width
        self.thruster_gap = 0.02 * self.length
        self.thruster_length = 0.1 * self.length
        self.thruster_tip_width = 0.6 * self.hull_rear_width
        self.thruster_rear_width = 0.75 * self.hull_rear_width
        self.beam_length = self.width - self.hull_width
        self.beam_width = 0.2 * self.hull_width
        self.beam_distance = 0.3 * self.length
        self.beam_base_length = 0.5 * self.length
        self.beam_base_width = 0.5 * self.hull_width
        self.platform_length = 0.4 * self.length
        self.platform_width = 0.45 * self.width

        self.agility = agility
        self.min_thrust = -500.0 * self.agility # min thrust force
        self.max_thrust = 1000.0 * self.agility # max thrust force

        # x-y-z (+): Forward-Starboard-Down (robot frame), North-East-Down (world frame)
        # yaw (+): clockwise
        self.x = None # x coordinate
        self.y = None # y coordinate
        self.theta = None # yaw angle used for
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.velocity_r = None # velocity wrt to current in world frame
        self.velocity = None # velocity wrt sea floor in world frame

        self.left_pos = None # left thruster angle (rad)
        self.right_pos = None # right thruster angle (rad)
        self.left_thrust = None # left thruster force (N)
        self.right_thrust = None # right thruster force (N)

        self.m = 400 # WAM-V weight when fully loaded (kg)
        self.Izz = 450 # moment of inertia Izz

        # hydrodynamic derivatives
        self.xDotU = 20
        self.yDotV = 0
        self.yDotR = 0
        self.nDotR = -980
        self.nDotV = 0
        self.xU = -100
        self.xUU = -150
        self.yV = -100
        self.yVV = -150
        self.yR = 0
        self.yRV = 0
        self.yVR = 0
        self.yRR = 0
        self.nR = -980
        self.nRR = -950
        self.nV = 0
        self.nVV = 0
        self.nRV = 0
        self.nVR = 0
        self.compute_constant_matrices() # ship maneuvering model matrices that are constant

        self.init_theta = 0.0
        self.init_velocity_r = np.array([0.0, 0.0, 0.0]) # relative velocity at initial position

        self.init_left_pos = 0.0 # left thruster angle at initial position
        self.init_right_pos = 0.0 # right thruster angle at initial position
        self.init_left_thrust = 0.0 # left thrust at initial position
        self.init_right_thrust = 0.0 # right thrust at initial position

    def compute_constant_matrices(self):
        self.M_RB = np.matrix([[self.m,0.0,0.0],[0.0,self.m,0.0],[0.0,0.0,self.Izz]])

        self.M_A = -1.0 * np.matrix([[self.xDotU,0.0,0.0],[0.0,self.yDotV,self.yDotR],
                                     [0.0,self.nDotV,self.nDotR]])

        self.D = -1.0 * np.matrix([[self.xU,0.0,0.0],[0.0,self.yV,self.yR],
                                   [0.0,self.nV,self.nR]])

    def reset(self, init_pos, init_theta, current_velocity=np.zeros(3)):
        # only called when resetting the environment
        self.x, self.y = init_pos
        self.theta = self.wrap_to_2pi(init_theta)
        self.pos = init_pos
        self.velocity_r = self.init_velocity_r
        self.update_velocity(current_velocity)
        self.left_pos = self.init_left_pos
        self.right_pos = self.init_right_pos
        self.left_thrust = self.init_left_thrust
        self.right_thrust = self.init_right_thrust
        self.min_thrust = -500.0 * self.agility # min thrust force
        self.max_thrust = 1000.0 * self.agility # max thrust force

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.x],[self.y]])
        return R_wr, t_wr

    def update_velocity(self, current_velocity=np.zeros(3)):
        self.velocity = self.velocity_r + current_velocity
        self.vel = self.velocity[0:2]

    def step(self, action, current_velocity=np.zeros(3)):
        # update thruster force
        self.left_thrust = np.clip(action[0], self.min_thrust, self.max_thrust)
        self.right_thrust = np.clip(action[1], self.min_thrust, self.max_thrust)

        for _ in range(self.N):

            # update robot pose in one time step
            self.update_velocity(current_velocity)
            dis = self.velocity * self.dt
            self.x += dis[0]
            self.y += dis[1]
            self.theta += dis[2]

            self.theta = self.wrap_to_2pi(self.theta)

            self.compute_motion()

        self.pos[0], self.pos[1] = self.x, self.y

    def compute_motion(self):
        # use 3 DOF ship maneuvering model from chapter 6.5 in Fossen's book
        velocity_r_b = self.project_to_robot_frame(self.velocity_r[:2])
        velocity_b = self.project_to_robot_frame(self.velocity[:2])
        u_r = velocity_r_b[0]
        v_r = velocity_r_b[1]
        u = velocity_b[0]
        v = velocity_b[1]
        r = self.velocity[2]
        C_RB = np.matrix([[0.0,-self.m*r,0.0],[self.m*r,0.0,0.0],[0.0,0.0,0.0]])
        C_A = np.matrix([[0.0,0.0,self.yDotV*v_r+self.yDotR*r],[0.0,0.0,-self.xDotU*u_r],
                         [-self.yDotV*v_r-self.yDotR*r,self.xDotU*u_r,0.0]])
        D_n = -1.0 * np.matrix([[self.xUU*np.abs(u_r),0.0,0.0],
                                [0.0,self.yVV*np.abs(v_r)+self.yRV*np.abs(r),self.yVR*np.abs(v_r)+self.yRR*np.abs(r)],
                                [0.0,self.nVV*np.abs(v_r)+self.nRV*np.abs(r),self.nVR*np.abs(v_r)+self.nRR*np.abs(r)]])
        N = C_A + self.D + D_n

        # compute propulsion forces and moment
        F_x_left = self.left_thrust * np.cos(self.left_pos)
        F_y_left = self.left_thrust * np.sin(self.left_pos)
        M_x_left = F_x_left * self.width/2
        M_y_left = -F_y_left * self.length/2

        F_x_right = self.right_thrust * np.cos(self.right_pos)
        F_y_right = self.right_thrust * np.sin(self.right_pos)
        M_x_right = -F_x_right * self.width/2
        M_y_right = -F_y_right * self.length/2

        F_x = F_x_left + F_x_right
        F_y = F_y_left + F_y_right
        M_n = M_x_left + M_y_left + M_x_right + M_y_right
        tau_p = np.matrix([[F_x],[F_y],[M_n]])

        # compute accelerations
        A = self.M_RB + self.M_A
        V = np.matrix([[u,v,r]]).transpose()
        V_r = np.matrix([[u_r,v_r,r]]).transpose()
        b = -C_RB*V - N*V_r + tau_p
        acc = np.linalg.inv(A.transpose()*A)*A.transpose()*b

        # apply accelerations to velocity
        V_r += acc * self.dt

        # project velocity to the world frame
        R_wr,_ = self.get_robot_transform()
        V_r[:2,:] = R_wr * V_r[:2,:]
        self.velocity_r = np.squeeze(np.array(V_r))

    def project_to_robot_frame(self,x,is_vector=True):
        assert isinstance(x,np.ndarray), "the input needs to be an numpy array"
        assert np.shape(x) == (2,)

        x_r = np.reshape(x,(2,1))

        R_wr, t_wr = self.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr

        if is_vector:
            x_r = R_rw * x_r
        else:
            x_r = R_rw * x_r + t_rw

        x_r.resize((2,))
        return np.array(x_r)

    def wrap_to_2pi(self, theta):
        while theta < 0.0:
            theta += 2 * np.pi
        while theta >= 2 * np.pi:
            theta -= 2 * np.pi
        return theta
