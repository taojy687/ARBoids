import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from envs.modules import Obstacle, WAMV
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

class TADEnv():
    def __init__(self,
                 defender_num=3,
                 boid_state=True,
                 form_reward=True,
                 LearningSide='Def',
                 ):

        # TAD Parameters
        self.Target_R = 15.0
        self.Sensing_R = 60.0
        self.Defend_R = 5.0
        self.Collision_R = 5.0
        self.Att_Sensing_R = 15.0

        # Simulation Parameters
        self.Total_T = 80.0
        self.Action_T = 0.2
        self.Current_T = 0.0
        self.LearningSide = LearningSide

        # APF Parameters
        self.Obs_R = 8.0

        # RL Env Parameters
        self.form_reward = form_reward
        self.boid_state = boid_state
        self.teammate_dim = (defender_num - 1) * 2
        self.feature1_dim = 6   # [d_Ti, phi_Ti, d_Ai, phi_Ai, v_A, phi_A]
        if self.boid_state:
            self.feature2_dim = 8   # [f_sep, phi_sep, f_ali, phi_ali, f_coh, phi_coh, a_boids, w_boids]
        else:
            self.feature2_dim = 0
        self.state_dim = self.teammate_dim + self.feature1_dim + self.feature2_dim
        self.action_dim = 2

        self.max_thrust, self.min_thrust = 1000.0, -500.0

        self.attacker_num = 1
        self.attacker = WAMV()

        self.defender_num = defender_num
        self.defender_list = [WAMV() for _ in range(self.defender_num)]    

    def reset(self, 
              agility=2.0,
              noisy_agility=False):
        '''
            Reset the whole environment
        '''
        # Reset environment states
        self.Current_T = 0.0

        init_radius = np.random.uniform(self.Sensing_R, self.Sensing_R + 5)
        init_theta = np.random.uniform(0.0, 2 * np.pi)

        # Reset attacker
        init_pos = init_radius * np.array([np.cos(init_theta), np.sin(init_theta)])
        if noisy_agility:
            agility = np.random.uniform(agility - 0.25, agility + 0.25)
        self.attacker.agility = agility
        self.attacker.reset(init_pos, -init_theta)
        self.att_action = np.zeros(2)

        # Reset defender
        init_theta = np.random.uniform(-np.pi, np.pi)
        index = 2.0 * np.pi / self.defender_num
        heading = np.random.uniform(-np.pi, np.pi)
        for i, defender in enumerate(self.defender_list):
            radius = np.random.uniform(7.0, 8.0)
            theta = init_theta + index * i
            init_pos = radius * np.array([np.cos(theta), np.sin(theta)])
            defender.reset(init_pos, heading + np.random.uniform(-np.pi / 3, np.pi / 3))

        # Store states
        self.def_att_dists = np.zeros(self.defender_num)
        self.def_def_dists = np.zeros((self.defender_num, self.defender_num - 1))

        self.Pos_Att = np.zeros(2 * self.attacker_num)
        self.Phi_Att = np.zeros(self.attacker_num)

        self.Pos_Att[0 : 2] = self.attacker.pos
        self.Phi_Att = self.attacker.theta

        self.Pos_Def = np.zeros(2 * self.defender_num)
        self.Phi_Def = np.zeros(self.defender_num)
        for i, defender in enumerate(self.defender_list):
            self.Pos_Def[2*i : 2*i+2] = defender.pos
            self.Phi_Def[i] = defender.theta

        # Defender step
        goal = self.attacker.pos
        positions = np.zeros((self.defender_num, 2))
        velocities = np.zeros((self.defender_num, 2))
        phis = np.zeros(self.defender_num)
        for i, defender in enumerate(self.defender_list):
            positions[i, :] = defender.pos
            velocities[i, :] = defender.vel
            phis[i] = defender.theta
        self._Boid_navi_step(positions, velocities, phis, goal)
        
        observations, observation = self._get_obs()
        self.Rewards = self._get_rewards(done=0)

        return observations, observation

    def step(self,
             rl_action:np.ndarray,
             controller='Boids',
             att_action=None,
             ):
        '''
            Environment step function
        
            Args:
                rl_action: [defender_num, 2]
                controller: Boids, RL, Res, AdaRes
                att_action: Attacker's action 
        '''
        if controller == 'RL':
            actions = self.action_to_thrust(rl_action)
        elif controller == 'Res':
            actions = np.clip(self.action_to_thrust(rl_action) + self.boids_actions, self.min_thrust, self.max_thrust)
        elif controller == 'AdaRes':
            actions = self.action_to_thrust(np.copy(rl_action[:, :2]))
            adas = rl_action[:, 2]
            for i in range(self.defender_num):
                actions[i] = adas[i] * actions[i] + (1 - adas[i]) * self.boids_actions[i]
        else:
            actions = self.boids_actions

        # Attacker Step
        goal = np.zeros(2)
        obstacles = [Obstacle(pos=defender.pos, radius=self.Obs_R) 
                    for defender in self.defender_list]
        pos_att = np.zeros(2 * self.attacker_num)
        phi_att = np.zeros(self.attacker_num)
        if att_action is None:
            self.att_action = self._APF_navi_step(self.attacker.pos, goal, obstacles, self.attacker.theta)
        else:
            self.att_action = self.action_to_thrust(att_action, self.attacker.agility)

        self.attacker.step(self.att_action, self.generate_random_current())
        pos_att[0:2] = self.attacker.pos
        phi_att = self.attacker.theta
        
        self.Pos_Att = np.vstack([self.Pos_Att, pos_att])
        self.Phi_Att = np.vstack([self.Phi_Att, phi_att])

        # Defender Step
        pos_def = np.zeros(self.defender_num * 2)
        phi_def = np.zeros(self.defender_num)
            
        for i, defender in enumerate(self.defender_list):
            defender.step(actions[i], self.generate_random_current())
            pos_def[2*i : 2*i+2] = defender.pos
            phi_def[i] = defender.theta
        
        self.Pos_Def = np.vstack([self.Pos_Def, pos_def])
        self.Phi_Def = np.vstack([self.Phi_Def, phi_def])

        # Get Observations
        goal = self.attacker.pos
        positions = np.zeros((self.defender_num, 2))
        velocities = np.zeros((self.defender_num, 2))
        phis = np.zeros(self.defender_num)
        for i, defender in enumerate(self.defender_list):
            positions[i, :] = defender.pos
            velocities[i, :] = defender.vel
            phis[i] = defender.theta
        self._Boid_navi_step(positions, velocities, phis, goal)
        observations, observation = self._get_obs()
        self.att_action = self.thrust_to_action(self.att_action, self.attacker.agility)

        # isTermination
        self.Current_T += self.Action_T
        done = self._isTerminate()

        # Get Rewards
        rewards = self._get_rewards(done)
        self.Rewards = np.vstack([self.Rewards, rewards])
        
        return observations, rewards, done, observation
    
    def generate_random_current(self,):
        '''
            Generate Random Current
        '''
        current_velocity = np.zeros(3)
        current_velocity[:2] += np.random.normal(0.0, 0.2, 2)
        current_velocity[2] += np.random.normal(0.0, 0.02)
        return current_velocity

    def _get_obs(self,):
        '''
            Get observation array

            Return (Defender, LearningSide = 'Def')
                observations: [defender_num, 14]
                obs_d = [d_Ti, phi_Ti, d_Ai, phi_Ai, v_A, phi_A, f_sep, phi_sep,
                        f_ali, phi_ali, f_coh, phi_coh, a_boids, w_boids]

            Return (Attacker)
                def_i = [d_Ai, phi_Ai]
                obs_a = [d_T, phi_T, def_1, def_2, def_3]
        '''
        def _calculate_dist_phi(vector, theta):
            dist = np.linalg.norm(vector).item()
            phi = np.arctan2(vector[1] ,vector[0]) - theta
            if phi > np.pi:
                phi -= 2 * np.pi
            elif phi < -np.pi:
                phi += 2 *np.pi
            return dist, phi

        observations = np.zeros((self.defender_num, self.state_dim))
        index = self.feature1_dim+self.feature2_dim

        for i, defender in enumerate(self.defender_list):
            theta = defender.theta
            observations[i, 0:2] = _calculate_dist_phi(-defender.pos, theta)

            dist, phi = _calculate_dist_phi(self.attacker.pos - defender.pos, theta)
            observations[i, 2:4] = dist, phi
            self.def_att_dists[i] = dist

            observations[i, 4:6] = _calculate_dist_phi(self.attacker.vel, theta)

            if self.boid_state:
                observations[i, 6:8] = _calculate_dist_phi(self.boids_states[i, 0:2], theta)
                observations[i, 8:10] = _calculate_dist_phi(self.boids_states[i, 2:4], theta)
                observations[i, 10:12] = _calculate_dist_phi(self.boids_states[i, 4:6], theta)
                observations[i, 12:14] = self.thrust_to_action(self.boids_actions[i])

            k = 0
            for j, teammate in enumerate(self.defender_list):
                if j == i:
                    continue
                else:
                    dist, phi = _calculate_dist_phi(teammate.pos - defender.pos, theta)
                    observations[i, index+2*k:index+2+2*k] = dist, phi
                    self.def_def_dists[i, k] = dist
                    k += 1

        observation = np.zeros(8)
        if self.LearningSide == 'Att':
            observation[0:2] = _calculate_dist_phi(-self.attacker.pos, self.attacker.theta)

            defender_pos = np.array([defender.pos for defender in self.defender_list])
            att_def_vec = defender_pos - self.attacker.pos
            self.def_att_dists = np.linalg.norm(att_def_vec, axis=1)
            sort_index = np.argsort(self.def_att_dists)
            
            for i in range(self.defender_num):
                index = sort_index[i]
                if self.def_att_dists[index] < self.Att_Sensing_R:
                    observation[2*i+2:2*i+4] = _calculate_dist_phi(att_def_vec[index], self.attacker.theta)

        return observations, observation

    
    def _isTerminate(self,):
        '''
            Check if the game is terminated

            Return
                0: not terminate
                1: attacker reach
                2: defender collide
                3: defender capture
                4: time out
        '''
        att_tar_dist = np.linalg.norm(self.attacker.pos)
        def_tar_dists = [np.linalg.norm(defender.pos) for defender in self.defender_list]
        # Since the attacker has superior maneuverability over the defenders, once the attacker
        # is closer to the target than all defenders, the defenders can no longer intercept it
        # in time. This is therefore treated as an immediate attacker success.
        if att_tar_dist < self.Target_R or all(att_tar_dist < dist for dist in def_tar_dists):
            return 1
        
        if (self.def_def_dists < self.Collision_R).any():
            return 2
        
        if (self.def_att_dists < self.Defend_R).any():
            return 3
        
        elif self.Current_T > self.Total_T - 1e-5:
            return 4
        else:
            return 0

    def _get_rewards(self, done):
        '''
            Reward Function

            Args:
                done: isTerminate
        '''
        rewards = np.zeros(self.defender_num)
        defender_pos = np.array([defender.pos for defender in self.defender_list])
        def_att_vec = self.attacker.pos - defender_pos
        dists = np.linalg.norm(def_att_vec, axis=1)

        if self.LearningSide == 'Def':
            # Defender Reward
            if done == 0:
                '''Step Reward'''
                # Formation Cost
                if self.form_reward:
                    tar_att_vec = self.attacker.pos / np.linalg.norm(self.attacker.pos)
                    def_att_vec = def_att_vec / np.vstack([dists, dists]).T
                    mean_def_att_vec = np.sum(def_att_vec, axis=0)
                    mean_def_att_vec_norm = np.linalg.norm(mean_def_att_vec)
                    mean_def_att_vec /= max(mean_def_att_vec_norm, 1e-6)
                    rewards += 2.0 * (0.5 * np.dot(tar_att_vec, mean_def_att_vec.T) - 1.0 * mean_def_att_vec_norm / self.defender_num)

            elif done == 1:
                # Attacker Reach Cost
                rewards -= 100.0
            elif done == 2:
                # Collision Cost
                rewards -= 100.0 * np.sum(self.def_def_dists < self.Collision_R, axis=1)
            elif done > 2:
                rewards += 50.0 * (dists < 3 * self.Defend_R) + 50.0 * (dists < self.Defend_R)
            return rewards

        else:
            # Attacker Reward
            reward = 0.0
            defender_pos = np.array([defender.pos for defender in self.defender_list])
            def_att_vec = self.attacker.pos - defender_pos
            dists = np.linalg.norm(def_att_vec, axis=1)
            min_dist = np.min(dists)
            att_tar_dist = np.linalg.norm(self.attacker.pos)

            # Avoid Reward
            if min_dist < self.Att_Sensing_R:
                reward -= 5.0 / min_dist
            
            # Attraction Reward
            reward += 0.5 * np.cos(self.attacker.theta + np.arctan2(self.attacker.pos[1], self.attacker.pos[0]))
            reward -= 0.1 * att_tar_dist

            # Main Reward
            if done == 1:
                reward += 100
            elif done > 2:
                reward -= 100
            return reward
    
    def force_to_thrust(self, force, phi, robot='def'):
        """Convert potential field force to thruster output"""
        force = force / np.linalg.norm(force)
        rotation = np.array([
            [np.cos(phi), np.sin(phi)],
            [-np.sin(phi), np.cos(phi)]
        ])
        control = np.dot(rotation, force)
        acc = control[0]
        ang = np.arctan2(control[1], control[0])
        if robot == 'def':
            max_thrust = self.defender_list[0].max_thrust
            min_thrust = self.defender_list[0].min_thrust
        else:
            max_thrust = self.attacker.max_thrust
            min_thrust = self.attacker.min_thrust
        l = (acc + 0.75 * ang) * max_thrust
        r = (acc - 0.75 * ang) * max_thrust
        return np.clip(np.array([l, r]), min_thrust, max_thrust)

    def action_to_thrust(self, action:np.ndarray, agility:float=1.0):
        return agility * (action * (self.max_thrust - self.min_thrust) + self.max_thrust + self.min_thrust) / 2.0
    
    def thrust_to_action(self, thrust:np.ndarray, agility:float=1.0):
        return (thrust * 2.0 - self.max_thrust - self.min_thrust) / (self.max_thrust - self.min_thrust) / agility

    def _APF_navi_step(self, 
                       position,
                       goal,
                       obstacles,
                       phi,
                       robot='att'
                       ):
        '''
            Artificial Potential Field Navigation
            
            Args:
                position: Robot's current position [x, y]
                goal: Robot's goal position [x, y]
                obstacles: List of obstacles [obs1, obs2, ...]
                phi: Robot's heading
        '''
        k_att = 0.1
        goal_direction = (goal - position)
        att_force = k_att * goal_direction
        goal_direction = goal_direction / np.linalg.norm(goal_direction)

        k_rep = 3000.0
        influence_radius = 50.0
        rep_force = np.zeros(2)
        for obs in obstacles:
            distance = max(np.linalg.norm(position - obs.pos) - obs.r, 0.1)
            if distance < influence_radius:
                obs_direction = position - obs.pos
                obs_direction = obs_direction / np.linalg.norm(obs_direction)
                perpendicular_direction = np.array([-obs_direction[1], obs_direction[0]]) 

                mag = k_rep * (1.0 / distance - 1.0 / influence_radius) / (distance ** 2)
                base_force = mag * obs_direction

                # Adjust repulsion direction: combine obstacle direction and normal direction
                alpha = 0.0  # Adjustment factor, controls weight of normal component
                adjusted_force = (1 - alpha) * base_force + alpha * mag * perpendicular_direction
                rep_force += adjusted_force

                # if use closest point
                influence_radius = distance

        force = att_force + rep_force + np.random.normal(0.0, 0.3, 2)
        return self.force_to_thrust(force, phi, robot)
    
    def _Boid_navi_step(self,
                        positions,
                        velocities,
                        phis,
                        goal,
                        robot='def'):
        # Parameters
        neighbor_radius = 15.0

        k_att = 0.5
        k_sep = 10.0
        k_ali = 0.1
        k_coh = 0.1

        self.boids_forces = np.zeros_like(positions)
        self.boids_actions = np.zeros((self.defender_num, self.action_dim))
        self.boids_states = np.zeros((positions.shape[0], positions.shape[1] * 3))
        for i, pos in enumerate(positions):
            dists = np.linalg.norm(positions - pos, axis=1)
            neighbors = dists < neighbor_radius
            neighbors[i] = False  # Exclude itself

            # Attraction: move towards target
            attraction = goal - pos

            # Separation: move away from too close neighbors
            separation = np.zeros(2)
            for j, is_neighbor in enumerate(neighbors):
                if is_neighbor:
                    diff = pos - positions[j]
                    dist = max(np.linalg.norm(diff) - 3.5, 0.01)
                    separation += diff / (dist ** 2)

            # Alignment: align with neighbors' direction
            alignment = np.mean(velocities, axis=0)

            # Cohesion: move towards neighbors' center
            cohesion = np.mean(positions, axis=0) - pos

            self.boids_forces[i] = k_att * attraction + k_sep * separation + k_ali * alignment + k_coh * cohesion
            self.boids_states[i] = [*separation, *alignment, *cohesion]
            self.boids_actions[i] = self.force_to_thrust(self.boids_forces[i], phis[i], robot)

    def plot_scenario(self,
                      save=False,
                      filename='videos/exp0.mp4'):

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

        target = plt.Circle((0, 0), self.Target_R, color='green', fill=True)
        robot_shape = np.array([
            [3.8, 0.0],
            [1.8, 1.8],
            [-1.8, 1.8],
            [-1.8, -1.8],
            [1.8, -1.8]
        ])

        def update(frame):
            ax.clear()
            ax.add_patch(target)

            # Plot Defenders
            for i in range(self.defender_num):
                phi = self.Phi_Def[frame, i]
                rotation = np.array([
                    [np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]
                ])
                defender = Polygon(
                            self.Pos_Def[frame, 2*i+0 : 2*i+2] + robot_shape @ rotation, 
                            closed=True, color='blue')
                ax.add_patch(defender)
                ax.plot(self.Pos_Def[:frame, 2*i+0], 
                        self.Pos_Def[:frame, 2*i+1], 
                        color='b', label='Defender')
            
            # Plot Attacker
            phi = self.Phi_Att[frame, 0]
            rotation = np.array([
                [np.cos(phi), np.sin(phi)],
                [-np.sin(phi), np.cos(phi)]
            ])
            attacker = Polygon(
                        self.Pos_Att[frame, :] + robot_shape @ rotation, 
                        closed=True, color='r')
            ax.add_patch(attacker)
            ax.plot(self.Pos_Att[:frame, 0], 
                    self.Pos_Att[:frame, 1], 
                    color='r', label='Attacker')
            
            ax.set_title('TAD Game')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([-self.Sensing_R - 15, self.Sensing_R + 15])
            ax.set_ylim([-self.Sensing_R - 15, self.Sensing_R + 15])
            ax.legend()

        ani = FuncAnimation(fig, update, frames=self.Pos_Def.shape[0], 
                            repeat=False, interval=20)
        if save:
            ani.save(filename)

        plt.show()
    
    def save(self, filename='data1.mat'):
        data = {
            'Pos_Def': self.Pos_Def,
            'Phi_Def': self.Phi_Def,
            'Pos_Att': self.Pos_Att,
            'Phi_Att': self.Phi_Att,
        }
        scipy.io.savemat(filename, data)

if __name__ == "__main__":
    env = TADEnv(defender_num=3,
                 boid_state=True,
                 form_reward=True,
                 LearningSide='Def')
    def_win_num = 0

    n = 100
    for _ in range(n):
        env.reset(agility=2.0)
        done = False
        while not done:
            action = np.zeros(2)
            controller = 'Boids'
            s, r, done, _ = env.step(action, controller)
        if done == 1:
            print('Attacker Win')
        elif done == 2:
            print('Defender Collision')
        elif done > 2:
            print('Defender Win')
            def_win_num += 1
        print('Mean Reward =', env.Rewards.mean())

    print('Defender SR =', def_win_num / n)
    # env.save()
    env.plot_scenario(save=False)
