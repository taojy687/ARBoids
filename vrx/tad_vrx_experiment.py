import launch
import rclpy
import time
from std_msgs.msg import Empty, Float64
import threading
import numpy as np
import subprocess
import math
import argparse
import os

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.actions import EmitEvent
from launch.events import Shutdown

from tf2_msgs.msg import TFMessage

import torch
from models import ActorSAC, ActorAdap
from utils import NPZLogger

def force_to_thruster(force, phi, is_attacker=True, min_thrust=-500.0, max_thrust=1000.0):
    force = force / np.linalg.norm(force)
    rotation = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])
    control = np.dot(rotation, force)
    acc = control[0]
    ang = np.arctan2(control[1], control[0])
    if is_attacker:
        l = (acc + 2.5 * ang) * max_thrust
        r = (acc - 2.5 * ang) * max_thrust
    else:
        l = (acc + 0.75 * ang) * max_thrust
        r = (acc - 0.75 * ang) * max_thrust
    return np.clip(np.array([l, r]), min_thrust, max_thrust)

def APF_navi_control(position, goal, obstacles, phi, min_thrust=-500.0, max_thrust=1000.0, improved=True):
    '''
        Artificial Potential Field Navigation
        
        Args:
            position: Robot's current position [x, y]
            goal: Robot's goal position [x, y]
            obstacles: List of obstacles' positions [[x1, y1], [x2, y2] ...]
            phi: Robot's heading
    '''
    if improved:
        alpha = 0.1
    else:
        alpha = 0.0

    k_att = 0.1
    goal_direction = goal - position
    att_force = k_att * goal_direction
    goal_direction = goal_direction / np.linalg.norm(goal_direction)

    k_rep = 2000.0
    influence_radius = 20.0
    rep_force = np.zeros(2)
    for obs in obstacles:
        distance = max(np.linalg.norm(position - obs) - 7.0, 0.1)
        if distance < influence_radius:
            obs_direction = position - obs
            obs_direction = obs_direction / np.linalg.norm(obs_direction)
            perpendicular_direction = np.array([-obs_direction[1], obs_direction[0]])

            mag = k_rep * (1.0 / distance - 1.0 / influence_radius) / (distance ** 2)
            base_force = mag * obs_direction

            adjusted_force = (1 - alpha) * base_force + alpha * mag * perpendicular_direction
            rep_force += adjusted_force

            influence_radius = distance

    force = att_force + rep_force
    return force_to_thruster(force, phi, True, min_thrust, max_thrust)

def Boids_navi_control(positions, velocities, phis, goal, min_thrust=-500.0, max_thrust=1000.0):
    '''
        Boids Model Navigation
        
        Args:
            positions: USV swarm's current positions [num, 2]
            velocities: USV swarm's current velocities [num, 2]
            phis: USV swarm's headings 
    '''

    # Parameters
    neighbor_radius = 15.0

    k_att = 0.5
    k_sep = 10.0
    k_ali = 0.1
    k_coh = 0.1

    boids_forces = np.zeros_like(positions)
    boids_actions = np.zeros((positions.shape[0], 2))
    boids_states = np.zeros((positions.shape[0], positions.shape[1] * 3))
    for i, pos in enumerate(positions):
        dists = np.linalg.norm(positions - pos, axis=1)
        neighbors = dists < neighbor_radius
        neighbors[i] = False    # Exclude itself

        attraction = goal - pos

        seperation = np.zeros(2)
        for j, is_neighbor in enumerate(neighbors):
            if is_neighbor:
                diff = pos - positions[j]
                dist = max(np.linalg.norm(diff) - 3.5, 0.01)
                seperation += diff / (dist ** 2)
        
        alignment = np.mean(velocities, axis=0)

        cohesion = np.mean(positions, axis=0) - pos

        boids_forces[i] = k_att * attraction + k_sep * seperation + k_ali * alignment + k_coh * cohesion
        boids_states[i] = [*seperation, *alignment, *cohesion]
        boids_actions[i] = force_to_thruster(boids_forces[i], phis[i], False, min_thrust, max_thrust)

    return boids_actions, boids_states
    
def thrust_to_action(thrust:np.ndarray, min_thrust=-500.0, max_thrust=1000.0):
    action = (thrust * 2.0 - max_thrust - min_thrust) / (max_thrust - min_thrust)
    return np.clip(action, -1.0, 1.0)

def action_to_thrust(action:np.ndarray, min_thrust=-500.0, max_thrust=1000.0):
    action = np.clip(action, -1.0, 1.0)
    return (action * (max_thrust - min_thrust) + max_thrust + min_thrust) / 2.0

def RL_navi_control(actor, observations, boids_actions=None, controller='AdaRes',
                    device=torch.device('cpu'), min_thrust=-500.0, max_thrust=1000.0):
    with torch.no_grad():
        s = torch.tensor(observations, dtype=torch.float).to(device)
        a, _ = actor(s, True, False)
        a = a.cpu().numpy().flatten()
    actions = a.reshape(observations.shape[0], -1)
    if controller == 'Res':
        actions = action_to_thrust(actions) + boids_actions
    elif controller == 'AdaRes':
        adas = actions[:, 2]
        actions = action_to_thrust(np.copy(actions[:, :2]))
        for i in range(len(adas)):
            actions[i] = adas[i] * actions[i] + (1 - adas[i]) * boids_actions[i]
    elif controller == 'RL':
        actions = action_to_thrust(actions)
    return np.clip(actions, min_thrust, max_thrust)

class ExperimentManager:
    def __init__(self, num_robots, save_traj=False, save_file='exp_traj.npz', device='cpu'):
        self.ls = None
        self.lock = threading.Lock()
        self.unpause_signal_node = rclpy.create_node('unpause_signal_node')
        self.unpause_signal_publishers = []
        
        self.defend_r = 5.5
        self.collision_r = 5.0
        self.target_r = 15.0
        self.sensing_r = 60.0
        self.total_time = 100.0
        self.origin = np.array([-448.7194, 234.3858])
        self.num_robots = num_robots

        self.save_traj = save_traj
        self.save_file = save_file
        self.timestamp_data = {}
        self.pose_data = {}
        self.vel_data = {}

        self.curr_time = 0.0
        self.curr_pos = np.zeros((num_robots, 2))
        self.curr_phi = np.zeros(num_robots)
        self.curr_vel = np.zeros((num_robots, 2))

        self.def_att_dists = np.zeros(self.num_robots - 1)
        self.def_def_dists = np.zeros((self.num_robots - 1, self.num_robots - 2))

        self.device = torch.device(device)

        for i in range(self.num_robots):
            self.unpause_signal_publishers.append(self.unpause_signal_node.create_publisher(Empty, 
                                                  f'/wamv{i+1}/unpause_signal', 10))
            self.timestamp_data[f'wamv{i+1}'] = []
            self.pose_data[f'wamv{i+1}'] = []
            self.vel_data[f'wamv{i+1}'] = []
        
        if self.save_traj:
            self.logger = NPZLogger(self.save_file)
        
    def generate_init_info(self, agility=2.0, setting=0):
        '''
            Generate initial positions

            Args:
                agility: attacker's agility level
                setting: 0 -> ocean setting (default); 
                         1 -> dock setting
        '''
        init_poses = ""
        self.agility = agility

        positions = np.zeros((self.num_robots - 1, 2))
        phis = np.zeros(self.num_robots - 1)
        goal = np.zeros(2)

        
        if setting == 0:
            self.origin = np.array([-448.7194, 234.3858])

            # Attacker init pose
            init_radius = np.random.uniform(self.sensing_r, self.sensing_r + 5)
            init_theta = np.random.uniform(0.0, 2 * np.pi)
            goal = init_radius * np.array([np.cos(init_theta), np.sin(init_theta)]) 
            init_pos = goal + self.origin
            self.curr_pos[0] = init_pos - self.origin

            init_poses += str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(-init_theta)

            # Defenders init poses
            def_theta = np.random.uniform(-np.pi, np.pi)
            index = 2 * np.pi / (self.num_robots - 1)

            for i in range(self.num_robots - 1):
                radius = np.random.uniform(7.0, 8.0)
                theta = def_theta + index * i

                positions[i] = radius * np.array([np.cos(theta), np.sin(theta)])
                init_pos = positions[i] + self.origin
                self.curr_pos[i+1] = positions[i]

                phis[i] = init_theta + np.random.uniform(-np.pi / 3, np.pi / 3)
                init_poses += ';' + str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(phis[i])
        
        elif setting == 1:
            self.origin = np.array([-576.2809, 270.6814])

            # Attacker init pose
            init_radius = np.random.uniform(self.sensing_r, self.sensing_r + 5)
            init_theta = np.random.uniform(0, np.pi / 4)
            goal = init_radius * np.array([np.cos(init_theta), np.sin(init_theta)]) 
            init_pos = goal + self.origin
            self.curr_pos[0] = init_pos - self.origin

            init_poses += str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(-init_theta)

            # Defenders init poses
            def_theta = init_theta + np.pi / 3
            index = 2 * np.pi / (self.num_robots - 1)

            for i in range(self.num_robots - 1):
                radius = np.random.uniform(8.0, 9.0)
                theta = def_theta + index * i

                positions[i] = radius * np.array([np.cos(theta), np.sin(theta)])
                init_pos = positions[i] + self.origin
                self.curr_pos[i+1] = positions[i]

                phis[i] = init_theta + np.random.uniform(-np.pi / 3, np.pi / 3)
                init_poses += ';' + str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(phis[i])

        boids_actions, boids_states = Boids_navi_control(positions, np.zeros((self.num_robots - 1, 2)), phis, goal)
        self.get_observations(positions, phis, goal, np.zeros(2), boids_states, boids_actions)
        
        return init_poses   

    def launch_simulation(self, init_poses, world_name, headless=False,
                          controller='AdaRes', modelname='adares1.pth'):
        # Create launch description
        ld = launch.LaunchDescription()

        # Add action to launch competition environment
        competition_launch_file = launch.actions.ExecuteProcess(
            cmd= ['ros2', 'launch', 'vrx_gz', 'tad.launch.py', 
                 "init_poses:="+init_poses, "world:="+world_name,
                 "headless:="+str(headless)],
            output='screen'
        )

        vrx_exit_event_handler = RegisterEventHandler(
            OnProcessExit(
                target_action=competition_launch_file,\
                on_exit=[
                    EmitEvent(event=Shutdown(reason='VRX Sim Ended'))
                ]
                
                )
        )

        ld.add_action(competition_launch_file)
        ld.add_action(vrx_exit_event_handler)

        self.unpause_signal_thread = threading.Thread(target=self.start_unpause_signal_thread)
        self.unpause_signal_thread.start()

        self.robot_info_thread = threading.Thread(target=self.start_robot_info_subscribers)
        self.robot_info_thread.start()

        self.robot_action_thread = threading.Thread(target=self.start_robot_action_controllers, args=(controller, modelname))
        self.robot_action_thread.start()

        self.experiment_monitoring_thread = threading.Thread(target=self.experiment_monitoring)
        self.experiment_monitoring_thread.start()

        # Launch simulation
        ls = launch.LaunchService()
        ls.include_launch_description(ld)

        self.ls = ls.run()

        self.unpause_signal_thread.join()
        self.robot_info_thread.join()
        self.experiment_monitoring_thread.join()

    def start_robot_info_subscribers(self):
        self.robot_info_node = rclpy.create_node('robot_info_node')
        self.robot_info_subscribers = []

        for i in range(self.num_robots):
            self.robot_info_subscribers.append(self.robot_info_node.create_subscription(
                                               TFMessage, f'/wamv{i+1}/pose', self.robot_info_callback, 10))

        executor = rclpy.executors.MultiThreadedExecutor(num_threads=1)
        executor.add_node(self.robot_info_node)
        executor.spin()
    
    def start_robot_action_controllers(self, controller='AdaRes', modelname='adares-iapf1.pth'):
        self.robot_action_node = rclpy.create_node('robot_action_node')
        self.robot_action_publishers = []
        
        for i in range(self.num_robots):
            self.robot_action_publishers.append(self.robot_action_node.create_publisher(
                Float64, f'/wamv{i+1}/thrusters/left/thrust', 10
            ))
            self.robot_action_publishers.append(self.robot_action_node.create_publisher(
                Float64, f'/wamv{i+1}/thrusters/right/thrust', 10
            ))
        
        def control_loop(controller='AdaRes',
                         modelname='adares-iapf1.pth',
                         boids_state=True):
            '''
                Define the USV controller

                Args:
                    controller: Types include 'AdaRes', 'Res', 'RL', 'Boids'
                    modelname: File path
                    boids_state: 
            '''
            
            feature1_dim = 6
            if boids_state:
                feature2_dim = 8
            else:
                feature2_dim = 0
            if controller == 'Res' or controller == 'RL':
                actor = ActorSAC(feature1_dim, feature2_dim, 2, hidden_dim=512)
                actor.load(modelname)
                actor.to(self.device)
            elif controller == 'AdaRes':
                actor = ActorAdap(feature1_dim, feature2_dim, 3, hidden_dim=512)
                actor.load(modelname)
                actor.to(self.device)

            time.sleep(15.0)
            thrust_limits = np.array([-500.0, 1000.0])
            while rclpy.ok():
                with self.lock:
                    actions = np.zeros((self.num_robots, 2))

                    # Attacker control
                    position = self.curr_pos[0]
                    phi = self.curr_phi[0]
                    goal = np.zeros(2)
                    obstacles = self.curr_pos[1:, :]
                    min_thrust, max_thrust = thrust_limits * self.agility
                    actions[0] = APF_navi_control(position, goal, obstacles, phi, min_thrust, max_thrust)

                    # Defenders control
                    positions = self.curr_pos[1:, :]
                    velocities = self.curr_vel[1:, :]
                    phis = self.curr_phi[1:]
                    goal = self.curr_pos[0, :]
                    att_vel = self.curr_vel[0]

                    boids_actions, boids_states = Boids_navi_control(positions, velocities, phis, goal)
                    if controller == 'Boids':
                        actions[1:, :] = boids_actions
                    else:
                        observations = self.get_observations(positions, phis, goal, att_vel, boids_states, boids_actions)
                        actions[1:, :] = RL_navi_control(actor, observations, boids_actions, controller, self.device)

                    for i in range(self.num_robots):
                        l_thrust = Float64()
                        r_thrust = Float64()
                        l_thrust.data, r_thrust.data = actions[i, 0], actions[i, 1]

                        self.robot_action_publishers[2*i].publish(l_thrust)
                        self.robot_action_publishers[2*i+1].publish(r_thrust)
                    
                    new_entry = {
                        'timestamp': self.curr_time,
                        'AttPos': self.curr_pos[0],
                        'AttPhi': self.curr_phi[0],
                        'AttVel': self.curr_vel[0],
                        'AttAct': actions[0],
                        'DefPos': self.curr_pos[1:].flatten(),
                        'DefPhi': self.curr_phi[1:].flatten(),
                        'DefVel': self.curr_vel[1:].flatten(),
                        'DefAct': actions[1:].flatten(),
                    }

                    if self.save_traj:
                        self.logger.log(new_entry)

                time.sleep(0.1)

        executor = rclpy.executors.MultiThreadedExecutor(num_threads=1)
        executor.add_node(self.robot_action_node)

        robot_action_spin_thread = threading.Thread(target=executor.spin, daemon=True)
        robot_action_spin_thread.start()

        control_thread = threading.Thread(target=control_loop, daemon=True, args=(controller, modelname,))
        control_thread.start()

    def start_unpause_signal_thread(self):
        # Send unpause signal 10 seconds after the launch
        time.sleep(20)
        self.start_time = time.time()
        msg = Empty()
        while rclpy.ok():
            for publisher in self.unpause_signal_publishers:
                publisher.publish(msg)
            time.sleep(0.05)

    def robot_info_callback(self,msg:TFMessage):
        def quaternion_to_euler(q):
            x, y, z, w = q.x, q.y, q.z, q.w

            # Roll
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return roll, pitch, yaw
    
        def wrap_to_pi(theta):
            while theta <= -math.pi:
                theta += 2 * math.pi
            while theta >= math.pi:
                theta -= 2 * math.pi
            return theta

        def calculate_vel(array):
            if len(array) < 2:
                return [0.0, 0.0]
            else:
                dt = 0.05
                curr_pos = np.array([array[-1][0], array[-1][1]])
                last_pos = np.array([array[-2][0], array[-2][1]])
                return (curr_pos - last_pos) / dt

        for transform_stamped in msg.transforms:
            child_frame = transform_stamped.child_frame_id   
            timestamp = transform_stamped.header.stamp              # sec, nanosec       
            translation = transform_stamped.transform.translation   # [x, y, z]
            rotation = transform_stamped.transform.rotation         # [x, y, z, w]
            if child_frame in self.pose_data.keys():
                index = int(child_frame[-1]) - 1
                timestamp = timestamp.sec + timestamp.nanosec * 1e-9
                self.timestamp_data[child_frame].append(timestamp)

                x, y = translation.x - self.origin[0], translation.y - self.origin[1]
                phi = wrap_to_pi(quaternion_to_euler(rotation)[2])
                vel = calculate_vel(self.pose_data[child_frame])
                self.curr_time = timestamp
                self.curr_pos[index] = [x, y]
                self.curr_phi[index] = phi
                self.curr_vel[index] = vel
            
                self.pose_data[child_frame].append([x, y, phi])
                self.vel_data[child_frame].append(vel)

    def get_observations(self, positions, phis, goal, goal_vel, boids_states=None, boids_actions=None):
        def _calculate_dist_phi(vector, theta):
            dist = np.linalg.norm(vector)
            phi = np.arctan2(vector[1], vector[0]) - theta 
            if phi > np.pi:
                phi -= 2 * np.pi
            elif phi < -np.pi:
                phi += 2 * np.pi
            return dist, phi
        
        '''
            [d_Ti, phi_Ti, d_Ai, phi_Ai, v_A, phi_A, f_sep, phi_sep,
            f_ali, phi_ali, f_coh, phi_coh, a_boids, w_boids]
        '''
        robot_num = positions.shape[0]
        if boids_states is not None:
            assert boids_actions is not None, 'If using Boids states, you need Boids actions'
            feature_dim = 14
        else:
            feature_dim = 6
        observations = np.zeros((robot_num, feature_dim+2*robot_num))
        for i, pos in enumerate(positions):
            theta = phis[i]
            observations[i, 0:2] = _calculate_dist_phi(-pos, theta)
            dist, phi = _calculate_dist_phi(goal - pos, theta)
            observations[i, 2:4] = dist, phi
            self.def_att_dists[i] = dist

            observations[i, 4:6] = _calculate_dist_phi(goal_vel, theta)

            if boids_states is not None:
                observations[i, 6:8] = _calculate_dist_phi(boids_states[i, 0:2], theta)
                observations[i, 8:10] = _calculate_dist_phi(boids_states[i, 2:4], theta)
                observations[i, 10:12] = _calculate_dist_phi(boids_states[i, 4:6], theta)
                observations[i, 12:14] = thrust_to_action(boids_actions[i])
            
            k=0
            for j, teammate_pos in enumerate(positions):
                if j == i:
                    continue
                else:
                    dist, phi = _calculate_dist_phi(teammate_pos - pos, theta)
                    observations[i, feature_dim+2*k:feature_dim+2+2*k] = dist, phi
                    self.def_def_dists[i, k] = dist
                    k += 1
            
        return observations

    def check_is_terminated(self):
        '''
            Check if simulation is terminated

            Return:
                0: not terminated
                1: attacker reach
                2: defender collide
                3: defender capture
                4: time out
        '''
        att_tar_dist = np.linalg.norm(self.curr_pos[0])
        def_tar_dists = [np.linalg.norm(self.curr_pos[i]) for i in range(1, self.num_robots)]

        # Check if attacker reached target (using exact radius)
        if att_tar_dist < self.target_r:
            print("\n\nAttacker Reach\n\n")
            return 1    

        elif (self.def_def_dists < self.collision_r).any():
            print("\n\nDefender Collide\n\n")
            return 2
        
        elif (self.def_att_dists < self.defend_r).any():
            print("\n\nDefender Capture\n\n")
            return 3
        
        elif self.curr_time > self.total_time - 1e-5:
            print("\n\nTime Out\n\n")
            return 4
        else:
            return 0 

    def experiment_monitoring(self):
        time.sleep(10.5)
        while rclpy.ok():
            with self.lock:
                if self.check_is_terminated():
                    self.end_simulation()
            time.sleep(0.05)
    
    def end_simulation(self):

        print("\n\nShutdown simulation\n\n")
        _process = subprocess.Popen(['pkill', '-f', 'gz sim'])
        _process.communicate()
        rclpy.shutdown()

if __name__ == "__main__":
    rclpy.init()

    parser = argparse.ArgumentParser(description="Run VRX Experiment")
    parser.add_argument("--setting", type=int, default=1, choices=[0, 1], help="World setting: 0 for ocean, 1 for dock")
    parser.add_argument("--agility", type=float, default=2.25, help="Attacker agility")
    parser.add_argument("--num_robots", type=int, default=4, help="Total number of robots (1 attacker + defenders)")
    parser.add_argument("--save_traj", action="store_true", help="Whether to save trajectory")
    parser.add_argument("--save_file", type=str, default="results/exp_traj1.npz", help="Path to save trajectory file")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode")
    parser.add_argument("--controller", type=str, default="RL", choices=['RL', 'AdaRes', 'Res', 'Boids'], help="Controller type")
    parser.add_argument("--modelname", type=str, default="checkpoints/rl-iapf1.pth", help="Path to model file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    ''' Experiment Args'''
    if args.setting == 0:
        world_name = "sydney_regatta_original"
    elif args.setting == 1:
        world_name = "sydney_regatta_original1"

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.modelname), exist_ok=True)
    
    exp_manager = ExperimentManager(args.num_robots, args.save_traj, args.save_file, args.device)

    init_poses = exp_manager.generate_init_info(agility=args.agility, setting=args.setting)
    exp_manager.launch_simulation(init_poses, world_name, args.headless, args.controller, args.modelname)

