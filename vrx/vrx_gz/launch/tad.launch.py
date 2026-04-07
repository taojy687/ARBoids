# Copyright 2021 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os
import vrx_gz.launch
from vrx_gz.model import Model

def launch(context, *args, **kwargs):

    world_name = LaunchConfiguration('world').perform(context)
    sim_mode = LaunchConfiguration('sim_mode').perform(context)
    bridge_competition_topics = LaunchConfiguration(
        'bridge_competition_topics').perform(context).lower() == 'true'
    robot = LaunchConfiguration('robot').perform(context)
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    gz_paused = LaunchConfiguration('paused').perform(context).lower() == 'true'
    competition_mode = LaunchConfiguration('competition_mode').perform(context).lower() == 'true'
    extra_gz_args = LaunchConfiguration('extra_gz_args').perform(context)
    init_poses_str = LaunchConfiguration('init_poses').perform(context)
    goal_str = LaunchConfiguration('goal').perform(context)

    init_poses = [[float(p.split(',')[0]), float(p.split(',')[1]), 0.0, 0.0, 0.0, 
                   float(p.split(',')[2])] for p in init_poses_str.split(';')] if init_poses_str else []
    goal = [0.0, 0.0]

    launch_processes = []
    models = []

    for i, pose in enumerate(init_poses):
        name = f'wamv{i+1}'
        model = Model(name, 'wam-v', pose)
        models.append(model)

    world_name, ext = os.path.splitext(world_name)

    launch_processes.extend(vrx_gz.launch.simulation(world_name, headless, gz_paused, extra_gz_args))

    world_name_base = os.path.basename(world_name)
    launch_processes.extend(vrx_gz.launch.spawn(sim_mode, world_name_base, models, robot))

    if (sim_mode == 'bridge' or sim_mode == 'full') and bridge_competition_topics:
        launch_processes.extend(vrx_gz.launch.competition_bridges(world_name_base, competition_mode))

    return launch_processes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', 
            default_value='sydney_regatta',
            description='Name of world'
        ),

        DeclareLaunchArgument(
            'sim_mode',
            default_value='full',
            description='Simulation mode: "full", "sim", "bridge".'
                        'full: spawns robot and launch ros_gz bridges, '
                        'sim: spawns robot only, '
                        'bridge: launch ros_gz bridges only.'
        ),

        DeclareLaunchArgument(
            'bridge_competition_topics',
            default_value='True',
            description='True to bridge competition topics, False to disable bridge.'
        ),

        DeclareLaunchArgument(
            'robot',
            default_value='',
            description='Name of robot to spawn if specified.'
        ),

        DeclareLaunchArgument(
            'headless',
            default_value='False',
            description='True to run simulation headless (no GUI).'
        ),

        DeclareLaunchArgument(
            'paused',
            default_value='False',
            description='True to start the simulation paused.'
        ),

        DeclareLaunchArgument(
            'competition_mode',
            default_value='False',
            description='True to disable debug topics. '
        ),

        DeclareLaunchArgument(
            'extra_gz_args',
            default_value='',
            description='Additional arguments to be passed to gz sim.'
        ),

        DeclareLaunchArgument(
            'init_poses',
            default_value='',
            description='Initial poses of robots: "x_1, y_1, theta_1;..."'
        ),

        DeclareLaunchArgument(
            'goal',
            default_value='',
            description='Goals of attacker: "x_1, y_1"'
        ),

        OpaqueFunction(function=launch),
    ])

