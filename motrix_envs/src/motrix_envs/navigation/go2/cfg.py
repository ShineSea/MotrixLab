# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
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
# ==============================================================================

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    action_scale = 0.2  # action scale


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0]  # Z-axis height matches the initial height of base in XML

     # position randomization range [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # randomly distributed over 20m x 20m range on ground

   # the default angles for all joints. key = joint name, value = target angle [rad]
    default_joint_angles = {
        "FL_hip"    :  0.1,  # [rad]
        "FL_thigh"  :  0.9,  # [rad]
        "FL_calf"   : -1.8,  # [rad]
        "FR_hip"    : -0.1,  # [rad]
        "FR_thigh"  :  0.9,  # [rad]
        "FR_calf"   : -1.8,  # [rad]
        "RL_hip"    :  0.1,  # [rad]
        "RL_thigh"  :  0.9,  # [rad]
        "RL_calf"   : -1.8,  # [rad]
        "RR_hip"    : -0.1,  # [rad]
        "RR_thigh"  :  0.9,  # [rad]
        "RR_calf"   : -1.8,  # [rad]
    }



@dataclass
class Commands:
    # offset range of target position relative to robot initial position
    # [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: offset relative to robot initial position (meters)
    # yaw: target absolute orientation (radians), random horizontal direction
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]


@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


@dataclass
class Asset:
    body_name = "base"
    foot_names = ["foot"]
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = [
        "base_collision_0", "base_collision_1", "base_collision_2",
        "fl_hip_0", "fr_hip_0", "rl_hip_0", "rr_hip_0",
    ]
    ground_name = "ground"


@dataclass
class Sensor:
    base_linvel = "local_linvel"
    base_gyro = "gyro"


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
             # ===== 导航任务核心奖励 =====
            "tracking_lin_vel": 1.5,      
            "tracking_ang_vel": 0.75,  
            "approach_reward": 1.0,       
            
            # ===== Locomotion稳定性奖励（保持但降低权重） =====
            "orientation": -2.5,              # 姿态稳定（降低权重）
            "lin_vel_z": -2.0,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques":   -0.0002,               # 扭矩惩罚
            "dof_vel": -5e-5,                  # 关节速度惩罚
            "dof_acc": -2.5e-7,                  # 关节加速度惩罚
            "action_rate": -0.01,          # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -20.0,          # 终止惩罚

            # ===== 到达奖励  ====
            "arrival_bonus":10.0,
            "stop_bonus":1.0
        }
    )

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1


@registry.envcfg("Isaac-Velocity-Flat-Unitree-Go2-v0")
@dataclass
class Go2FlatEnvCfg(EnvCfg):
    render_spacing: float = 0.0
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 20
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # maximum joint velocity threshold, greater tolerance during early training

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
