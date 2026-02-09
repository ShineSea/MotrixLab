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

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import  Go2FlatEnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("Isaac-Velocity-Flat-Unitree-Go2-v0", "np")
class Go2FlatEnv (NpEnv):
    _cfg: Go2FlatEnvCfg
    
    def __init__(self, cfg: Go2FlatEnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：54维
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_range = 0.1  # 随机生成范围：±0.1m（0.2m×0.2m区域）
    
        # 导航统计计数器
        self.navigation_stats_step = 0
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        # base的四元数索引
        self._base_quat_dof_start = 6
        self._base_quat_dof_end = 10
        #关节角的索引
        self._joint_angle_dof_start=10
        self._joint_angle_dof_end=22

        self._init_dof_pos[self._joint_angle_dof_start:self._joint_angle_dof_end] = self.default_angles
        self.action_filter_alpha = 0.3
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
   
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        cfg = self._cfg
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        
        # 初始化接触检测矩阵
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """初始化终止接触检测"""
        cfg = self._cfg
        # 查找基座几何体
        base_indices = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            try:
                base_idx = self._model.get_geom_index(base_name)
                if base_idx is not None:
                    base_indices.append(base_idx)
                else:
                    print(f"Warning: Geom '{base_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding base geom '{base_name}': {e}")

        # 创建基座-地面接触检测矩阵
        if base_indices:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            # 使用空数组
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("Warning: No base contacts configured for termination")

    def _init_foot_contact(self):
        """初始化足部接触检测"""
        cfg = self._cfg
        foot_indices = []
        for foot_name in cfg.asset.foot_names:
            try:
                foot_idx = self._model.get_geom_index(foot_name)
                if foot_idx is not None:
                    foot_indices.append(foot_idx)
                else:
                    print(f"Warning: Foot geom '{foot_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding foot geom '{foot_name}': {e}")
        
        # 创建足部-地面接触检测矩阵
        if foot_indices:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices],
                dtype=np.uint32
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0
            print("Warning: No foot contacts configured")
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self.get_local_linvel(data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions, state):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_target_jq(actions)
        return state

    def _compute_target_jq(self, actions):
        # Compute target position from actions.
        target_jq = actions * self.cfg.control_config.action_scale + self.default_angles
        return target_jq

    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data

        obs=self._compute_obs(data,state.info)
       
        # 更新目标标记和箭头
        self._update_target_marker(data, state.info["pose_commands"])
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        base_lin_vel_xy = root_vel[:, :2]
        reached_all,desired_vel_xy,velocity_commands=self._compute_velocity_commands(data,state.info)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 计算奖励
        reward = self._compute_reward(data, state.info)
        
        # 计算终止条件
        state = self._compute_terminated(state)
        return state.replace(
            obs=obs,
            reward=reward,
        )
    
    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data

        terminated = np.zeros(self._num_envs, dtype=bool)
        terminated |= self._check_dof_velocity_failure(data)
        terminated |= self._check_base_contact_failure(data)
        terminated |= self._check_side_flip_failure(data)

        self._debug_termination(
            state,
            terminated=terminated
        )

        return state.replace(
            terminated=terminated  
        )
    
    def _compute_reward(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """
        导航任务奖励计算
        """
        reached_all,desired_vel_xy,velocity_commands=self._compute_velocity_commands(data,info)
        mask = reached_all.astype(np.float32)
        inv = 1.0 - mask
        reward_dict = {
            # always-on
            "lin_vel_z":        self._reward_lin_vel_z(data),
            "ang_vel_xy":       self._reward_ang_vel_xy(data),
            "orientation":      self._reward_orientation(data),
            "torques":          self._reward_torques(data),
            "dof_vel":          self._reward_dof_vel(data),
            "dof_acc":          self._reward_dof_acc(data, info),
            "action_rate":      self._reward_action_rate(info),
            "termination":      self._reward_termination(data),
            "stand_still":      self._reward_stand_still(data, velocity_commands),
           
            # 未到达
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, velocity_commands) * inv,
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, velocity_commands) * inv,  
            "approach_reward":  self._reward_approch(data,info) * inv,
            

            # 已到达
            "arrival_bonus":    self._reward_arrival_bonus(info, reached_all) * mask,
            "stop_bonus":       self._reward_stop_bonus(data, reached_all) * mask,
        }


        rewards = {k: v * self.cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        # rwd = np.clip(rwd, 0.0, 10000.0)
        return rwd

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]
        
        # 先生成机器人的初始位置（在世界坐标系中）
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0], pos_range[2],  # x_min, x_max
            num_envs
        )
        robot_init_y = np.random.uniform(
            pos_range[1], pos_range[3],  # y_min, y_max
            num_envs
        )
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # 生成目标位置：相对于机器人初始位置的偏移
        target_offset = np.random.uniform(
            low = cfg.commands.pose_command_range[:2],
            high = cfg.commands.pose_command_range[3:5],
            size = (num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset  # 世界坐标系中的目标位置

        # 生成目标朝向（绝对朝向，水平方向随机）
        target_headings = np.random.uniform(
            low = cfg.commands.pose_command_range[2],
            high = cfg.commands.pose_command_range[5],
            size = (num_envs, 1)
        )

        pose_commands = np.concatenate([target_positions, target_headings],axis = 1)

        # 设置初始状态 - 避免给四元数添加噪声
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # 创建噪声 - 不要给四元数添加噪声
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)
        
        # target_marker (DOF 0-2): 不添加噪声，会在_update_target_marker中设置
        
        # base的位置 (DOF 3-5): 使用前面生成的随机初始位置
        noise_pos[:, 3] = robot_init_x - cfg.init_state.pos[0]  # 相对默认位置的偏移
        noise_pos[:, 4] = robot_init_y - cfg.init_state.pos[1]
        # Z轴不添加噪声，保持固定高度避免坠落感
        # base的四元数 (DOF 6-9): 不添加噪声，保持为单位四元数
        
        # 关节角度(DOF 10:)不添加噪声，保证初始站立稳定
        # noise_pos[:, 10:] = 0.0  # 已经初始化为0

        # 所有速度都设为0，确保完全静止
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel + noise_vel
        
        # 归一化base的四元数（DOF 6-9）
        # 新的DOF结构：target_marker占0-2, base_pos占3-5, base_quat占6-9
        for env_idx in range(num_envs):
            quat = dof_pos[env_idx, self._base_quat_dof_start:self._base_quat_dof_end]  # [qx, qy, qz, qw]
            quat_norm = np.linalg.norm(quat) # 返回 sqrt(q0^2 + q1^2 + q2^2 + q3^2)
            if quat_norm > 1e-6:  # 避免除以零
                dof_pos[env_idx, self._base_quat_dof_start:self._base_quat_dof_end] = quat / quat_norm
            else:
                dof_pos[env_idx, self._base_quat_dof_start:self._base_quat_dof_end] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 默认单位四元数
            
            # 归一化箭头的四元数（如果箭头body存在）
            if self._robot_arrow_body is not None:
                # robot_heading_arrow的四元数（DOF 25-28: qx, qy, qz, qw）
                robot_arrow_quat = dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end]
                quat_norm = np.linalg.norm(robot_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = robot_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                
                # desired_heading_arrow的四元数（DOF 32-35: qx, qy, qz, qw）
                desired_arrow_quat = dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end]
                quat_norm = np.linalg.norm(desired_arrow_quat)
                if quat_norm > 1e-6:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = desired_arrow_quat / quat_norm
                else:
                    dof_pos[env_idx, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": np.linalg.norm(target_offset, axis=1),  
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 上一步关节速度
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),  # 足部接触状态
        }
        
        obs=self._compute_obs(data,info)
        return obs, info
    

    def _compute_obs(self,data:mtx.SceneData, info: dict) -> np.ndarray:
        cfg=self._cfg
         # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self.get_gyro(data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 导航目标
        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定（只看位置，与奖励计算保持一致）
        position_threshold = 0.1
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令（与平地navigation一致，简单P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向（从当前位置指向目标）
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = info["current_actions"]
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)  # 54 + 1 = 55维
        return obs

    def _compute_velocity_commands(self,data:mtx.SceneData, info: dict):
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        # 导航目标
        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定（只看位置，与奖励计算保持一致）
        position_threshold = 0.1
        reached_all = distance_to_target < position_threshold  # 楼梯任务：只要到达位置即可
        
        # 计算期望速度命令（与平地navigation一致，简单P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向（从当前位置指向目标）
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        return reached_all,desired_vel_xy,velocity_commands
    
    def _compute_distance_to_target(self,data,info):
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        robot_position = root_pos[:, :2]
        target_position = info["pose_commands"][:, :2]
        return np.linalg.norm(target_position - robot_position, axis=1)
        
         # ------------ reward functions----------------
    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
    
    def _reward_lin_vel_z(self, data):
        # Penalize z axis base linear velocity
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1)

    def _reward_orientation(self, data):
        # 将重力向量从世界坐标系变换到基座局部坐标系
        # 将x,y分量的平方
        # Penalize non flat base orientation
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = Quaternion.rotate_inverse(base_quat, np.array([0, 0, -1], dtype=np.float32))
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, data: mtx.SceneData):
        # Penalize torques
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, data):
        terminated = np.zeros(self._num_envs, dtype=bool)
        terminated |= self._check_dof_velocity_failure(data)
        terminated |= self._check_base_contact_failure(data)
        terminated |= self._check_side_flip_failure(data)
        return terminated

    # def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
    #     # Reward long steps
    #     feet_air_time = info["feet_air_time"]
    #     first_contact = (feet_air_time > 0.0) * info["contacts"]
    #     # reward only on first contact with the ground
    #     rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
    #     # no reward for zero command
    #     rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
    #     return rew_airTime

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, data, commands: np.ndarray):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )

    def _reward_arrival_bonus(self,info,reached_all):
    # 首次到达位置的一次性奖励
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        return first_time_reach
    

    def _reward_approch(self,data,info):
        # 距离接近奖励：激励靠近目标
        # 使用历史最近距离来计算进步
        distance_to_target=self._compute_distance_to_target(data,info)
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  
        return approach_reward
    
    def _reward_stop_bonus(self,data,reached_all):
        base_lin_vel=self.get_local_linvel(data)
        gyro=self.get_gyro(data)
        # 到达与停止判定（奖励加成）
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # 放宽到0.05 rad/s ≈ 2.86°/s
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        stop_base = 2 * (0.8 * np.exp(- (speed_xy / 0.2)**2) + 1.2 * np.exp(- (np.abs(gyro[:, 2]) / 0.1)**4))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)
        return stop_bonus

    # def _reward_hip_pos(self, data, commands: np.ndarray):
    #     return (0.8 - np.abs(commands[:, 1])) * np.sum(
    #         np.square(self.get_dof_pos(data)[:, self.hip_indices] - self.default_angles[self.hip_indices]),
    #         axis=1,
    #     )

    # def _reward_calf_pos(self, data, commands: np.ndarray):
    #     return (0.8 - np.abs(commands[:, 1])) * np.sum(
    #         np.square(self.get_dof_pos(data)[:, self.calf_indices] - self.default_angles[self.calf_indices]),
    #         axis=1,
    #     )

    def _check_timeout(self, state: NpEnvState) -> np.ndarray:
        if not self._cfg.max_episode_steps:
            return np.zeros(self._num_envs, dtype=bool)
        return state.info["steps"] >= self._cfg.max_episode_steps

     # 检查DOF速度是否超限（防止inf/数值发散） 
    def _check_dof_velocity_failure(self, data) -> np.ndarray:
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)

        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (
            np.isnan(dof_vel).any(axis=1)
            | np.isinf(dof_vel).any(axis=1)
            | (vel_max > 1e6)
        )
        return vel_overflow | vel_extreme
    
    # 机器人基座接触地面终止
    def _check_base_contact_failure(self, data) -> np.ndarray:
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape(
            (self._num_envs, self.num_termination_check)
        )
        return termination_check.any(axis=1)
    
    # 侧翻终止：倾斜角度超过75°
    def _check_side_flip_failure(self, data) -> np.ndarray:
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]

        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]

        tilt_angle = np.arctan2(gxy, np.abs(gz))
        return tilt_angle > np.deg2rad(75)

    def _debug_termination(self, state, terminated):
        if not (terminated.any()):
            return
        if state.info["steps"][0] % 100 != 0:
            return
        print(
            f"[termination] "
            f"terminated={int(terminated.sum())} "
        )