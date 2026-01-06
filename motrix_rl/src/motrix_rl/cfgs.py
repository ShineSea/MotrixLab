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

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class basic:
    @rlcfg("cartpole")
    @dataclass
    class CartPolePPO(PPOCfg):
        max_env_steps: int = 10_000_000
        check_point_interval: int = 500

        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 4
    
    @rlcfg("dm-walker", backend="jax")
    @rlcfg("dm-stander", backend="jax")
    @rlcfg("dm-runner", backend="jax")
    @dataclass
    class WalkerPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 4

    @rlcfg("dm-stander", backend="torch")
    @rlcfg("dm-walker", backend="torch")
    @dataclass
    class WalkerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32

    @rlcfg("dm-runner", backend="torch")
    @dataclass
    class RunnerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        # Override PPO configuration
        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 2
        mini_batches: int = 32

class manipulation:
    @rlcfg("franka_lift_cube")
    @dataclass
    class FrankaLiftPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 4096 * 50000
        check_point_interval: int = 500
        share_policy_value_features: bool = True
        
        # Override PPO configuration
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 24
        learning_epochs: int = 8
        mini_batches: int = 4
        learning_rate: float = 3e-4
        learning_rate_scheduler_kl_threshold: float = 0.01
        entropy_loss_scale: float = 0.001
        rewards_shaper_scale: float = 0.01
    @rlcfg("franka_open_cabinet")
    @dataclass
    class FrankaOpenCabinetPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 4096 * 24000
        check_point_interval: int = 500
        share_policy_value_features: bool = True
        
        # Override PPO configuration
        # learning_rate: float = 1e-1
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        rollouts: int = 16
        learning_epochs: int = 2
        mini_batches: int = 8
        learning_rate: float = 3e-4

class locomotion:
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkPPO(PPOCfg):
        """
        Go1 Walk RL config
        """

        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60000
        num_envs: int = 2048

        # Override PPO configuration
        rollouts: int = 24
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        learning_epochs: int = 5
        mini_batches: int = 3
        learning_rate: float = 3e-4

class navigation:
    @rlcfg("anymal_c_navigation_flat")
    @dataclass
    class AnymalCPPOConfig(PPOCfg):

        # ===== 基础训练参数 =====
        seed: int = 42         # 随机种子
        num_envs: int = 2048               # 训练时并行环境数量
        play_num_envs: int = 16            # 评估时并行环境数量
        max_env_steps: int = 100_000_000   # 最大训练步数
        check_point_interval: int = 100    # 检查点保存间隔（每100次迭代保存一次）

        # ===== PPO算法核心参数 =====
        learning_rate: float = 3e-4        # 学习率
        rollouts: int = 48                 # 经验回放轮数（增大每代训练时长）
        learning_epochs: int = 6           # 每次更新的训练轮数（增大每代训练时长）
        mini_batches: int = 32             # 小批量数量
        discount_factor: float = 0.99      # 折扣因子
        lambda_param: float = 0.95         # GAE参数
        grad_norm_clip: float = 1.0        # 梯度裁剪

        # ===== PPO裁剪参数 =====
        ratio_clip: float = 0.2            # PPO裁剪比率
        value_clip: float = 0.2            # 价值裁剪
        clip_predicted_values: bool = True # 裁剪预测值

        # 中型网络（默认配置，适合大部分任务）
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)