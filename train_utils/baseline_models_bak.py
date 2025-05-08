'''
@Author: Ricca
@Date: 2024-07-16
@Description: Custom Model
@LastEditTime:
'''
import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        ac_shape = observation_space["ac_attr"].shape
        passen_shape= observation_space["passen_attr"].shape
        mask_shape = observation_space["passen_mask"].shape
        sinr_shape = observation_space["sinr_attr"].shape
        uncertainty_shape = observation_space["uncertainty_attr"].shape

        self.ac_encoder = nn.Sequential(
            nn.Linear(ac_shape[-1], 16),
            nn.ReLU(),
        )

        self.user_linear = nn.Sequential(
            nn.Linear(passen_shape[-1], 16),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            nn.Linear(sinr_shape[-1], 16),
            nn.ReLU(),
        )

        uncertainty_outdim = 1500
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(uncertainty_shape[0]*uncertainty_shape[-1], uncertainty_outdim),
            nn.ReLU(),
        )

        input_dim = int(
            ac_shape[-1] + passen_shape[0]*passen_shape[1] + sinr_shape[0]*sinr_shape[-1] + mask_shape[-1] + uncertainty_outdim
        )

        self.relu = nn.ReLU()

        self.output = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, features_dim)
        )

    def forward(self, observations):
        # obsevations[]
        ac_attr, passen_attr, passen_mask, sinr_attr,uncertainty_attr = (observations["ac_attr"],
                                                                        observations["passen_attr"],
                                                                        observations["passen_mask"],
                                                                        observations["sinr_attr"],
                                                                        observations["uncertainty_attr"])
        batch_size = ac_attr.size(0)
        uncertainty_attr = self.uncertainty_encoder(uncertainty_attr.reshape(batch_size, -1))
        _input = torch.cat(
            (
                ac_attr.reshape(batch_size, -1),
                passen_attr.reshape(batch_size, -1),
                passen_mask.reshape(batch_size, -1),
                sinr_attr.reshape(batch_size, -1),
                uncertainty_attr.reshape(batch_size, -1),
            ), dim=1)
        output = self.output(_input).reshape(_input.size(0),-1)
        return output
# 不设置uncertainty_attr norm则收敛
