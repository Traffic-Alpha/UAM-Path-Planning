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
        passen_shape = observation_space["passen_attr"].shape
        mask_shape = observation_space["passen_mask"].shape
        sinr_shape = observation_space["sinr_attr"].shape
        uncertainty_shape = observation_space["uncertainty_attr"].shape

        self.hidden_dim = 32
        self.linear_encoder_ac = nn.Sequential(
            nn.Linear(ac_shape[-1], self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_passen = nn.Sequential(
            nn.Linear(passen_shape[0]*passen_shape[-1], self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_mask = nn.Sequential(
            nn.Linear(mask_shape[-1], self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_sinr = nn.Sequential(
            nn.Linear(sinr_shape[0]*sinr_shape[-1], self.hidden_dim),
            nn.ReLU(),
        )
        # uncertainty_outdim = 1500
        self.linear_encoder_uncertainty = nn.Sequential(
            nn.Linear(uncertainty_shape[0] * uncertainty_shape[-1], self.hidden_dim),
            nn.ReLU(),
        )

        input_dim = int(
            self.hidden_dim * 5
        )

        self.output = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim)
        )

    def forward(self, observations):
        ac_attr, passen_attr, passen_mask, sinr_attr, uncertainty_attr = (observations["ac_attr"],
                                                                        observations["passen_attr"],
                                                                        observations["passen_mask"],
                                                                        observations["sinr_attr"],
                                                                        observations["uncertainty_attr"])
        batch_size = ac_attr.size(0)
        # Passenger attribute encoder
        # passen_output = self.passen_encoder(passen_attr, passen_mask) # (1,4,32)

        # Aircraft attribute encoder
        # h0 = torch.zeros(1, batch_size, self.hidden_dim).to(ac_attr.device)
        # c0 = torch.zeros(1, batch_size, self.hidden_dim).to(ac_attr.device)
        # ac_output, _ = self.ac_encoder(ac_attr, (h0, c0))  # input : B N H, output : B N H (1,1,32)

        # sinr and uncertainty encoder -> MAP feature
        # sinr_output = self.sinr_encoder(sinr_attr.unsqueeze(1)) # B C H W (1,16,4,4)
        # _, _, sinr_H, sinr_W = sinr_output.shape

        # uncertainty_output = self.uncertainty_encoder(uncertainty_attr.unsqueeze(1)) # B C H W (1,128,6,6)
        # _, _, uncertainty_W, uncertainty_H = uncertainty_output.shape

        passen_output = self.linear_encoder_passen(passen_attr.reshape(batch_size, -1))
        ac_output = self.linear_encoder_ac(ac_attr.reshape(batch_size, -1))
        sinr_output = self.linear_encoder_sinr(sinr_attr.reshape(batch_size, -1))
        uncertainty_output = self.linear_encoder_uncertainty(uncertainty_attr.reshape(batch_size, -1))
        mask_output = self.linear_encoder_mask(passen_mask.reshape(batch_size, -1))

        # feature fusion
        # all_feature_output = torch.cat((ac_output.reshape(batch_size,-1),
        #                         passen_output.reshape(batch_size,-1),
        #                         mask_output.reshape(batch_size, -1),
        #                         sinr_output.reshape(batch_size, -1),
        #                         uncertainty_output.reshape(batch_size, -1)), dim=1) # B N H (1,5,32)

        all_feature_output = torch.cat((ac_output,
                                passen_output,
                                mask_output,
                                sinr_output,
                                uncertainty_output), dim=1)

        all_feature_output = self.output(all_feature_output) # B N H (1,176,32)

        return all_feature_output