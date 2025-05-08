'''
@Author: Ricca
@Date: 2024-07-02 21:20:00
@LastEditTime: 2024-07-02 21:20:00
@LastEditors: Ricca
'''
import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from train_utils.my_transformer import UserTranformer
from train_utils.CSMA import CSMA
from train_utils.channel_reconstruct import CRU
class FusionModel(BaseFeaturesExtractor):
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
        uncertainty_outdim = 1500
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(uncertainty_shape[0] * uncertainty_shape[-1], uncertainty_outdim),
            nn.ReLU(),
        )

        input_dim = int(
            self.hidden_dim * 4 + uncertainty_outdim
        )

        # self.ac_encoder = (
        #     nn.LSTM(ac_shape[-1], self.hidden_dim, 1, batch_first=True)
        # )
        # self.passen_encoder = UserTranformer(input_dim=passen_shape[-1], d_model=self.hidden_dim, nhead=1, dim_feedforward=64,
        #                                           num_encoder_layers=1)
        # self.sinr_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2, 1, padding=0),
        # )

        # self.uncertainty_encoder = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, padding=0),
        #     # nn.Conv2d(32, 128, 3, padding=0),
        #     # nn.ReLU(),
        #     # nn.MaxPool2d(2, 2, padding=0),
        # )
        #
        # self.csma1 = CSMA(in_dim=32, hidden_dim=1000, out_dim=32)
        # self.csma2 = CSMA(in_dim=144, hidden_dim=1000, out_dim=144)
        #
        # self.channel_reconstruct = CRU(op_channel=176)
        # input_dim = int(
        #     ac_shape[-1] + passen_shape[0] * passen_shape[1] + sinr_shape[0] * sinr_shape[-1] + mask_shape[-1]
        #     + uncertainty_outdim
        # )
        self.output = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, features_dim)
        )
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, observations):
        # obsevations[]
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

        # passen_output = passen_attr
        # ac_output = ac_attr
        # sinr_output = sinr_attr
        # uncertainty_output = self.linear_encoder_uncertainty(uncertainty_attr.reshape(batch_size, -1))
        # mask_output = passen_mask


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

        all_feature_output = self.project(all_feature_output) # B N H (1,176,32)

        return all_feature_output

