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

class Attention(nn.Module):
    def __init__(self, input_dim, head_num):
        super(Attention, self).__init__()
        self.hidden_size = input_dim//2
        self.num_heads = head_num
        self.q1 = nn.Linear(input_dim, input_dim//2)
        self.kv1 = nn.Linear(input_dim, input_dim)

        self.q2 = nn.Linear(input_dim, input_dim//2)
        self.kv2 = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x, h_window_size, w_window_size):
        """分割特征图到小窗口中。"""
        B, num_heads, N, C = x.shape
        H = W = int(N**0.5)
        x = x.reshape(B * num_heads, N, C).reshape(B * num_heads, H, W, C)
        x = x.reshape(B * num_heads, H // h_window_size, h_window_size, W // w_window_size, w_window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, h_window_size, w_window_size, C)
        return windows

    def window_reverse(self,windows, h_window_size, w_window_size, N, head):
        """重建小窗口到原始特征图尺寸。"""
        H = W = int(N ** 0.5)
        Bhead = int(windows.shape[0] / (H * W / h_window_size / w_window_size))
        x = windows.reshape(Bhead, H // h_window_size, W // w_window_size, h_window_size, w_window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(Bhead, H, W, -1)
        x = x.reshape(Bhead // head, head, H, W, -1).permute(0, 2, 3, 1, 4).reshape(Bhead // head, H * W, -1)
        return x
    def forward(self, x):
        B, N, C = x.shape
        # globle attention
        q1 = self.q1(x).reshape(B, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
        kv1 = self.kv1(x).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]

        # Calculate attention scores
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights1 = self.softmax(scores1)

        # Apply attention weights to value to get context vector
        context_vector1 = torch.matmul(attention_weights1, v1)
        context_vector1 = context_vector1.permute(0, 2, 1, 3).reshape(B, N, -1)

        # local attention
        q2 = self.q2(x).reshape(B, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
        kv2 = self.kv2(x).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]

        # 为局部注意力在窗口内分割输入
        q2, k2, v2 = self.window_partition(q2, 1, 1), self.window_partition(k2, 1, 1), \
            self.window_partition(v2, 1, 1)  # (18,1,1,16)

        # Calculate attention scores
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights2 = self.softmax(scores2)

        # Apply attention weights to value to get context vector
        context_vector2 = torch.matmul(attention_weights2, v2)

        context_vector2 = self.window_reverse(context_vector2, 1, 1, N, self.num_heads//2)  # (18,1,1,32)
        context_vector = torch.cat((context_vector1, context_vector2), dim=-1)
        context_vector += x
        return context_vector

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

        self.ac_encoder = (
            nn.LSTM(ac_shape[-1]//3, self.hidden_dim, 1, batch_first=True)
        )
        self.passen_encoder = UserTranformer(input_dim=passen_shape[-1], d_model=self.hidden_dim, nhead=1, dim_feedforward=64,
                                                  num_encoder_layers=1)
        self.sinr_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 1, padding=0),
        )

        self.uncertainty_encoder = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.attention = Attention(self.hidden_dim*2, head_num=4)

        self.output = nn.Sequential(
            nn.Linear(576, 2048),
            nn.ReLU(),
            nn.Linear(2048, features_dim)
        )

        self.output2 = nn.Sequential(
            nn.Linear(576, features_dim),
        )

    def forward(self, observations):
        ac_attr, passen_attr, passen_mask, sinr_attr, uncertainty_attr = (observations["ac_attr"],
                                                                        observations["passen_attr"],
                                                                        observations["passen_mask"],
                                                                        observations["sinr_attr"],
                                                                        observations["uncertainty_attr"])
        batch_size = ac_attr.size(0)
        # Passenger attribute encoder
        passen_output = self.passen_encoder(passen_attr, passen_mask) # (1,4,32)

        # Aircraft attribute encoder
        h0 = torch.zeros(1, batch_size,  self.hidden_dim).to(ac_attr.device)
        c0 = torch.zeros(1, batch_size,  self.hidden_dim).to(ac_attr.device)
        ac_output, _ = self.ac_encoder(ac_attr.reshape(batch_size, 3, -1), (h0, c0))  # input : B N H, output : B N H (1,3,32)

        # sinr and uncertainty encoder -> MAP feature
        sinr_output = self.sinr_encoder(sinr_attr.unsqueeze(1)) # B C H W (1,32,3,3)
        _, _, sinr_H, sinr_W = sinr_output.shape

        uncertainty_output = self.uncertainty_encoder(uncertainty_attr.unsqueeze(1)) # B C H W (1,32,3,3)
        _, _, uncertainty_W, uncertainty_H = uncertainty_output.shape

        seq_output = torch.cat((ac_output, passen_output), dim=1)  # B N H (1,7,32)
        _, N_seq, H_seq = seq_output.shape

        map_feature1 = sinr_output # B 128 H W
        map_feature2 = uncertainty_output # B 512 H W (1,32,3,3)

        all_feature1_list = []
        all_feature2_list = []
        for n in range(N_seq):
            seq_feature = seq_output[:, n, :]  # B C (1,32)
            seq_feature = seq_feature.unsqueeze(1).unsqueeze(1)  # B W H C (1,1,1,32)
            seq_feature = seq_feature.expand(-1, uncertainty_W, uncertainty_H, -1)  # B W H C (1,3,3,32)

            all_feature1 = torch.cat((seq_feature.permute(0, 3, 1, 2), map_feature1), dim=1)  # B W H C
            all_feature2 = torch.cat((seq_feature.permute(0, 3, 1, 2), map_feature2), dim=1)  # B W H C (1,64,3,3)

            all_feature1_list.append(all_feature1)
            all_feature2_list.append(all_feature2)

        all_feature1 = torch.stack(all_feature1_list, dim=1)  # B C N H W
        all_feature2 = torch.stack(all_feature2_list, dim=1)  # B C N H W (1,7,64,3,3)

        # feature fusion
        all_feature_output = torch.cat((all_feature1, all_feature2), dim=1)  # B N C H W

        all_feature_output = all_feature_output.mean(dim=1).reshape(batch_size, -1, sinr_W*sinr_H).permute(0,2,1)  # (B 3*3 C)

        all_feature_output = self.attention(all_feature_output)

        all_feature_output = self.output2(all_feature_output.reshape(batch_size, -1)) # B N H (1,176,32)

        return all_feature_output

