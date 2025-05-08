'''
@Author: Ricca
@Date: 2024-07-16
@Description: cross attention module
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CSMA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CSMA, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv3d(in_dim, hidden_dim, 1)
        self.conv3 = nn.Conv3d(in_dim, hidden_dim, 1)
        self.conv_out = nn.Conv3d(hidden_dim, out_dim, 1)
        self.maxpool3d = nn.MaxPool3d(2, 2 )
        # self.sub = 2
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, C, N, H, W = x.shape
        q = self.conv1(x)
        # q = self.maxpool3d(q)
        q = q.view(batch_size, self.hidden_dim, -1)
        k = self.conv2(x)
        # k = self.maxpool3d(k)
        k = k.view(batch_size, self.hidden_dim, -1).permute(0, 2, 1)
        feat_nl = torch.matmul(q, k)
        feat_nl = F.softmax(feat_nl, dim=-1)

        v = self.conv3(x)
        # v = self.maxpool3d(v)
        v = v.view(batch_size, self.hidden_dim, -1)
        feats = torch.matmul(feat_nl, v)
        feats = feats.view(batch_size, self.hidden_dim, -1, H, W)
        feats = self.conv_out(feats)
        feats = feats + x
        feats = torch.mean(feats, dim=2)
        return feats