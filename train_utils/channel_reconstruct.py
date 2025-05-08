'''
@Author: Ricca
@Date: 2024-07-16
@Description: 通道重构模块
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRU(nn.Module):
    """
    通道重构单元，实现通道上的压缩、变换和融合。

    参数:
        op_channel (int): 输出通道数。
        alpha (float): 上分支比例，默认为1/2。
        squeeze_radio (int): 压缩比例，默认为2。
        group_size (int): 分组卷积中的组大小，默认为2。
        group_kernel_size (int): 分组卷积的核大小，默认为3。
    """
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # 上分支
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # 下分支
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # 分割
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # 变换
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # 融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2
