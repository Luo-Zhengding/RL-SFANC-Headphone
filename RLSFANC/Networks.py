"""
Lightweight 1D CNN used as the Q-network for control-filter selection.

The network processes a raw time-domain noise waveform and outputs Q-values
over K pre-trained control filters. A large kernel in the first layer captures
global structure; residual blocks with smaller kernels extract local patterns.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, prev_channel, channel, conv_kernel, conv_stride, conv_pad):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv1d(
                in_channels=prev_channel,
                out_channels=channel,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_pad,
            ),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_pad,
            ),
            nn.BatchNorm1d(channel),
        )
        self.bn = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.res(x)
        if x.shape[1] == identity.shape[1]:
            x += identity
        elif x.shape[1] > identity.shape[1]:
            if x.shape[1] % identity.shape[1] == 0:
                x += identity.repeat(1, x.shape[1] // identity.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock need to be divisible by the previous dims.")
        else:
            if identity.shape[1] % x.shape[1] == 0:
                identity += x.repeat(1, identity.shape[1] // x.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock need to be divisible by the previous dims.")
            x = identity
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNRes(nn.Module):
    """1D residual CNN Q-network. Input: (B, 1, L), output: Q-values of shape (B, K)."""

    def __init__(
        self,
        channels=[[128], [128] * 2],
        conv_kernels=[80, 3],
        conv_strides=[4, 1],
        conv_padding=[38, 1],
        pool_padding=[0, 0],
        num_classes=7,
    ):
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNNRes, self).__init__()

        prev_channel = 1
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=prev_channel,
                out_channels=channels[0][0],
                kernel_size=conv_kernels[0],
                stride=conv_strides[0],
                padding=conv_padding[0],
            ),
            nn.BatchNorm1d(channels[0][0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[0]),
        )

        prev_channel = channels[0][0]
        self.res_blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            block = []
            for conv_channel in channels[i]:
                block.append(ResBlock(prev_channel, conv_channel, conv_kernels[i], conv_strides[i], conv_padding[i]))
                prev_channel = conv_channel
            self.res_blocks.append(nn.Sequential(*block))

        self.pool_blocks = nn.ModuleList()
        for i in range(1, len(pool_padding)):
            self.pool_blocks.append(nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i]))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.prev_channel = prev_channel
        self.linear = nn.Sequential(nn.Linear(prev_channel, num_classes))

    def forward(self, inwav):
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            inwav = self.res_blocks[i](inwav)
            if i < len(self.pool_blocks):
                inwav = self.pool_blocks[i](inwav)
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out
