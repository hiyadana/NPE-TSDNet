# -*- coding: utf-8 -*-
"""
PyTorch 버전의 다중 스케일 특징 추출 유닛 모듈

TensorFlow로 작성된 MS_conv 클래스를 PyTorch의 nn.Module을 사용하여 재구현한다.
세 개의 병렬 컨볼루션 경로와 최종 결합 레이어로 구성된 아키텍처를 동일하게 유지한다.
"""

import torch
import torch.nn as nn
from pytorch_version.conv_layer import ConvLayer

class MS_conv(nn.Module):
    """
    다중 스케일 컨볼루션(Multi-Scale Convolution) 블록을 구현하는 PyTorch 모듈.

    세 개의 병렬 경로(1x1, 3x3, 1x5->5x1 컨볼루션)에서 특징을 추출하고,
    결과를 결합한 후 1x1 컨볼루션으로 최종 특징 맵을 생성한다.
    """
    def __init__(self, in_channels, out_channels=64):
        """
        MS_conv 모듈을 초기화한다.

        Args:
            in_channels (int): 입력 텐서의 채널 수.
            out_channels (int, optional): 최종 출력 텐서의 채널 수. 기본값은 64.
        """
        super(MS_conv, self).__init__()

        # 경로 1: 1x1 컨볼루션. 출력 채널 32.
        self.path1 = ConvLayer(in_channels, 32, kernel_size=1, activation=nn.LeakyReLU)

        # 경로 2: 3x3 컨볼루션. 출력 채널 64.
        self.path2 = ConvLayer(in_channels, 64, kernel_size=3, padding=1, activation=nn.LeakyReLU) # padding=1 for 'same' with 3x3 kernel

        # 경로 3: 1x5와 5x1 비대칭 컨볼루션을 순차적으로 적용. 출력 채널 32.
        self.path3 = nn.Sequential(
            ConvLayer(in_channels, 32, kernel_size=(1, 5), padding=(0, 2), activation=nn.LeakyReLU), # padding for 'same'
            ConvLayer(32, 32, kernel_size=(5, 1), padding=(2, 0), activation=nn.LeakyReLU) # padding for 'same'
        )

        # 최종 1x1 컨볼루션: 결합된 특징 맵(32+64+32=128)을 입력으로 받아 최종 출력 채널로 맞춘다.
        self.final_conv = ConvLayer(128, out_channels, kernel_size=1, activation=None)

    def forward(self, x):
        """
        순전파(forward pass)를 정의한다.
        """
        # 세 개의 병렬 경로로 입력을 통과시킨다.
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)

        # 각 경로의 출력을 채널(dimension 1)을 기준으로 결합(concatenate)한다.
        combined = torch.cat([out1, out2, out3], dim=1)

        # 결합된 텐서를 최종 1x1 컨볼루션 레이어에 통과시킨다.
        output = self.final_conv(combined)

        return output
