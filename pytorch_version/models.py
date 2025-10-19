# -*- coding: utf-8 -*-
"""
PyTorch 버전의 NPE-TSDNet 모델 아키텍처 정의

이 스크립트는 G-Net, O-Net, 그리고 이 둘을 결합한 최종 NPE-TSDNet 모델을
PyTorch의 nn.Module 클래스로 정의한다.
"""

import torch
import torch.nn as nn
from pytorch_version.multi_scale_conv import MS_conv
from pytorch_version.conv_layer import ConvLayer


class G_Net(nn.Module):
    """
    G-네트워크(Gain Correction Network) PyTorch 모델.

    8개의 다중 스케일 컨볼루션(MS_conv) 블록과 최종 출력 레이어로 구성된다.
    사전 학습 시, 입력 이미지로부터 열(column) 단위의 이득 보정 계수를 추정하는 역할을 한다.
    """
    def __init__(self):
        super(G_Net, self).__init__()
        # G-Net의 특징 추출부: 8개의 MS_conv 블록과 1개의 1x1 ConvLayer
        self.features = nn.Sequential(
            MS_conv(in_channels=1, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            ConvLayer(in_channels=64, out_channels=1, kernel_size=1, activation=None)
        )

    def forward(self, x):
        # 특징 맵을 계산
        feature_map = self.features(x)
        # 열 단위 이득 보정 계수 계산 (Height 축으로 평균)
        # PyTorch에서 (N, C, H, W) 텐서의 높이(H)는 dim=2에 해당한다.
        gain_vector = torch.mean(feature_map, dim=2, keepdim=True)
        return gain_vector


class O_Net(nn.Module):
    """
    O-네트워크(Offset Correction Network) PyTorch 모델.

    15개의 다중 스케일 컨볼루션(MS_conv) 블록과 최종 출력 레이어로 구성된다.
    사전 학습 시, 입력 이미지로부터 열(column) 단위의 오프셋 보정 계수를 추정하는 역할을 한다.
    """
    def __init__(self):
        super(O_Net, self).__init__()
        # O-Net의 특징 추출부: 15개의 MS_conv 블록과 1개의 1x1 ConvLayer
        self.features = nn.Sequential(
            MS_conv(in_channels=1, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            MS_conv(in_channels=64, out_channels=64),
            ConvLayer(in_channels=64, out_channels=1, kernel_size=1, activation=None)
        )

    def forward(self, x):
        # 특징 맵을 계산
        feature_map = self.features(x)
        # 열 단위 오프셋 보정 계수 계산 (Height 축으로 평균)
        offset_vector = torch.mean(feature_map, dim=2, keepdim=True)
        return offset_vector


class NPE_TSDNet(nn.Module):
    """
    G-Net과 O-Net을 결합한 최종 NPE-TSDNet 모델.
    """
    def __init__(self):
        super(NPE_TSDNet, self).__init__()
        # G-Net과 O-Net의 특징 추출 부분만 가져와서 정의한다.
        # 사전 학습된 가중치를 나중에 이 부분에 불러온다.
        self.g_net_features = G_Net().features
        self.o_net_features = O_Net().features

    def forward(self, x):
        """
        전체 모델의 순전파(forward pass)를 정의한다.
        이득 보정 후 오프셋 보정을 순차적으로 적용한다.
        """
        # 1. G-Net을 통해 이득 보정 계수를 계산하고 적용한다.
        gain_map = self.g_net_features(x)
        gain_vector = torch.mean(gain_map, dim=2, keepdim=True)
        # PyTorch의 브로드캐스팅을 통해 gain_vector가 x의 크기에 맞게 확장되어 곱해진다.
        g_out = x * gain_vector

        # 2. O-Net을 통해 오프셋 보정 계수를 계산하고 적용한다.
        offset_map = self.o_net_features(g_out)
        offset_vector = torch.mean(offset_map, dim=2, keepdim=True)
        # 브로드캐스팅을 통해 offset_vector가 g_out의 크기에 맞게 확장되어 더해진다.
        o_out = g_out + offset_vector

        return o_out