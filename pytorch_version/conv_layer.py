# -*- coding: utf-8 -*-
"""
PyTorch 버전의 컨볼루션 레이어 모듈

TensorFlow로 작성된 기존의 ConvLayer 클래스를 PyTorch의 nn.Module을
사용하여 재구현한다. nn.Conv2d와 활성화 함수를 캡슐화하여 재사용성을 높인다.
"""

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """
    하나의 2D 컨볼루션 레이어와 활성화 함수를 포함하는 PyTorch 모듈.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation=nn.ReLU, bias=True):
        """
        ConvLayer 모듈을 초기화한다.

        Args:
            in_channels (int): 입력 채널 수.
            out_channels (int): 출력 채널 수.
            kernel_size (int or tuple): 컨볼루션 커널의 크기.
            stride (int or tuple, optional): 컨볼루션의 스트라이드. 기본값은 1.
            padding (str or int, optional): 패딩 유형. PyTorch Conv2d와 동일. 기본값은 'same'.
            activation (nn.Module class, optional): 사용할 활성화 함수의 클래스 (예: nn.ReLU). None일 경우 활성화 함수 없음. 기본값은 nn.ReLU.
            bias (bool, optional): 편향(bias)을 사용할지 여부. 기본값은 True.
        """
        super(ConvLayer, self).__init__()
        
        # PyTorch의 nn.Conv2d 레이어를 정의한다.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        
        # 활성화 함수를 설정한다.
        if activation is not None:
            # LeakyReLU의 경우, 원본 코드의 leak 파라미터(0.2)를 적용한다.
            if activation == nn.LeakyReLU:
                self.activation = nn.LeakyReLU(0.2, inplace=True)
            else:
                self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        """
        순전파(forward pass)를 정의한다.
        """
        # 컨볼루션 연산을 적용한다.
        x = self.conv(x)
        # 활성화 함수가 설정된 경우 적용한다.
        if self.activation is not None:
            x = self.activation(x)
        return x
