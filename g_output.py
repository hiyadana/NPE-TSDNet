# -*- coding: utf-8 -*-
"""
Gain 보정 적용 모듈

이 스크립트는 계산된 이득 보정(gain correction) 계수를 원본 이미지에 적용하는 클래스를 정의한다.
"""

import tensorflow as tf

class G_OutPut(object):
    """
    G-네트워크의 최종 출력을 생성하는 클래스.
    
    이 클래스는 원본 입력 이미지와 추정된 이득 보정 계수를 입력받아,
    요소별 곱셈(element-wise multiplication)을 통해 이득 불균일성이 보정된 이미지를 출력한다.
    """
    def __init__(self, inpt_g, g_pre):
        """
        G_OutPut 객체를 초기화한다.

        Args:
            inpt_g (tf.Tensor): 원본 입력 이미지 텐서. 형태: [batch, height, width, channels].
            g_pre (tf.Tensor): 추정된 이득 보정 계수 텐서. 형태: [batch, 1, width, channels].
        """
        # tf.multiply를 사용하여 입력 이미지와 이득 보정 계수를 요소별로 곱한다.
        # 여기서 텐서플로우의 브로드캐스팅(broadcasting) 기능이 사용된다.
        # g_pre의 형태가 [batch, 1, width, channels]이므로, 높이(height) 차원에 대해
        # inpt_g의 높이와 일치하도록 자동으로 확장되어 곱셈이 수행된다.
        # 결과적으로 각 열의 모든 픽셀이 해당 열의 보정 계수와 곱해져 이득이 보정된다.
        self.output = tf.multiply(inpt_g, g_pre, name='g_op')
