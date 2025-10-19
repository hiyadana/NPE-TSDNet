# -*- coding: utf-8 -*-
"""
오프셋 보정 적용 모듈

이 스크립트는 계산된 오프셋 보정(offset correction) 계수를
이득(gain)이 보정된 이미지에 적용하는 클래스를 정의한다.
"""

import tensorflow as tf

class O_OutPut(object):
    """
    O-네트워크의 최종 출력을 생성하는 클래스.
    
    이 클래스는 이득이 보정된 이미지와 추정된 오프셋 보정 계수를 입력받아,
    요소별 덧셈(element-wise addition)을 통해 오프셋 불균일성이 보정된 최종 이미지를 출력한다.
    """
    def __init__(self, inpt_o, o_pre):
        """
        O_OutPut 객체를 초기화한다.

        Args:
            inpt_o (tf.Tensor): 이득이 보정된 이미지 텐서. 형태: [batch, height, width, channels].
            o_pre (tf.Tensor): 추정된 오프셋 보정 계수 텐서. 형태: [batch, 1, width, channels].
        """
        # tf.add를 사용하여 이득 보정 이미지와 오프셋 보정 계수를 요소별로 더한다.
        # 여기서도 텐서플로우의 브로드캐스팅(broadcasting) 기능이 사용된다.
        # o_pre의 형태가 [batch, 1, width, channels]이므로, 높이(height) 차원에 대해
        # inpt_o의 높이와 일치하도록 자동으로 확장되어 덧셈이 수행된다.
        # 결과적으로 각 열의 모든 픽셀에 해당 열의 보정 계수가 더해져 오프셋이 보정된다.
        self.output = tf.add(inpt_o, o_pre, name='op')
        
        # 학습 및 예측 시 feed_dict를 구성하는 데 사용될 빈 딕셔너리.
        self.train_dicts = {}
        self.pred_dicts = {}
