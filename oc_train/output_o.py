# -*- coding: utf-8 -*-
"""
O-네트워크 출력 처리 모듈

@author: Administrator
"""

import tensorflow as tf

class OutPut_O(object):
    """
    O-네트워크(Offset Correction Network)의 출력을 처리하는 클래스.

    이 클래스는 O-네트워크의 최종 특징 맵을 입력받아, 각 열(column)의 평균값을 계산한다.
    이것은 이미지의 오프셋(offset) 보정 계수가 주로 열에 따라 변한다는 가정에 기반한 연산이다.
    """
    def __init__(self, o_pre):
        """
        OutPut_O 객체를 초기화한다.

        Args:
            o_pre (tf.Tensor): O-네트워크의 최종 출력 텐서. 형태는 [batch, height, width, channels].
        """
        # 입력 텐서의 높이(axis=1)를 기준으로 평균을 계산한다.
        # keepdims=True는 차원을 유지하여 출력 텐서의 형태가 [batch, 1, width, channels]가 되도록 한다.
        # 이 결과는 각 열에 대한 평균값, 즉 오프셋 보정 계수(offset correction factor)의 추정치로 사용된다.
        self.output = tf.reduce_mean(o_pre, axis=1, keepdims=True, name="pre_o")
        
        # 학습 및 예측 시 feed_dict를 구성하는 데 사용될 빈 딕셔너리.
        self.train_dicts = {}
        self.pred_dicts = {}
