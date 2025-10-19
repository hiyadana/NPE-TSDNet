# -*- coding: utf-8 -*-
"""
비용 함수(Cost Function) 모듈

이 스크립트는 모델의 예측 값과 실제 값 사이의 손실(loss)을 계산하는 클래스를 정의한다.
"""

import tensorflow as tf

class CostFunction(object):
    """
    모델의 비용(손실)을 계산하는 클래스.
    
    이 클래스는 예측 이미지와 원본 이미지(정답)를 비교하여
    평균 제곱 오차(Mean Squared Error, MSE)를 계산한다.
    """
    def __init__(self, y_pre, y):
        """
        CostFunction 객체를 초기화한다.

        Args:
            y_pre (tf.Tensor): 모델이 예측한 이미지 텐서.
            y (tf.Tensor): 원본 이미지(정답) 텐서.
        """
        # 1. (y - y_pre): 실제 값과 예측 값의 차이(오차)를 계산한다.
        # 2. tf.square(...): 오차의 각 요소에 대해 제곱을 계산한다.
        # 3. tf.reduce_mean(...): 제곱된 오차들의 전체 평균을 계산하여 최종 손실 값(MSE)을 구한다.
        self.output = tf.reduce_mean(tf.square(y - y_pre), name='loss')
