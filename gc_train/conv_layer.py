# -*- coding: utf-8 -*-
"""
컨볼루션 레이어 모듈

이 스크립트는 TensorFlow를 사용하여 표준 2D 컨볼루션 레이어를
쉽게 생성할 수 있도록 캡슐화한 ConvLayer 클래스를 정의한다.
"""

import tensorflow as tf

class ConvLayer(object):
    """
    단일 2D 컨볼루션 레이어를 생성하는 클래스.

    이 클래스는 가중치(W)와 편향(b) 변수를 초기화하고, 컨볼루션 연산을 수행하며,
    지정된 활성화 함수를 적용하는 과정을 포함한다.
    """

    def __init__(self, inpt, filter_shape, strides=(1, 1, 1, 1),
                 padding="SAME", activation=tf.nn.relu, bias_setting=True, cl_name="cl"):
        """
        ConvLayer 객체를 초기화한다.

        Args:
            inpt (tf.Tensor): 입력 텐서.
            filter_shape (list or tuple): 컨볼루션 필터의 형태. [높이, 너비, 입력 채널, 출력 채널].
            strides (tuple, optional): 컨볼루션의 스트라이드. 기본값은 (1, 1, 1, 1).
            padding (str, optional): 패딩 유형. "SAME" 또는 "VALID". 기본값은 "SAME".
            activation (function, optional): 활성화 함수. 기본값은 tf.nn.relu.
            bias_setting (bool, optional): 편향(bias)을 사용할지 여부. 기본값은 True.
            cl_name (str, optional): 레이어의 이름 접두사. 가중치와 편향 변수명에 사용됨. 기본값은 "cl".
        """
        # 가중치(weight)와 편향(bias) 변수의 이름을 설정.
        w_name = cl_name + "_w"
        b_name = cl_name + "_b"
        
        # 입력 텐서 설정.
        self.input = inpt
        
        # 컨볼루션 커널(가중치) 변수 초기화. 절단 정규분포로 초기화한다.
        self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name=w_name)
        
        # 편향 변수 초기화. bias_setting이 True일 경우에만 생성.
        if bias_setting:
            self.b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1),
                                 dtype=tf.float32, name=b_name)
        else:
            self.b = None
            
        # 컨볼루션 연산 수행.
        conv_output = tf.nn.conv2d(self.input, filter=self.W, strides=strides,
                                   padding=padding)
        # 편향이 존재하면 컨볼루션 결과에 더해줌.
        conv_output = conv_output + self.b if self.b is not None else conv_output
        
        # 활성화 함수 적용. activation이 None이면 선형 출력.
        self.output = conv_output if activation is None else activation(conv_output)
        
        # 학습 가능한 파라미터(가중치, 편향)를 리스트로 저장.
        self.params = [self.W, self.b] if self.b is not None else [self.W]

    @staticmethod
    def LeakyRelu(x, leak=0.2, name='LeakyRelu'):
        """
        Leaky ReLU 활성화 함수를 구현한다.

        Args:
            x (tf.Tensor): 입력 텐서.
            leak (float, optional): 음수 영역의 기울기. 기본값은 0.2.
            name (str, optional): 연산의 이름. 기본값은 'LeakyRelu'.

        Returns:
            tf.Tensor: Leaky ReLU가 적용된 텐서.
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * tf.abs(x)
