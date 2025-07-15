# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:21:07 2021

@author: Administrator
"""

"""卷积层"""
import tensorflow as tf

class ConvLayer(object):

    def __init__(self, inpt, filter_shape, strides=(1, 1, 1, 1),
                 padding="SAME", activation=tf.nn.relu, bias_setting=True, cl_name="cl"):
        #给params设置name
        w_name = cl_name + "_w"
        b_name = cl_name + "_b"
        # 设置输入;
        self.input = inpt
        # 初始化卷积核;
        self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name=w_name)
        if bias_setting:
            self.b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1),
                                 dtype=tf.float32, name=b_name)
        else:
            self.b = None
        # 计算卷积操作的输出;
        conv_output = tf.nn.conv2d(self.input, filter=self.W, strides=strides,
                                   padding=padding)
        conv_output = conv_output + self.b if self.b is not None else conv_output
        # 设置输出;
        self.output = conv_output if activation is None else activation(conv_output)
        # 设置参数;
        self.params = [self.W, self.b] if self.b is not None else [self.W, ]

    def LeakyRelu(x, leak=0.2, name='LeakyRelu'):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * tf.abs(x)