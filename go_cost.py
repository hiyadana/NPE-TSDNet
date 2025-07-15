# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:22:26 2021

@author: Administrator
"""

import tensorflow as tf

class CostFunction(object):
    def __init__(self, y_pre, y):
        self.output = tf.reduce_mean(tf.square(y - y_pre), name='loss')