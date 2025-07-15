# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:28:50 2021

@author: Administrator
"""

import tensorflow as tf

class OutPut_O(object):
    def __init__(self, o_pre):
        self.output = tf.reduce_mean(o_pre, axis=1, keepdims=True, name="pre_o")
        self.train_dicts = {}
        self.pred_dicts = {}