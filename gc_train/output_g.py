# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:24:52 2021

@author: Administrator
"""

import tensorflow as tf

class OutPut_G(object):
    def __init__(self, g_pre):
        self.output = tf.reduce_mean(g_pre, axis=1, keepdims=True, name="pre_g")                                                      #axis应该为1.该tensor四个维度，从左往右为0，1，2，3，按列计算是1所标识的第2个维度也就是行数变为1
        self.train_dicts = {}
        self.pred_dicts = {}