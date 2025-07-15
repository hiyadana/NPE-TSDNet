# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:28:50 2021

@author: Administrator
"""

import tensorflow as tf

class O_OutPut(object):
    def __init__(self, inpt_o, o_pre):
        self.output = tf.add(inpt_o, o_pre, name='op')
        self.train_dicts = {}
        self.pred_dicts = {}