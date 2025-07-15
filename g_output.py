# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:24:52 2021

@author: Administrator
"""

import tensorflow as tf

class G_OutPut(object):
    def __init__(self, inpt_g, g_pre):
        self.output = tf.multiply(inpt_g, g_pre, name='g_op')