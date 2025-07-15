"""多尺度特征提取单元"""
import tensorflow as tf
from conv_layer import ConvLayer

class MS_conv(object):
    def __init__(self, inpt, channels=64, ms_name="ms"):
        #为参数设置name
        name_1 = ms_name + "_1"
        name_2 = ms_name + "_2"
        name_3_1 = ms_name + "_3_1"
        name_3_2 = ms_name + "_3_2"
        name_0 = ms_name + "_0"

        self.inpt = inpt
        self.channels = channels

        MS_1_conv = ConvLayer(self.inpt, filter_shape=[1, 1, self.channels, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                     padding="SAME", cl_name=name_1)

        MS_2_conv = ConvLayer(self.inpt, filter_shape=[3, 3, self.channels, 64], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                padding="SAME", cl_name=name_2)

        MS_3_1_conv = ConvLayer(self.inpt, filter_shape=[1, 5, self.channels, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                              padding="SAME", cl_name=name_3_1)
        MS_3_2_conv = ConvLayer(MS_3_1_conv.output, filter_shape=[5, 1, 32, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                padding="SAME", cl_name=name_3_2)


        frames = [MS_1_conv.output, MS_2_conv.output, MS_3_2_conv.output]
        layer_pj = tf.concat(frames, 3)  # 按通道方向拼接，３表示［ａ，ｂ，ｃ，ｄ］中的ｄ
        layer_conv = ConvLayer(layer_pj, filter_shape=[1, 1, 128, 64], strides=[1, 1, 1, 1], activation=None,
                                   padding="SAME", cl_name=name_0)

        self.params = MS_1_conv.params + MS_2_conv.params + MS_3_1_conv.params + MS_3_2_conv.params + layer_conv.params
        self.output = layer_conv.output