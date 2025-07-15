# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:05:59 2021

@author: Administrator
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from g_getdata import GetData as GD
from conv_layer import ConvLayer                                                                                  #卷积层
from multi_scale_conv import MS_conv
from output_g import OutPut_G as OPG
from g_cost import CostFunction as GCF
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # 定义输入输出占位符;
    #tf.compat.v1.disable_eager_execution()
    x_g = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='x_g')
    y_g = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='y_g')

    inpt_g = x_g
    # 卷积层layer1;
    layer_g_1_conv = MS_conv(inpt_g, 1, ms_name="g_1")

    # 卷积层layer2
    layer_g_2_conv = MS_conv(layer_g_1_conv.output, ms_name="g_2")

    # 卷积层layer3
    layer_g_3_conv = MS_conv(layer_g_2_conv.output, ms_name="g_3")

    # 卷积层layer4
    layer_g_4_conv = MS_conv(layer_g_3_conv.output, ms_name="g_4")

    # 卷积层layer5
    layer_g_5_conv = MS_conv(layer_g_4_conv.output, ms_name="g_5")

    # 卷积层layer6
    layer_g_6_conv = MS_conv(layer_g_5_conv.output, ms_name="g_6")

    # 卷积层layer7
    layer_g_7_conv = MS_conv(layer_g_6_conv.output, ms_name="g_7")

    # 卷积层layer8
    layer_g_8_conv = MS_conv(layer_g_7_conv.output, ms_name="g_8")

########################################################################################################################
    layer_g_conv = ConvLayer(layer_g_8_conv.output, filter_shape=[1, 1, 64, 1], strides=[1, 1, 1, 1], activation=None,
                               padding="SAME", cl_name="g_0")

    layer_g_output = OPG(layer_g_conv.output)

    cost_cal = GCF(layer_g_output.output, y_g)
########################################################################################################################
    # 分别设置训练参数;
    params_g = layer_g_1_conv.params + layer_g_2_conv.params + layer_g_3_conv.params + layer_g_4_conv.params + \
             layer_g_5_conv.params + layer_g_6_conv.params + layer_g_7_conv.params + layer_g_8_conv.params + \
               layer_g_conv.params

    train_dicts = layer_g_output.train_dicts                                                                  #训练dicts;

    pred_dicts = layer_g_output.pred_dicts                                                                    #预测dicts;

    cost = cost_cal.output                                                                                 # 定义代价cost;

    predictor = layer_g_output.output                                                                     # 定义网络的预测;
    # 定义训练器;
    num_epoch = tf.Variable(0, name='epoch', trainable=False)
    assign_op = tf.assign_add(num_epoch, 1)
    boundaries = [25]
    learning_rates = [0.001, 0.0001]
    with tf.control_dependencies([assign_op]):
        learning_rate = tf.train.piecewise_constant(x=num_epoch, boundaries=boundaries, values=learning_rates)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=params_g)
########################################################################################################################
########################################################################################################################
    init = tf.global_variables_initializer()                                                                #初始化所有变量
    saver = tf.train.Saver()                                                                                 #用于保存模型
    # 定义训练参数
    training_epochs = 50
    batch_size = 16
    display_step = 1
    # 训练集、验证集、测试集数据位置及其标签位置和txt文件名
    train_path = "../dataset/trainset"  # 训练集数据位置
    vali_path = "../dataset/valiset"  # 测试集数据位置
    # 测试结果
    vali_pre_save = "../dataset/gValiResults/"
    vali_txt_result = './Vali_cost.txt'
    vali_excel_path = './'
    # 训练得到的模型保存位置和名称
    model_path_name = "./Nmodel/g_model.ckpt"
    excel_path = "./"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #设置定量的GPU使用量
    config = tf.ConfigProto()

    train_jpg_num = 200000
    train_batch_num = int(train_jpg_num / batch_size)

    ffff = np.empty(shape=(training_epochs * train_batch_num, 3))
########################################################################################################################
########################################################################################################################
    # 开始训练
    print("Start to train...")
    with tf.Session() as sess:
        sess.run(init)
        FFF = 0
        for epoch in range(training_epochs):
            #参数;
            cost_sum = 0.0
            avg_cost = 0.0
            training_batch = 0
            # 跑多个batch;
            #用于标识训练进展
            train_batch_ed = 1

            for i in range(train_batch_num):
                training_batch += 1
                batch_1_img = i * batch_size + 1                                                  #该batch中第一张图像的名称

                #获得输入x_batch,y_batch
                x_batch,y_batch = GD.get_data(train_path, batch_1_img, batch_size)
                train_dicts.update({x_g: x_batch, y_g: y_batch})

                # 执行训练;
                sess.run(train_op, feed_dict=train_dicts)
                # 计算cost;
                avg_cost = sess.run(cost, feed_dict=train_dicts)
                #标识训练进展
                train_progress = "Training epoch: " + str(epoch+1) + ".\n" + "Remaining training epoch: " +\
                                 str(training_epochs-epoch-1) + ".\n" + "Training progress of the epoch: " +\
                                 str(train_batch_ed) + ".\n" + "Remaining training progress of the epoch: " + \
                                 str(train_batch_num - train_batch_ed) + ".\n" + "avg_cost:" + str(avg_cost) + ".\n"
                print("\r" + train_progress, end= '')
                ffff[FFF] = [epoch + 1, train_batch_ed, avg_cost]
                FFF = FFF + 1
                train_batch_ed = train_batch_ed + 1

        #保存训练好的模型
        saver.save(sess, model_path_name)

        print("Finished!")

        ffff_data = pd.DataFrame(ffff)
        excel_result = excel_path + 'Cost' + '.xlsx'
        writer = pd.ExcelWriter(excel_result)
        ffff_data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

         # 开始验证;
        x_dirs = os.listdir(vali_path)
        vali_jpg_num = len([name for name in x_dirs if name.endswith(".mat")])

        for j in range(vali_jpg_num):

            vali_num = j+1
            x_batch,y_batch = GD.get_data(vali_path, vali_num, 1)
            pred_dicts.update({x_g: x_batch})
            y_y_ = sess.run(predictor, feed_dict=pred_dicts)
            cost_t = GCF(y_y_, y_batch).output
            cost_tt=sess.run(cost_t)

            ori_array = np.array(y_batch)
            pre_array = np.array(y_y_)

            txt_text = 'Vali_'+ str(vali_num) + ': ' + str(cost_tt) + '\n'
            listVali = open(vali_txt_result, 'a')
            listVali.write(txt_text)
            listVali.close()

            ori_2d = ori_array.squeeze()#压缩所有为1的维度
            pre_2d = pre_array.squeeze()
            ori_data = pd.DataFrame(ori_2d)
            pre_data = pd.DataFrame(pre_2d)
            ori_sheet_name = 'page_' + str(1)
            pre_sheet_name = 'page_' + str(2)
            vali_excel_result=vali_excel_path+'Vali_label_'+str(vali_num).zfill(6)+'.xlsx'
            writer = pd.ExcelWriter(vali_excel_result)
            ori_data.to_excel(writer, ori_sheet_name, float_format='%.5f')
            pre_data.to_excel(writer, pre_sheet_name, float_format='%.5f')
            writer.save()