# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:26:04 2021

@author: Administrator
"""

"""获得数据"""

from PIL import Image
import os
import numpy as np
import scipy.io as scio

class GetData(object):
    def get_data(mat_path, batch_1_num, batch_size):

        for i in range(batch_size):
            mat_name_num = batch_1_num + i
            mat_name = str(mat_name_num).zfill(6) + ".mat"
            mat_fpath = os.path.join(mat_path, mat_name)
            mat_data = scio.loadmat(mat_fpath)

            data_img = mat_data['Nuf']
            label_img = mat_data['Ori']

            if i == 0:
                size_h = data_img.shape[0]
                size_w = data_img.shape[1]
                data_arr = np.empty(shape=(batch_size, size_h, size_w))
                ori_arr = np.empty(shape=(batch_size, size_h, size_w))

            data_img_array = np.asarray(data_img)
            ori_img_array = np.asarray(label_img)

            data_img_array = data_img_array.astype('float32')
            ori_img_array = ori_img_array.astype('float32')

            data_img_arr = data_img_array/255
            ori_img_arr = ori_img_array/255

            data_arr[i] = data_img_arr
            ori_arr[i] = ori_img_arr

        x_batch = data_arr.reshape((batch_size, size_h, size_w, 1))
        y_batch = ori_arr.reshape((batch_size, size_h, size_w, 1))

        return x_batch, y_batch