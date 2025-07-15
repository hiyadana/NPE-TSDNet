# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:26:04 2021

@author: Administrator
"""

"""获得数据"""

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
            label_G = mat_data['G']

            if i == 0:
                size_h = data_img.shape[0]
                size_w = data_img.shape[1]
                data_arr = np.empty(shape=(batch_size, size_h, size_w))
                G_arr = np.empty(shape=(batch_size, 1, size_w))

            data_img_array = data_img / 255

            data_arr[i] = data_img_array
            G_arr[i] = label_G

        x_batch = data_arr.reshape((batch_size, size_h, size_w, 1))
        g_batch = G_arr.reshape((batch_size, 1, size_w, 1))

        return x_batch, g_batch