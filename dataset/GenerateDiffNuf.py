import cv2
import random
import numpy as np
import scipy.io as scio
import os

mat_savepath = ".\\DiffNufTest\\HighNuf\\" #MediumNuf  LowNuf

img_path = ".\\DiffNufTest\\Ori\\"
img_list = os.listdir(img_path)
jpg_num = len(img_list)
mat_index = 0
# 获得噪声参数
mu_g = 1  # 条带增益噪声均值
mu_o = 0  # 高斯偏置噪声均值
sigma_g = 0.15  # 条带增益噪声标准差MediumNuf-0.10  LowNuf-0.05
sigma_o = 25  # 偏置噪声标准差MediumNuf-15  LowNuf-5
print("Starting...")
for index in range(jpg_num):
    img_name = img_list[index]
    img_file = img_path+img_name
    ori_img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    size_h,size_w = ori_img.shape

    ori_arr = np.asarray(ori_img)
    nuf = np.asarray(ori_img)
    nuf = np.require(nuf, dtype='f4', requirements=['O', 'W'])
    nuf.flags.writeable = True  # 将数组改为读写模式

    G_list = []
    O_list = []


    # 对图像加噪
    for i in range(0, size_w):
        # 获取该像元的增益和偏置
        g_ori_i = round(random.uniform(mu_g - sigma_g, mu_g + sigma_g), 4)
        o_ori_i = round(random.gauss(mu_o, sigma_o), 2)

        nuf[:, i] = g_ori_i * nuf[:, i] + o_ori_i

        G_i = round(1 / g_ori_i, 4)
        O_i = round(-o_ori_i / g_ori_i, 2)

        G_list.append(G_i)
        O_list.append(O_i)

    G_arr = np.asarray(G_list)
    O_arr = np.asarray(O_list)

    mat_dir = mat_savepath + img_name.split(".")[0] + ".mat"
    scio.savemat(mat_dir, {'Ori': ori_arr, 'Nuf': nuf})
    nuf_dir = mat_savepath + img_name.split(".")[0] + ".png"
    cv2.imwrite(nuf_dir,nuf)

    print(index)