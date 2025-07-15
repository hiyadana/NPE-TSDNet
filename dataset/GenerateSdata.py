from glob import glob
import cv2
import random
import numpy as np
import scipy.io as scio


mat_savepath = "./traindata" # "./validata"
loc_list = glob("./MS-COCO/*.jpg") #MS-COCO数据下载地址：http://cocodataset.org

jpg_num = len(loc_list)
ori_list = list(range(jpg_num))
random.shuffle(ori_list)

train_num = 500000
mat_list = list(range(1,train_num+1))
random.shuffle(mat_list)

mat_index = 0
print("Starting...")
for index in range(jpg_num):
    ori_num = ori_list[index]
    ms_img = cv2.imread(loc_list[ori_num])
    size_h,size_w,channel = ms_img.shape
    h_cai = 128  # 裁剪高度
    w_cai = 128  # 裁剪宽度
    for right in range(w_cai, size_w, w_cai):
        for lower in range(h_cai, size_h, h_cai):

            mat_num = mat_list[mat_index]

            cropped_img = ms_img[lower - h_cai:lower, right - w_cai:right]
            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

            ori_arr = np.asarray(gray_img)
            img = np.asarray(gray_img)
            img = np.require(img, dtype='f4', requirements=['O', 'W'])
            img.flags.writeable = True  # 将数组改为读写模式

            G_list = []
            O_list = []

            # 获得噪声参数
            mu_g = 1  # 条带增益噪声均值
            mu_o = 0  # 高斯偏置噪声均值
            sigma_g = random.randint(0, 18) / 100  # 条带增益噪声标准差
            sigma_o = random.randint(0, 30)  # 偏置噪声标准差
            # 对图像加噪
            for i in range(0, w_cai):
                # 获取该像元的增益和偏置
                g_ori_i = round(random.uniform(mu_g - sigma_g, mu_g + sigma_g), 4)
                o_ori_i = round(random.gauss(mu_o, sigma_o), 2)

                img[:, i] = g_ori_i * img[:, i] + o_ori_i

                G_i = round(1 / g_ori_i, 4)
                O_i = round(-o_ori_i / g_ori_i, 2)

                G_list.append(G_i)
                O_list.append(O_i)

            G_arr = np.asarray(G_list)
            O_arr = np.asarray(O_list)

            mat_dir = mat_savepath + str(mat_num).zfill(6) + ".mat"
            scio.savemat(mat_dir, {'Ori': ori_arr, 'Nuf': img, 'G': G_arr, 'O': O_arr})

            print(mat_index)
            mat_index = mat_index + 1

            if mat_index>train_num-1:
                exit(0)