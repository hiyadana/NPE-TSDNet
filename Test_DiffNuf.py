import tensorflow as tf
import os
import numpy as np
import scipy.io as scio
import cv2

model_saving_path = '.\\model\\'
meta_file = '.\\model\\g_o_model.ckpt.meta'
meta_name = 'g_o_model.ckpt.meta'



tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #设置定量的GPU使用量
config = tf.compat.v1.ConfigProto()
test_path = '.\\dataset\\DiffNufTest\\HighNuf\\'
save_path = '.\\dataset\\DiffNufTest\\Results\\HighNuc\\Ours\\'
test_files = os.listdir(test_path)

with tf.compat.v1.Session() as sess:
    #从 .meta文件导入原始网络结构图：
    new_saver = tf.compat.v1.train.import_meta_graph(meta_file)
    new_saver.restore(sess,tf.train.latest_checkpoint(model_saving_path)) # 如果保存好的模型移动了位置，需要以记事本打开checkpoint文件修正绝对路径
    graph = tf.compat.v1.get_default_graph()
    new_x = tf.compat.v1.get_default_graph().get_tensor_by_name("x_:0")
    new_y = tf.compat.v1.get_default_graph().get_tensor_by_name("y_:0")
    output = tf.compat.v1.get_default_graph().get_tensor_by_name("op:0")
    for files in test_files:
        if os.path.splitext(files)[1] == '.mat':
            test_name = os.path.splitext(files)[0]
            test_mat = test_path + files
            mat_data = scio.loadmat(test_mat)
            x_gray = mat_data['Nuf']
            x_arr = np.asarray(x_gray)
            size_h = np.size(x_arr, 0)
            size_w = np.size(x_arr, 1)
            x_arr_norm = x_arr.astype('float32')/255
            x_batch = x_arr_norm.reshape((1, size_h, size_w, 1))
            y_y_=sess.run(output, feed_dict={new_x: x_batch})

            pre_array = np.array(y_y_)

            img_size_h = pre_array.shape[1]
            img_size_w = pre_array.shape[2]

            pre_img = pre_array.reshape((img_size_h, img_size_w))

            ##################存成mat
            scio.savemat(save_path + test_name + ".mat", {'pre': pre_img * 255})
            cv2.imwrite(save_path + test_name + ".png", pre_img * 255)
