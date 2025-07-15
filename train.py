import tensorflow as tf
import os
from getdata import GetData as GD
import numpy as np
from PIL import Image
from gc_train.conv_layer import ConvLayer
from gc_train.multi_scale_conv import MS_conv
from gc_train.output_g import OutPut_G as OPG
from g_output import G_OutPut as GOP
from oc_train.output_o import OutPut_O as OPO
from o_output import O_OutPut as OOP
from go_cost import CostFunction as GOCF


if __name__ == "__main__":

    # 定义输入输出占位符;
    #tf.compat.v1.disable_eager_execution()#2.0以上的tf后均需加.compat.v1，包括子程序conv_layer
    x_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='x_')
    y_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='y_')

    inpt_g = x_
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

    g_out = GOP(inpt_g, layer_g_output.output)
########################################################################################################################
    # 卷积层layer1;
    layer_o_1_conv = MS_conv(g_out.output, 1, ms_name="o_1")

    # 卷积层layer2
    layer_o_2_conv = MS_conv(layer_o_1_conv.output, ms_name="o_2")

    # 卷积层layer3
    layer_o_3_conv = MS_conv(layer_o_2_conv.output, ms_name="o_3")

    # 卷积层layer4
    layer_o_4_conv = MS_conv(layer_o_3_conv.output, ms_name="o_4")

    # 卷积层layer5
    layer_o_5_conv = MS_conv(layer_o_4_conv.output, ms_name="o_5")

    # 卷积层layer6
    layer_o_6_conv = MS_conv(layer_o_5_conv.output, ms_name="o_6")

    # 卷积层layer7
    layer_o_7_conv = MS_conv(layer_o_6_conv.output, ms_name="o_7")

    # 卷积层layer8
    layer_o_8_conv = MS_conv(layer_o_7_conv.output, ms_name="o_8")

    # 卷积层layer9
    layer_o_9_conv = MS_conv(layer_o_8_conv.output, ms_name="o_9")

    # 卷积层layer10
    layer_o_10_conv = MS_conv(layer_o_9_conv.output, ms_name="o_10")

    # 卷积层layer11;
    layer_o_11_conv = MS_conv(layer_o_10_conv.output, ms_name="o_11")

    # 卷积层layer12
    layer_o_12_conv = MS_conv(layer_o_11_conv.output, ms_name="o_12")

    # 卷积层layer13
    layer_o_13_conv = MS_conv(layer_o_12_conv.output, ms_name="o_13")

    # 卷积层layer14
    layer_o_14_conv = MS_conv(layer_o_13_conv.output, ms_name="o_14")

    # 卷积层layer15
    layer_o_15_conv = MS_conv(layer_o_14_conv.output, ms_name="o_15")
    ####################################################################################################################
    layer_o_conv = ConvLayer(layer_o_15_conv.output, filter_shape=[1, 1, 64, 1], strides=[1, 1, 1, 1], activation=None,
                             padding="SAME", cl_name="o_0")

    layer_o_output = OPO(layer_o_conv.output)

    o_out = OOP(g_out.output, layer_o_output.output)

    cost_cal = GOCF(o_out.output, y_)
    ####################################################################################################################
    # 分别设置训练参数;
    params_g = layer_g_1_conv.params + layer_g_2_conv.params + layer_g_3_conv.params + layer_g_4_conv.params + \
               layer_g_5_conv.params + layer_g_6_conv.params + layer_g_7_conv.params + layer_g_8_conv.params + \
               layer_g_conv.params
    params_o = layer_o_1_conv.params + layer_o_2_conv.params + layer_o_3_conv.params + layer_o_4_conv.params + \
               layer_o_5_conv.params + layer_o_6_conv.params + layer_o_7_conv.params + layer_o_8_conv.params + \
               layer_o_9_conv.params + layer_o_10_conv.params + layer_o_11_conv.params + layer_o_12_conv.params + \
               layer_o_13_conv.params + layer_o_14_conv.params + layer_o_15_conv.params + layer_o_conv.params
    params = params_g + params_o

    train_dicts = o_out.train_dicts                                                                          # 训练dicts;

    pred_dicts = o_out.pred_dicts                                                                            # 预测dicts;

    cost = cost_cal.output                                                                                 # 定义代价cost;

    predictor = o_out.output                                                                              # 定义网络的预测;

    # 定义训练器;
    num_epoch = tf.Variable(0, name='epoch', trainable=False)#这个不需加.compat.v1
    assign_op = tf.assign_add(num_epoch, 1)
    boundaries = [25]
    learning_rates = [0.0001, 0.00001]
    with tf.control_dependencies([assign_op]):
        learning_rate = tf.train.piecewise_constant(x=num_epoch, boundaries=boundaries, values=learning_rates)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=params)
########################################################################################################################
########################################################################################################################
    # 训练集、验证集数据位置及其标签位置和txt文件名
    train_path = "./dataset/trainset"  # 训练集数据位置
    vali_path = "./dataset/valiset"  # 测试集数据位置
    # 验证结果
    vali_pre_save = "./dataset/goValiResults/"
    vali_txt_result = './Vali_cost.txt'
    vali_excel_path = './'
    # 训练得到的模型保存位置和名称
    model_path_name = "./Nmodel/g_o_model.ckpt"
    model_g_path = './gc_train/model/' #注意将checkpoint中的路径修改为其所在目录
    model_o_path = './oc_train/model/' #注意将checkpoint中的路径修改为其所在目录

    init = tf.global_variables_initializer()  # 初始化所有变量

    saver_g = tf.train.Saver(params_g)
    saver_o = tf.train.Saver(params_o)

    saver = tf.train.Saver()  # 用于保存模型
    # 定义训练参数
    training_epochs = 50
    batch_size = 16
    display_step = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #设置定量的GPU使用量
    config = tf.ConfigProto()

    train_jpg_num = 300000
    train_batch_num = int(train_jpg_num / batch_size)

    ffff = np.empty(shape=(training_epochs * train_batch_num, 3))
########################################################################################################################
########################################################################################################################
    # 开始训练
    print("Start to train...")
    with tf.Session() as sess:
        sess.run(init)
        saver_g.restore(sess, tf.train.latest_checkpoint(model_g_path))
        saver_o.restore(sess, tf.train.latest_checkpoint(model_o_path))
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
                batch_1_img = i * batch_size + 200001   # 前200000个训练数据已用于预训练子网络                                                                        #该batch中第一张图像的名称

                #获得输入x_batch,y_batch
                x_batch,y_batch = GD.get_data(train_path, batch_1_img, batch_size)
                train_dicts.update({x_: x_batch, y_: y_batch})

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

         # 开始验证;
        x_dirs = os.listdir(vali_path)
        vali_jpg_num = len([name for name in x_dirs if name.endswith(".mat")])

        for j in range(vali_jpg_num):

            vali_num = j+1
            x_batch,y_batch = GD.get_data(vali_path, vali_num, 1)
            pred_dicts.update({x_: x_batch})
            y_y_ = sess.run(predictor, feed_dict=pred_dicts)
            cost_t = GOCF(y_y_, y_batch).output
            cost_tt=sess.run(cost_t)

            ori_array = np.array(y_batch)
            pre_array = np.array(y_y_)

            img_size_h = ori_array.shape[1]
            img_size_w = ori_array.shape[2]

            ori_img = ori_array.reshape((img_size_h, img_size_w))
            pre_img = pre_array.reshape((img_size_h, img_size_w))

            pre_img = (pre_img - np.min(pre_img)) / (np.max(pre_img) - np.min(pre_img))

            ori_img_show = Image.fromarray(np.uint8(ori_img * 255))
            pre_img_show = Image.fromarray(np.uint8(pre_img * 255))

            pre_name = vali_pre_save + "Vali_Pre_" + str(j + 1).zfill(6) + ".jpg"
            pre_img_show.save(pre_name)

            ori_img_show.show()
            pre_img_show.show()

            txt_vali = 'Vali_'+ str(vali_num) + ': ' + str(cost_tt) + '\n'
            listVali = open(vali_txt_result, 'a')
            listVali.write(txt_vali)
            listVali.close()