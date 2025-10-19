# -*- coding: utf-8 -*-
"""
O-네트워크 사전 학습(Pre-training) 스크립트

이 스크립트는 전체 NPE-TSDNet 모델의 일부인 O-네트워크(Offset Correction Network)를
사전 학습하기 위해 사용된다. 학습된 모델은 나중에 전체 모델의 fine-tuning 시 초기 가중치로 사용된다.
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from o_getdata import GetData as GD
from conv_layer import ConvLayer
from multi_scale_conv import MS_conv
from output_o import OutPut_O as OPO
from o_cost import CostFunction as OCF

if __name__ == "__main__":

    # === 모델 입력 및 출력 플레이스홀더 정의 (TensorFlow 1.x 스타일) ===
    # tf.compat.v1.disable_eager_execution() # TF2.x에서 TF1.x 코드를 실행하기 위해 필요할 수 있음
    x_o = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1], name="x_o") # 입력 이미지
    y_o = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1], name="y_o") # 목표 오프셋 보정 계수(Ground Truth Offset)

    # --- O-네트워크(Offset Correction Network) 정의 ---
    # train.py와 동일한 15계층의 O-네트워크 아키텍처를 사용한다.
    inpt_o = x_o
    layer_o_1_conv = MS_conv(inpt_o, 1, ms_name="o_1")
    layer_o_2_conv = MS_conv(layer_o_1_conv.output, ms_name="o_2")
    layer_o_3_conv = MS_conv(layer_o_2_conv.output, ms_name="o_3")
    layer_o_4_conv = MS_conv(layer_o_3_conv.output, ms_name="o_4")
    layer_o_5_conv = MS_conv(layer_o_4_conv.output, ms_name="o_5")
    layer_o_6_conv = MS_conv(layer_o_5_conv.output, ms_name="o_6")
    layer_o_7_conv = MS_conv(layer_o_6_conv.output, ms_name="o_7")
    layer_o_8_conv = MS_conv(layer_o_7_conv.output, ms_name="o_8")
    layer_o_9_conv = MS_conv(layer_o_8_conv.output, ms_name="o_9")
    layer_o_10_conv = MS_conv(layer_o_9_conv.output, ms_name="o_10")
    layer_o_11_conv = MS_conv(layer_o_10_conv.output, ms_name="o_11")
    layer_o_12_conv = MS_conv(layer_o_11_conv.output, ms_name="o_12")
    layer_o_13_conv = MS_conv(layer_o_12_conv.output, ms_name="o_13")
    layer_o_14_conv = MS_conv(layer_o_13_conv.output, ms_name="o_14")
    layer_o_15_conv = MS_conv(layer_o_14_conv.output, ms_name="o_15")

    # O-네트워크의 최종 출력 레이어.
    layer_o_conv = ConvLayer(layer_o_15_conv.output, filter_shape=[1, 1, 64, 1], strides=[1, 1, 1, 1], activation=None,
                               padding="SAME", cl_name="o_0")
    # 최종 특징 맵에서 열별 평균을 계산하여 오프셋 보정 계수를 추정.
    layer_o_output = OPO(layer_o_conv.output)

    # --- 손실 함수 및 최적화 정의 ---
    # 추정된 오프셋 보정 계수와 실제 오프셋 보정 계수(y_o) 간의 손실을 계산.
    cost_cal = OCF(layer_o_output.output, y_o)
    
    # O-네트워크의 모든 학습 가능한 파라미터를 수집.
    params_o = layer_o_1_conv.params + layer_o_2_conv.params + layer_o_3_conv.params + layer_o_4_conv.params + \
             layer_o_5_conv.params + layer_o_6_conv.params + layer_o_7_conv.params + layer_o_8_conv.params + \
               layer_o_9_conv.params + layer_o_10_conv.params + layer_o_11_conv.params + layer_o_12_conv.params + \
               layer_o_13_conv.params + layer_o_14_conv.params + layer_o_15_conv.params + layer_o_conv.params

    train_dicts = layer_o_output.train_dicts  # 학습 시 feed_dict
    pred_dicts = layer_o_output.pred_dicts    # 예측 시 feed_dict
    cost = cost_cal.output                    # 손실 값 텐서
    predictor = layer_o_output.output         # 네트워크의 예측 출력 (추정된 오프셋 보정 계수)

    # 학습률 스케줄링 및 Adam 옵티마이저 정의
    num_epoch = tf.Variable(0, name='epoch', trainable=False)
    assign_op = tf.compat.v1.assign_add(num_epoch, 1)
    boundaries = [25]
    learning_rates = [0.001, 0.0001]
    with tf.control_dependencies([assign_op]):
        learning_rate = tf.compat.v1.train.piecewise_constant(x=num_epoch, boundaries=boundaries, values=learning_rates)
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=params_o)

    # --- 경로 및 하이퍼파라미터 설정 ---
    init = tf.compat.v1.global_variables_initializer() # 변수 초기화
    saver = tf.compat.v1.train.Saver()                 # 모델 저장을 위한 Saver

    training_epochs = 50
    batch_size = 16
    display_step = 1

    train_path = "../dataset/trainset"      # 학습 데이터셋 위치
    vali_path = "../dataset/valiset"        # 검증 데이터셋 위치
    vali_pre_save = "../dataset/oValiResults/" # 검증 결과 저장 위치
    vali_txt_result = './Vali_cost.txt'      # 검증 손실 로그 파일
    
    # 사전 학습된 O-네트워크 모델 저장 경로
    model_path_name = "./model/o_model.ckpt"
    excel_path = "./"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    
    # 사전 학습에는 처음 200,000개의 이미지를 사용.
    train_jpg_num = 200000
    train_batch_num = int(train_jpg_num / batch_size)

    # 학습 손실 기록을 위한 numpy 배열 (3번에 한번씩 기록하므로 크기를 1/3로 설정)
    ffff = np.empty(shape=(training_epochs * train_batch_num // 3, 3))

    # --- 모델 학습 및 검증 ---
    print("Start to train O-Network...")
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        FFF = 0 # 손실 기록 배열 인덱스
        for epoch in range(training_epochs):
            train_batch_ed = 1
            for i in range(train_batch_num):
                batch_1_img = i * batch_size + 1

                # 배치 데이터 로드 (입력 이미지와 "정답 오프셋 계수")
                x_batch, y_batch = GD.get_data(train_path, batch_1_img, batch_size)
                train_dicts.update({x_o: x_batch, y_o: y_batch})

                # 학습 연산 실행
                sess.run(train_op, feed_dict=train_dicts)
                # 손실 값 계산
                avg_cost = sess.run(cost, feed_dict=train_dicts)
                
                # 학습 진행 상황 출력
                train_progress = f"Training epoch: {epoch+1}/{training_epochs}, Batch: {train_batch_ed}/{train_batch_num}, Cost: {avg_cost:.9f}"
                print("\r" + train_progress, end='')
                
                # 3 배치마다 손실 값을 기록
                if train_batch_ed % 3 == 0:
                    ffff[FFF] = [epoch + 1, train_batch_ed, avg_cost]
                    FFF += 1
                train_batch_ed += 1
            print() # 에포크 완료 후 줄바꿈

        # 학습된 O-네트워크 모델 저장
        saver.save(sess, model_path_name)
        print("O-Network training finished!")

        # 학습 과정의 손실 값을 Excel 파일로 저장
        ffff_data = pd.DataFrame(ffff)
        excel_result = excel_path + 'Cost_O' + '.xlsx'
        with pd.ExcelWriter(excel_result) as writer:
            ffff_data.to_excel(writer, sheet_name='page_1', float_format='%.5f')

        # --- 검증 시작 ---
        print("Start O-Network validation...")
        x_dirs = os.listdir(vali_path)
        vali_jpg_num = len([name for name in x_dirs if name.endswith(".mat")])

        with open(vali_txt_result, 'w') as listVali: # 파일을 쓰기 모드로 열어 초기화
            for j in range(vali_jpg_num):
                vali_num = j + 1
                x_batch, y_batch = GD.get_data(vali_path, vali_num, 1)
                pred_dicts.update({x_o: x_batch})
                
                # 예측 실행 (추정된 오프셋 계수 생성)
                y_y_ = sess.run(predictor, feed_dict=pred_dicts)
                # 손실 계산
                cost_t = OCF(y_y_, y_batch).output
                cost_tt = sess.run(cost_t)

                # 검증 손실 값을 텍스트 파일에 기록
                txt_text = f'Vali_{vali_num}: {cost_tt}\n'
                listVali.write(txt_text)

                # 정답 오프셋 계수와 예측된 오프셋 계수를 Excel 파일로 저장
                ori_array = np.array(y_batch)
                pre_array = np.array(y_y_)
                
                ori_2d = ori_array.squeeze() # 1인 차원 제거
                pre_2d = pre_array.squeeze()
                ori_data = pd.DataFrame(ori_2d)
                pre_data = pd.DataFrame(pre_2d)
                
                vali_excel_result = vali_excel_path + 'Vali_label_O_' + str(vali_num).zfill(6) + '.xlsx'
                with pd.ExcelWriter(vali_excel_result) as writer:
                    ori_data.to_excel(writer, sheet_name='label_offset', float_format='%.5f')
                    pre_data.to_excel(writer, sheet_name='predicted_offset', float_format='%.5f')
        print("Validation finished!")
