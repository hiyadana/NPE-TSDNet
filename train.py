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

    # === 모델 입력 및 출력 플레이스홀더 정의 ===
    # TensorFlow 2.x 이상 버전과의 호환성을 위해 v1 모드를 활성화할 수 있음.
    # tf.compat.v1.disable_eager_execution()
    x_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='x_') # 입력 이미지 플레이스홀더
    y_ = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='y_') # 목표 이미지(Ground Truth) 플레이스홀더

    # --- G-네트워크(Gain Correction Network) 정의 ---
    # G-네트워크는 입력 이미지의 Gain(이득) 불균일성을 보정하는 역할을 수행.
    inpt_g = x_
    # 8개의 다중 스케일 컨볼루션(Multi-Scale Convolution) 레이어로 구성됨.
    layer_g_1_conv = MS_conv(inpt_g, 1, ms_name="g_1")
    layer_g_2_conv = MS_conv(layer_g_1_conv.output, ms_name="g_2")
    layer_g_3_conv = MS_conv(layer_g_2_conv.output, ms_name="g_3")
    layer_g_4_conv = MS_conv(layer_g_3_conv.output, ms_name="g_4")
    layer_g_5_conv = MS_conv(layer_g_4_conv.output, ms_name="g_5")
    layer_g_6_conv = MS_conv(layer_g_5_conv.output, ms_name="g_6")
    layer_g_7_conv = MS_conv(layer_g_6_conv.output, ms_name="g_7")
    layer_g_8_conv = MS_conv(layer_g_7_conv.output, ms_name="g_8")

    # G-네트워크의 최종 출력 레이어. 1x1 컨볼루션을 통해 채널을 1로 축소.
    layer_g_conv = ConvLayer(layer_g_8_conv.output, filter_shape=[1, 1, 64, 1], strides=[1, 1, 1, 1], activation=None,
                               padding="SAME", cl_name="g_0")
    layer_g_output = OPG(layer_g_conv.output)
    # 입력 이미지와 G-네트워크의 출력을 결합하여 Gain 보정이 적용된 이미지를 생성.
    g_out = GOP(inpt_g, layer_g_output.output)

    # --- O-네트워크(Offset Correction Network) 정의 ---
    # O-네트워크는 G-네트워크를 통과한 이미지의 Offset(오프셋) 불균일성을 보정.
    # 15개의 다중 스케일 컨볼루션 레이어로 구성됨.
    layer_o_1_conv = MS_conv(g_out.output, 1, ms_name="o_1")
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
    
    # O-네트워크의 최종 출력 레이어. 1x1 컨볼루션을 통해 채널을 1로 축소.
    layer_o_conv = ConvLayer(layer_o_15_conv.output, filter_shape=[1, 1, 64, 1], strides=[1, 1, 1, 1], activation=None,
                             padding="SAME", cl_name="o_0")
    layer_o_output = OPO(layer_o_conv.output)
    # G-네트워크의 출력과 O-네트워크의 출력을 결합하여 최종 보정 이미지를 생성.
    o_out = OOP(g_out.output, layer_o_output.output)

    # --- 손실 함수 및 최적화 정의 ---
    # 최종 출력과 목표 이미지 간의 손실(cost)을 계산.
    cost_cal = GOCF(o_out.output, y_)
    
    # G-네트워크와 O-네트워크의 모든 학습 가능한 파라미터를 수집.
    params_g = layer_g_1_conv.params + layer_g_2_conv.params + layer_g_3_conv.params + layer_g_4_conv.params + \
               layer_g_5_conv.params + layer_g_6_conv.params + layer_g_7_conv.params + layer_g_8_conv.params + \
               layer_g_conv.params
    params_o = layer_o_1_conv.params + layer_o_2_conv.params + layer_o_3_conv.params + layer_o_4_conv.params + \
               layer_o_5_conv.params + layer_o_6_conv.params + layer_o_7_conv.params + layer_o_8_conv.params + \
               layer_o_9_conv.params + layer_o_10_conv.params + layer_o_11_conv.params + layer_o_12_conv.params + \
               layer_o_13_conv.params + layer_o_14_conv.params + layer_o_15_conv.params + layer_o_conv.params
    params = params_g + params_o

    train_dicts = o_out.train_dicts  # 학습 시 feed_dict에 사용될 딕셔너리
    pred_dicts = o_out.pred_dicts    # 예측 시 feed_dict에 사용될 딕셔너리
    cost = cost_cal.output           # 손실 값 텐서
    predictor = o_out.output         # 네트워크의 최종 예측 출력 텐서

    # 학습률 스케줄링 및 옵티마이저 정의
    num_epoch = tf.Variable(0, name='epoch', trainable=False) # 현재 에포크 수를 추적하는 변수
    assign_op = tf.assign_add(num_epoch, 1)
    boundaries = [25] # 학습률을 변경할 에포크 지점
    learning_rates = [0.0001, 0.00001] # 각 구간에 적용될 학습률
    with tf.control_dependencies([assign_op]):
        # 에포크에 따라 학습률을 단계적으로 감소시킴 (Piecewise Constant Learning Rate)
        learning_rate = tf.train.piecewise_constant(x=num_epoch, boundaries=boundaries, values=learning_rates)
        # Adam 옵티마이저를 사용하여 손실을 최소화하도록 학습 연산을 정의.
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=params)

    # --- 경로 및 하이퍼파라미터 설정 ---
    # 데이터셋 및 결과 저장 경로
    train_path = "./dataset/trainset"      # 학습 데이터셋 위치
    vali_path = "./dataset/valiset"        # 검증 데이터셋 위치
    vali_pre_save = "./dataset/goValiResults/" # 검증 결과 이미지 저장 위치
    vali_txt_result = './Vali_cost.txt'      # 검증 손실 값 로그 파일
    
    # 모델 저장 경로
    model_path_name = "./Nmodel/g_o_model.ckpt" # 전체 fine-tuning된 모델 저장 경로
    model_g_path = './gc_train/model/' # 사전에 학습된 G-네트워크 모델 경로
    model_o_path = './oc_train/model/' # 사전에 학습된 O-네트워크 모델 경로

    init = tf.global_variables_initializer()  # 모든 변수 초기화

    # 각 서브네트워크의 파라미터를 로드하기 위한 Saver 객체 생성
    saver_g = tf.train.Saver(params_g)
    saver_o = tf.train.Saver(params_o)
    # 전체 모델을 저장하기 위한 Saver 객체 생성
    saver = tf.train.Saver()

    # 학습 하이퍼파라미터
    training_epochs = 50  # 전체 학습 에포크 수
    batch_size = 16       # 배치 크기
    display_step = 1      # 결과 표시 간격

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 사용할 GPU 장치 번호 설정
    config = tf.ConfigProto() # GPU 사용량 설정을 위한 ConfigProto 객체

    train_jpg_num = 300000 # 전체 학습 이미지 수
    train_batch_num = int(train_jpg_num / batch_size) # 에포크 당 배치 수

    # 학습 과정의 손실 값을 기록하기 위한 numpy 배열
    ffff = np.empty(shape=(training_epochs * train_batch_num, 3))

    # --- 모델 학습 및 검증 ---
    print("Start to train...")
    with tf.Session() as sess:
        sess.run(init)
        # 사전에 학습된 G-네트워크와 O-네트워크의 가중치를 불러옴.
        # 이는 fine-tuning을 위해 사용됨.
        saver_g.restore(sess, tf.train.latest_checkpoint(model_g_path))
        saver_o.restore(sess, tf.train.latest_checkpoint(model_o_path))
        
        FFF = 0 # 손실 기록 배열의 인덱스
        for epoch in range(training_epochs):
            cost_sum = 0.0
            avg_cost = 0.0
            training_batch = 0
            train_batch_ed = 1 # 현재 에포크 내 진행된 배치 수

            for i in range(train_batch_num):
                training_batch += 1
                # 각 배치에서 사용할 첫 번째 이미지의 인덱스 계산.
                # 200000개 이후의 데이터를 사용하는 이유는, 이전 데이터는 서브네트워크 사전 학습에 사용되었기 때문.
                batch_1_img = i * batch_size + 200001   

                # 배치 데이터 로드
                x_batch, y_batch = GD.get_data(train_path, batch_1_img, batch_size)
                train_dicts.update({x_: x_batch, y_: y_batch})

                # 학습 연산 실행
                sess.run(train_op, feed_dict=train_dicts)
                # 현재 배치의 손실 값 계산
                avg_cost = sess.run(cost, feed_dict=train_dicts)
                
                # 학습 진행 상황 출력
                train_progress = f"Training epoch: {epoch+1}.\n" \
                                 f"Remaining training epoch: {training_epochs-epoch-1}.\n" \
                                 f"Training progress of the epoch: {train_batch_ed}.\n" \
                                 f"Remaining training progress of the epoch: {train_batch_num - train_batch_ed}.\n" \
                                 f"avg_cost:{avg_cost}.\n"
                print("\r" + train_progress, end='')
                
                # 에포크, 배치 번호, 손실 값을 배열에 저장
                ffff[FFF] = [epoch + 1, train_batch_ed, avg_cost]
                FFF += 1
                train_batch_ed += 1

        # 학습이 완료된 전체 모델을 저장
        saver.save(sess, model_path_name)
        print("Finished!")

        # --- 검증 시작 ---
        x_dirs = os.listdir(vali_path)
        vali_jpg_num = len([name for name in x_dirs if name.endswith(".mat")])

        for j in range(vali_jpg_num):
            vali_num = j + 1
            # 검증 데이터 로드 (배치 크기는 1)
            x_batch, y_batch = GD.get_data(vali_path, vali_num, 1)
            pred_dicts.update({x_: x_batch})
            
            # 모델 예측 실행
            y_y_ = sess.run(predictor, feed_dict=pred_dicts)
            # 예측 결과에 대한 손실 값 계산
            cost_t = GOCF(y_y_, y_batch).output
            cost_tt = sess.run(cost_t)

            # 결과를 이미지로 변환하고 저장
            ori_array = np.array(y_batch)
            pre_array = np.array(y_y_)

            img_size_h = ori_array.shape[1]
            img_size_w = ori_array.shape[2]

            ori_img = ori_array.reshape((img_size_h, img_size_w))
            pre_img = pre_array.reshape((img_size_h, img_size_w))

            # 예측 이미지를 0-1 범위로 정규화
            pre_img = (pre_img - np.min(pre_img)) / (np.max(pre_img) - np.min(pre_img))

            # 이미지를 uint8 형태로 변환하여 저장 및 표시
            ori_img_show = Image.fromarray(np.uint8(ori_img * 255))
            pre_img_show = Image.fromarray(np.uint8(pre_img * 255))

            pre_name = vali_pre_save + "Vali_Pre_" + str(j + 1).zfill(6) + ".jpg"
            pre_img_show.save(pre_name)

            # 원본 및 예측 이미지 표시 (주석 처리 가능)
            # ori_img_show.show()
            # pre_img_show.show()

            # 검증 손실 값을 텍스트 파일에 기록
            txt_vali = f'Vali_{vali_num}: {cost_tt}\n'
            with open(vali_txt_result, 'a') as listVali:
                listVali.write(txt_vali)
