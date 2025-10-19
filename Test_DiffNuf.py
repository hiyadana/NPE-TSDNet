import tensorflow as tf
import os
import numpy as np
import scipy.io as scio
import cv2

"""
NPE-TSDNet 최종 모델 테스트 스크립트

이 스크립트는 'train.py'를 통해 학습된 최종 모델('g_o_model.ckpt')을 불러와
테스트 데이터셋에 적용하고, 보정된 결과 이미지를 저장한다.
"""

# --- 경로 설정 ---
# 저장된 모델이 위치한 경로
model_saving_path = '.\\model\\'
# 모델의 구조를 담고 있는 meta 파일 경로
meta_file = '.\\model\\g_o_model.ckpt.meta'

# --- TensorFlow 및 환경 설정 ---
# TensorFlow 1.x 스타일의 그래프 기반 실행을 위해 Eager Execution을 비활성화한다.
tf.compat.v1.disable_eager_execution()
# 사용할 GPU 장치 번호를 설정한다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()

# --- 테스트 데이터 및 결과 저장 경로 설정 ---
test_path = '.\\dataset\\DiffNufTest\\HighNuf\\'
# 테스트할 .mat 파일들이 있는 경로
save_path = '.\\dataset\\DiffNufTest\\Results\\HighNuc\\Ours\\'
# 보정된 이미지를 저장할 경로
test_files = os.listdir(test_path)

# TensorFlow 세션 시작
with tf.compat.v1.Session() as sess:
    # --- 모델 불러오기 ---
    # 1. .meta 파일로부터 모델의 그래프 구조를 불러온다.
    new_saver = tf.compat.v1.train.import_meta_graph(meta_file)
    # 2. 체크포인트 파일로부터 가장 최근에 저장된 모델의 가중치를 불러온다.
    # 만약 모델 경로가 변경되었다면, 'checkpoint' 파일을 직접 수정해야 할 수 있다.
    new_saver.restore(sess, tf.train.latest_checkpoint(model_saving_path))
    
    # --- 그래프에서 필요한 텐서 가져오기 ---
    # 불러온 그래프에서 이름으로 텐서를 찾아온다. 이 이름들은 학습 시점에 정의된 이름과 일치해야 한다.
    new_x = tf.compat.v1.get_default_graph().get_tensor_by_name("x_:0")       # 입력 플레이스홀더
    new_y = tf.compat.v1.get_default_graph().get_tensor_by_name("y_:0")       # 정답 플레이스홀더 (여기서는 사용되지 않음)
    output = tf.compat.v1.get_default_graph().get_tensor_by_name("op:0")      # 최종 출력 텐서

    # --- 테스트 루프 ---
    print(f"Start testing on files in {test_path}...")
    for file in test_files:
        # .mat 파일만 처리
        if os.path.splitext(file)[1] == '.mat':
            test_name = os.path.splitext(file)[0]
            test_mat_path = os.path.join(test_path, file)
            
            # 1. 데이터 로드 및 전처리
            mat_data = scio.loadmat(test_mat_path)
            x_gray = mat_data['Nuf'] # 'Nuf' 키로 입력 이미지를 가져옴
            x_arr = np.asarray(x_gray)
            size_h, size_w = x_arr.shape
            
            # 타입을 float32로 바꾸고 0-1 범위로 정규화
            x_arr_norm = x_arr.astype('float32') / 255.0
            # 모델 입력에 맞게 (1, height, width, 1) 형태로 차원 변경
            x_batch = x_arr_norm.reshape((1, size_h, size_w, 1))
            
            # 2. 모델 예측 실행
            y_y_ = sess.run(output, feed_dict={new_x: x_batch})

            # 3. 결과 후처리
            pre_array = np.array(y_y_)
            # (1, height, width, 1) -> (height, width) 형태로 차원 축소
            pre_img = pre_array.reshape((pre_array.shape[1], pre_array.shape[2]))
            # 0-255 범위로 값 복원
            pre_img_denorm = pre_img * 255.0

            # 4. 결과 저장
            # .mat 파일로 저장
            scio.savemat(os.path.join(save_path, test_name + ".mat"), {'pre': pre_img_denorm})
            # .png 이미지 파일로 저장
            cv2.imwrite(os.path.join(save_path, test_name + ".png"), pre_img_denorm)
            print(f"Processed and saved: {test_name}.png")

    print("Testing finished!")