from glob import glob
import cv2
import random
import numpy as np
import scipy.io as scio

"""
합성 비균일(Non-Uniformity) 학습 데이터 생성 스크립트

이 스크립트는 깨끗한 원본 이미지(예: MS-COCO 데이터셋)를 사용하여
인공적인 비균일 노이즈(gain, offset)가 포함된 학습 데이터를 생성한다.

생성 과정:
1. 원본 이미지를 불러와 128x128 크기로 자른다.
2. 각 잘린 이미지의 열(column)마다 무작위적인 이득(gain) 및 오프셋(offset) 노이즈를 적용한다.
3. 원본 이미지, 노이즈 적용 이미지, 그리고 노이즈를 복원하기 위한 정답(ground truth) 
   이득/오프셋 보정 계수를 하나의 .mat 파일에 함께 저장한다.
"""

# --- 경로 및 파라미터 설정 ---
mat_savepath = "./traindata/"  # 생성된 .mat 파일을 저장할 경로
# 원본 이미지가 있는 경로. MS-COCO 데이터셋(http://cocodataset.org) 사용을 권장.
loc_list = glob("./MS-COCO/*.jpg")

jpg_num = len(loc_list)
ori_list = list(range(jpg_num))
random.shuffle(ori_list) # 이미지 사용 순서를 무작위로 섞음

train_num = 500000 # 생성할 총 학습 샘플 수
mat_list = list(range(1, train_num + 1))
random.shuffle(mat_list) # .mat 파일의 번호를 무작위로 섞음

mat_index = 0
print("Starting data generation...")

# --- 데이터 생성 루프 ---
# 모든 원본 이미지를 순회
for index in range(jpg_num):
    ori_num = ori_list[index]
    ms_img = cv2.imread(loc_list[ori_num])
    size_h, size_w, channel = ms_img.shape
    
    h_cai = 128  # 잘라낼 이미지의 높이
    w_cai = 128  # 잘라낼 이미지의 너비

    # 이미지를 128x128 크기의 패치로 자르기 (겹치지 않게)
    for right in range(w_cai, size_w, w_cai):
        for lower in range(h_cai, size_h, h_cai):
            if mat_index >= train_num:
                print("Finished generating all samples.")
                exit(0)

            mat_num = mat_list[mat_index]

            # 이미지 자르기 및 흑백 변환
            cropped_img = ms_img[lower - h_cai:lower, right - w_cai:right]
            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

            ori_arr = np.asarray(gray_img) # 원본 이미지 (정답 레이블)
            img = np.asarray(gray_img, dtype='f4') # 노이즈를 적용할 이미지 복사본

            G_list = [] # 정답 이득 보정 계수를 저장할 리스트
            O_list = [] # 정답 오프셋 보정 계수를 저장할 리스트

            # --- 열 단위 노이즈 생성 및 적용 ---
            # 노이즈 파라미터 랜덤화
            mu_g = 1.0  # 이득 노이즈의 평균
            mu_o = 0.0  # 오프셋 노이즈의 평균
            sigma_g = random.randint(0, 18) / 100.0  # 이득 노이즈의 표준편차 (0 ~ 0.18)
            sigma_o = random.randint(0, 30)          # 오프셋 노이즈의 표준편차 (0 ~ 30)
            
            # 이미지의 모든 열(column)에 대해 다른 노이즈를 적용
            for i in range(w_cai):
                # 현재 열에 적용할 이득(gain) 및 오프셋(offset) 노이즈 생성
                g_ori_i = round(random.uniform(mu_g - sigma_g, mu_g + sigma_g), 4)
                o_ori_i = round(random.gauss(mu_o, sigma_o), 2)

                # 노이즈 적용: Noisy_Img = g * Ori_Img + o
                img[:, i] = g_ori_i * img[:, i] + o_ori_i

                # --- 정답 보정 계수 계산 ---
                # 위 수식으로부터 Ori_Img = (1/g) * Noisy_Img - o/g
                # 따라서, 이득 보정 계수는 1/g, 오프셋 보정 계수는 -o/g 가 된다.
                G_i = round(1 / g_ori_i, 4)
                O_i = round(-o_ori_i / g_ori_i, 2)

                G_list.append(G_i)
                O_list.append(O_i)

            G_arr = np.asarray(G_list) # (128,) 형태의 벡터
            O_arr = np.asarray(O_list) # (128,) 형태의 벡터

            # --- .mat 파일 저장 ---
            # 생성된 데이터를 .mat 파일로 저장
            # Ori: 원본 이미지 (최종 학습용 정답)
            # Nuf: 노이즈 적용된 이미지 (입력)
            # G: 이득 보정 계수 (G-Net 사전 학습용 정답)
            # O: 오프셋 보정 계수 (O-Net 사전 학습용 정답)
            mat_filename = os.path.join(mat_savepath, str(mat_num).zfill(6) + ".mat")
            scio.savemat(mat_filename, {'Ori': ori_arr, 'Nuf': img, 'G': G_arr, 'O': O_arr})

            print(f"Generated: {mat_filename} ({mat_index + 1}/{train_num})")
            mat_index += 1
