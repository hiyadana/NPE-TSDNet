import cv2
import random
import numpy as np
import scipy.io as scio
import os

"""
고정된 수준의 비균일 노이즈를 가진 테스트 데이터셋 생성 스크립트

이 스크립트는 원본 테스트 이미지에 특정 강도의 비균일 노이즈를 적용하여
'High', 'Medium', 'Low' 등 다양한 난이도의 테스트셋을 생성하는 데 사용된다.

생성 과정:
1. './DiffNufTest/Ori/' 폴더에서 원본 이미지를 불러온다.
2. 이미지 전체에 걸쳐 열(column) 단위로 고정된 표준편차를 갖는 이득/오프셋 노이즈를 적용한다.
3. .mat 파일에는 원본 이미지('Ori')와 노이즈 적용 이미지('Nuf')만을 저장한다.
4. 시각적 확인을 위해 노이즈 적용 이미지를 .png 파일로도 저장한다.
"""

# --- 경로 및 노이즈 수준 설정 ---
# 생성된 테스트셋을 저장할 경로. 주석을 참고하여 노이즈 수준에 맞는 폴더를 선택.
mat_savepath = ".\\DiffNufTest\\HighNuf\"  # 또는 "./DiffNufTest/MediumNuf/", "./DiffNufTest/LowNuf/"

# 원본 테스트 이미지가 있는 경로
img_path = ".\\DiffNufTest\\Ori\" 
img_list = os.listdir(img_path)
jpg_num = len(img_list)

# --- 노이즈 파라미터 설정 ---
# 이 값들을 조절하여 다양한 노이즈 수준의 테스트셋을 생성할 수 있다.
mu_g = 1.0  # 이득 노이즈의 평균
mu_o = 0.0  # 오프셋 노이즈의 평균

# HighNuf 설정
sigma_g = 0.15  # 이득 노이즈 표준편차 (MediumNuf: 0.10, LowNuf: 0.05)
sigma_o = 25    # 오프셋 노이즈 표준편차 (MediumNuf: 15, LowNuf: 5)

print(f"Starting test data generation for: {os.path.basename(os.path.normpath(mat_savepath))}")

# --- 데이터 생성 루프 ---
for index, img_name in enumerate(img_list):
    img_file = os.path.join(img_path, img_name)
    ori_img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    if ori_img is None:
        print(f"Warning: Could not read image {img_file}. Skipping.")
        continue
        
    size_h,size_w = ori_img.shape

    ori_arr = np.asarray(ori_img) # 원본 이미지
    nuf = np.asarray(ori_img, dtype='f4') # 노이즈를 적용할 이미지 복사본

    # --- 열 단위 노이즈 적용 ---
    # 이미지의 모든 열에 대해 다른 노이즈 값을 적용
    for i in range(size_w):
        # 현재 열에 적용할 이득 및 오프셋 노이즈 생성
        g_ori_i = round(random.uniform(mu_g - sigma_g, mu_g + sigma_g), 4)
        o_ori_i = round(random.gauss(mu_o, sigma_o), 2)

        # 노이즈 적용: Noisy_Img = g * Ori_Img + o
        nuf[:, i] = g_ori_i * nuf[:, i] + o_ori_i

    # --- .mat 및 .png 파일 저장 ---
    file_basename = os.path.splitext(img_name)[0]
    
    # .mat 파일 저장: 원본('Ori')과 노이즈 적용 이미지('Nuf')를 포함
    mat_dir = os.path.join(mat_savepath, file_basename + ".mat")
    scio.savemat(mat_dir, {'Ori': ori_arr, 'Nuf': nuf})

    # .png 파일 저장: 시각적 확인을 위해 노이즈 적용 이미지를 저장
    nuf_dir = os.path.join(mat_savepath, file_basename + ".png")
    cv2.imwrite(nuf_dir,nuf)

    print(f"Generated: {file_basename}.mat/.png ({index + 1}/{jpg_num})")

print("Finished generating test data.")
