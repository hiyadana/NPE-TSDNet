# -*- coding: utf-8 -*-
"""
PyTorch 버전의 NPE-TSDNet 최종 모델 테스트 스크립트
"""

import torch
import os
import numpy as np
import scipy.io as scio
import cv2
import glob

from pytorch_version.models import NPE_TSDNet

def test_net():
    """학습된 최종 모델을 불러와 테스트 데이터셋에 대한 추론을 수행한다."""
    # --- 1. 경로 및 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "./pytorch_version/saved_models/npe_tsdnet_model.pth"
    test_data_path = ".\\dataset\\DiffNufTest\\HighNuf\\" # 원본 TF 프로젝트의 경로와 동일
    save_path = ".\\dataset\\DiffNufTest\\Results\\HighNuc\\Ours_PyTorch\\" # PyTorch 결과 저장을 위한 새 폴더
    os.makedirs(save_path, exist_ok=True)

    # --- 2. 모델 불러오기 ---
    model = NPE_TSDNet().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 모델을 평가 모드로 설정
    print("NPE-TSDNet model loaded successfully.")

    # --- 3. 테스트 루프 ---
    test_files = glob.glob(os.path.join(test_data_path, '*.mat'))
    if not test_files:
        print(f"Error: No .mat files found in {test_data_path}")
        return

    print(f"Starting testing on {len(test_files)} files...")
    # torch.no_grad() 컨텍스트 내에서 추론을 수행하여 메모리 사용량과 계산 속도를 최적화한다.
    with torch.no_grad():
        for mat_file in test_files:
            base_name = os.path.basename(mat_file)
            test_name = os.path.splitext(base_name)[0]
            
            # 1. 데이터 로드 및 전처리
            mat_data = scio.loadmat(mat_file)
            input_img = mat_data['Nuf'].astype(np.float32)
            
            # Numpy -> Tensor 변환, 정규화, 차원 추가, 디바이스 이동
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)

            # 2. 모델 추론 (순전파)
            corrected_tensor = model(input_tensor)

            # 3. 결과 후처리
            # Tensor -> Numpy 변환 및 차원 축소, 0-255 범위로 복원
            corrected_img = corrected_tensor.squeeze().cpu().numpy() * 255.0
            # 픽셀 값이 0-255 범위를 벗어나지 않도록 클리핑
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

            # 4. 결과 저장
            save_mat_path = os.path.join(save_path, test_name + ".mat")
            save_png_path = os.path.join(save_path, test_name + ".png")
            
            # .mat 파일로 저장 (키: 'pre')
            scio.savemat(save_mat_path, {'pre': corrected_img})
            # .png 이미지 파일로 저장
            cv2.imwrite(save_png_path, corrected_img)
            
            print(f"Processed and saved: {base_name}")

    print("Testing finished!")

if __name__ == '__main__':
    test_net()
