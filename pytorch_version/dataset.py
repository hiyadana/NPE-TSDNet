# -*- coding: utf-8 -*-
"""
PyTorch 버전의 데이터셋 및 데이터 로더 정의

TensorFlow 프로젝트의 수동 데이터 로딩 방식을 PyTorch의 표준적인
`Dataset`과 `DataLoader`를 사용하는 방식으로 재구현한다.
"""

import os
import glob
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NucDataset(Dataset):
    """
    비균일성 보정(NUC)을 위한 PyTorch 커스텀 데이터셋.
    
    .mat 파일들로부터 데이터를 읽어와, 학습 모드에 따라 적절한
    입력(Nuf)과 정답(Ori, G, 또는 O)을 텐서 형태로 반환한다.
    """
    def __init__(self, data_path, mode='train'):
        """
        NucDataset 객체를 초기화한다.

        Args:
            data_path (str): .mat 파일들이 저장된 디렉토리 경로.
            mode (str, optional): 데이터셋의 모드. 
                'pretrain_g': G-Net 사전 학습용 (정답: G 벡터)
                'pretrain_o': O-Net 사전 학습용 (정답: O 벡터)
                'train': 최종 모델 학습용 (정답: Ori 이미지)
                기본값은 'train'.
        """
        super(NucDataset, self).__init__()
        if not os.path.isdir(data_path):
            raise ValueError(f"Provided data path does not exist: {data_path}")
        self.mat_files = glob.glob(os.path.join(data_path, '*.mat'))
        if not self.mat_files:
            raise ValueError(f"No .mat files found in {data_path}")
        self.mode = mode

    def __len__(self):
        """데이터셋의 총 샘플 수를 반환한다."""
        return len(self.mat_files)

    def __getitem__(self, index):
        """
        지정된 인덱스(index)에 해당하는 샘플 1개를 불러온다.
        """
        # .mat 파일 로드
        mat_path = self.mat_files[index]
        mat_data = scio.loadmat(mat_path)

        # 입력 이미지(Nuf) 로드 및 정규화
        input_img = mat_data['Nuf'].astype(np.float32) / 255.0
        # (H, W) -> (1, H, W) 형태로 채널 차원 추가
        input_tensor = torch.from_numpy(input_img).unsqueeze(0)

        # 학습 모드에 따라 정답(label) 데이터를 선택
        if self.mode == 'pretrain_g':
            label_data = mat_data['G'].astype(np.float32).squeeze()
            label_tensor = torch.from_numpy(label_data)
            # (W,) -> (1, 1, W) 형태로 차원 변경 (모델 출력과 일치)
            label_tensor = label_tensor.view(1, 1, -1)
        elif self.mode == 'pretrain_o':
            label_data = mat_data['O'].astype(np.float32).squeeze()
            label_tensor = torch.from_numpy(label_data)
            # (W,) -> (1, 1, W) 형태로 차원 변경
            label_tensor = label_tensor.view(1, 1, -1)
        elif self.mode == 'train':
            label_data = mat_data['Ori'].astype(np.float32) / 255.0
            # (H, W) -> (1, H, W) 형태로 채널 차원 추가
            label_tensor = torch.from_numpy(label_data).unsqueeze(0)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return input_tensor, label_tensor


if __name__ == '__main__':
    # 데이터셋 및 데이터로더 테스트 코드
    # 이 파일을 직접 실행하면 데이터 로딩 과정을 확인할 수 있다.
    
    # 실제 데이터 경로로 수정해야 함
    # 예: TRAIN_DATA_PATH = '../dataset/trainset'
    TRAIN_DATA_PATH = './dataset/trainset' # 경로를 실제 환경에 맞게 수정하세요.

    print(f"Testing NucDataset with path: {TRAIN_DATA_PATH}")
    
    try:
        # 최종 학습 모드 테스트
        print("\n--- Testing 'train' mode ---")
        train_dataset = NucDataset(data_path=TRAIN_DATA_PATH, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
        
        inputs, labels = next(iter(train_loader))
        print(f"Batch input shape: {inputs.shape}")   # 예상: [4, 1, 128, 128]
        print(f"Batch label shape: {labels.shape}")   # 예상: [4, 1, 128, 128]

        # G-Net 사전 학습 모드 테스트
        print("\n--- Testing 'pretrain_g' mode ---")
        pretrain_g_dataset = NucDataset(data_path=TRAIN_DATA_PATH, mode='pretrain_g')
        pretrain_g_loader = DataLoader(dataset=pretrain_g_dataset, batch_size=4, shuffle=True)
        
        inputs, labels = next(iter(pretrain_g_loader))
        print(f"Batch input shape: {inputs.shape}")   # 예상: [4, 1, 128, 128]
        print(f"Batch label shape: {labels.shape}")   # 예상: [4, 1, 1, 128]

    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure the `TRAIN_DATA_PATH` is set correctly and contains .mat files.")
