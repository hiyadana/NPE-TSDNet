# -*- coding: utf-8 -*-
"""
PyTorch 버전의 O-네트워크 사전 학습 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from pytorch_version.models import O_Net
from pytorch_version.dataset import NucDataset

def train_o_net():
    """O-네트워크 사전 학습을 수행한다."""
    # --- 1. 하이퍼파라미터 및 경로 설정 ---
    epochs = 50
    batch_size = 16
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 원본 프로젝트의 경로 구조를 따른다.
    train_path = "./dataset/trainset"
    model_save_dir = "./pytorch_version/saved_models/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "o_model.pth")

    # --- 2. 데이터셋 및 데이터로더 준비 ---
    print("Loading O-Net pre-training data...")
    try:
        train_dataset = NucDataset(data_path=train_path, mode='pretrain_o')
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and data has been generated.")
        return

    # --- 3. 모델, 손실 함수, 옵티마이저 초기화 ---
    model = O_Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 25 에포크에서 학습률을 1/10로 줄인다.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # --- 4. 학습 루프 ---
    print("Starting O-Net pre-training...")
    for epoch in range(epochs):
        model.train() # 모델을 학습 모드로 설정
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사도 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels)

            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0: # 100 배치마다 로그 출력
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.9f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Summary --- Loss: {epoch_loss:.9f}, LR: {scheduler.get_last_lr()[0]}")
        
        # 스케줄러 업데이트
        scheduler.step()

    # --- 5. 모델 저장 ---
    print("Finished training. Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"O-Net model saved to {model_path}")

if __name__ == '__main__':
    train_o_net()
