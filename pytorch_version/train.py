# -*- coding: utf-8 -*-
"""
PyTorch 버전의 NPE-TSDNet 전체 모델 학습(Fine-tuning) 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from pytorch_version.models import NPE_TSDNet
from pytorch_version.dataset import NucDataset

def train_full_net():
    """사전 학습된 가중치를 불러와 전체 네트워크를 fine-tuning한다."""
    # --- 1. 하이퍼파라미터 및 경로 설정 ---
    epochs = 50
    batch_size = 16
    learning_rate = 0.0001 # Fine-tuning 시에는 더 작은 학습률을 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "./dataset/trainset"
    model_save_dir = "./pytorch_version/saved_models/"
    g_model_path = os.path.join(model_save_dir, "g_model.pth")
    o_model_path = os.path.join(model_save_dir, "o_model.pth")
    final_model_path = os.path.join(model_save_dir, "npe_tsdnet_model.pth")

    # --- 2. 데이터셋 및 데이터로더 준비 ---
    print("Loading full model training data...")
    try:
        # 최종 학습에서는 원본(Ori) 이미지를 정답으로 사용
        train_dataset = NucDataset(data_path=train_path, mode='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 3. 모델 초기화 및 사전 학습 가중치 로드 ---
    model = NPE_TSDNet().to(device)

    # G-Net과 O-Net의 사전 학습된 가중치를 불러온다.
    if os.path.exists(g_model_path) and os.path.exists(o_model_path):
        print("Loading pre-trained weights for G-Net and O-Net...")
        # G_Net 클래스에서 저장한 state_dict를 불러옴
        g_state_dict = torch.load(g_model_path, map_location=device)
        # NPE_TSDNet의 g_net_features 모듈에 주입
        model.g_net_features.load_state_dict(g_state_dict)

        o_state_dict = torch.load(o_model_path, map_location=device)
        model.o_net_features.load_state_dict(o_state_dict)
        print("Pre-trained weights loaded successfully.")
    else:
        print("Warning: Pre-trained models not found. Training from scratch.")

    # --- 4. 손실 함수, 옵티마이저, 스케줄러 초기화 ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 25 에포크에서 학습률을 1/10로 줄인다.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # --- 5. 학습 루프 ---
    print("Starting NPE-TSDNet fine-tuning...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.9f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Summary --- Loss: {epoch_loss:.9f}, LR: {scheduler.get_last_lr()[0]}")
        
        scheduler.step()

    # --- 6. 모델 저장 ---
    print("Finished training. Saving final model...")
    torch.save(model.state_dict(), final_model_path)
    print(f"NPE-TSDNet model saved to {final_model_path}")

if __name__ == '__main__':
    train_full_net()
