# -*- coding: utf-8 -*-
"""
데이터 획득 모듈

이 스크립트는 학습 및 검증에 필요한 이미지 데이터를 불러오는 기능을 담당한다.
.mat 형식의 파일에서 입력 이미지와 정답 이미지를 읽어와 배치 형태로 제공한다.
"""

from PIL import Image
import os
import numpy as np
import scipy.io as scio

class GetData(object):
    """
    데이터 로딩을 위한 클래스.
    """
    @staticmethod
    def get_data(mat_path, batch_1_num, batch_size):
        """
        지정된 경로에서 .mat 파일들을 배치 단위로 불러온다.

        Args:
            mat_path (str): .mat 파일들이 저장된 디렉토리 경로.
            batch_1_num (int): 배치에서 시작할 파일의 번호.
            batch_size (int): 한 배치에 포함될 파일의 개수.

        Returns:
            tuple: (x_batch, y_batch)
                - x_batch (numpy.ndarray): (batch_size, height, width, 1) 형태의 입력 이미지 데이터.
                - y_batch (numpy.ndarray): (batch_size, height, width, 1) 형태의 정답 이미지 데이터.
        """

        for i in range(batch_size):
            # 파일 번호를 6자리 문자열(예: 000001)로 포맷팅하여 파일명을 생성.
            mat_name_num = batch_1_num + i
            mat_name = str(mat_name_num).zfill(6) + ".mat"
            mat_fpath = os.path.join(mat_path, mat_name)
            
            # scipy.io.loadmat을 사용하여 .mat 파일 로드.
            mat_data = scio.loadmat(mat_fpath)

            # .mat 파일 내에서 'Nuf' 키는 입력 이미지(Non-uniform), 'Ori' 키는 정답 이미지(Original)를 의미.
            data_img = mat_data['Nuf']
            label_img = mat_data['Ori']

            # 첫 번째 루프에서 이미지 크기를 기반으로 전체 배치를 담을 numpy 배열을 초기화.
            if i == 0:
                size_h = data_img.shape[0]
                size_w = data_img.shape[1]
                data_arr = np.empty(shape=(batch_size, size_h, size_w))
                ori_arr = np.empty(shape=(batch_size, size_h, size_w))

            # 이미지 데이터를 numpy 배열로 변환.
            data_img_array = np.asarray(data_img)
            ori_img_array = np.asarray(label_img)

            # 데이터 타입을 float32로 변환.
            data_img_array = data_img_array.astype('float32')
            ori_img_array = ori_img_array.astype('float32')

            # 픽셀 값을 0~1 범위로 정규화.
            data_img_arr = data_img_array / 255.0
            ori_img_arr = ori_img_array / 255.0

            # 정규화된 이미지 데이터를 배치 배열에 추가.
            data_arr[i] = data_img_arr
            ori_arr[i] = ori_img_arr

        # TensorFlow의 Conv2D 레이어 입력 형식에 맞게 (batch, height, width, channels)로 차원 변경.
        x_batch = data_arr.reshape((batch_size, size_h, size_w, 1))
        y_batch = ori_arr.reshape((batch_size, size_h, size_w, 1))

        return x_batch, y_batch
