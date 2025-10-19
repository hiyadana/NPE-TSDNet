import numpy as np
from scipy.signal import convolve2d

def coarseness(graypic):
    """
    이미지의 조잡도(Coarseness) 또는 거칠기(Roughness)를 계산한다.
    이 지표는 이미지의 전체 밝기 대비 수평 및 수직 방향의 변화량 합계를 나타낸다.
    값이 높을수록 이미지가 더 거칠거나 노이즈가 많음을 의미한다.

    Args:
        graypic (np.ndarray): 분석할 흑백 이미지. 2D 배열 형태.

    Returns:
        float: 계산된 조잡도 값.
    """
    # --- 필터 정의 ---
    # 수평 방향의 그래디언트(인접 픽셀 간의 차이)를 계산하기 위한 필터.
    h_m = np.array([[1, -1]])
    # 수직 방향의 그래디언트를 계산하기 위한 필터.
    v_m = h_m.T  # h_m의 전치 행렬

    # 정확한 계산을 위해 입력 이미지를 float 타입으로 변환.
    graypic = graypic.astype(np.float64)

    # --- 수평 방향 변화량 계산 ---
    # convolve2d 함수를 사용하여 이미지와 수평 필터를 컨볼루션한다.
    # 'valid' 모드는 필터가 이미지 경계를 벗어나지 않는 영역만 계산하므로
    # MATLAB 코드에서 경계를 제거하는 것과 동일한 효과를 낸다.
    i_h = convolve2d(graypic, h_m, mode='valid')
    # 수평 방향 그래디언트의 절댓값 합(L1-norm)을 계산.
    i_h_l1 = np.sum(np.abs(i_h))

    # --- 수직 방향 변화량 계산 ---
    # 이미지와 수직 필터를 컨볼루션한다.
    i_v = convolve2d(graypic, v_m, mode='valid')
    # 수직 방향 그래디언트의 절댓값 합(L1-norm)을 계산.
    i_v_l1 = np.sum(np.abs(i_v))

    # --- 정규화 및 최종 계산 ---
    # 원본 이미지의 전체 픽셀 값 합(L1-norm)을 계산.
    i_l1 = np.sum(np.abs(graypic))
    
    # 0으로 나누는 것을 방지.
    if i_l1 == 0:
        return 0.0

    # 조잡도(IR) 계산: (수평 변화량 + 수직 변화량) / 전체 픽셀 값 합
    ir = (i_h_l1 + i_v_l1) / i_l1

    return ir
