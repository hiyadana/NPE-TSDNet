import numpy as np

def ln(img):
    """
    이미지의 저주파 비균일성(Low-frequency Non-uniformity) 지표를 계산한다.

    이 함수는 이미지 중앙 픽셀과 네 모서리 픽셀 간의 상대적 차이를 평균하여
    이미지 전체의 대략적인 균일성 수준을 평가한다.
    입력 이미지가 균일한 평면(flat field)이라고 가정하며, 그럴 경우 Ln 값은 0에 가깝다.

    Args:
        img (np.ndarray): 분석할 흑백 이미지. 2D 배열 형태.

    Returns:
        float: 계산된 저주파 비균일성 값.
    """
    # 이미지의 높이(h)와 너비(w)를 가져온다.
    h, w = img.shape

    # 정확한 계산을 위해 이미지를 float 타입으로 변환.
    img = img.astype(np.float64)

    # --- 중앙 및 모서리 픽셀 값 추출 ---
    # 이미지의 중앙 픽셀 값을 가져온다.
    # 인덱스는 0부터 시작하므로 h//2, w//2를 사용한다.
    lc = img[h // 2, w // 2]

    # 중앙 픽셀 값이 0인 경우 0으로 나누기 오류가 발생하므로 0.0을 반환.
    if lc == 0:
        return 0.0

    # 네 개의 모서리 부근(가장자리에서 8픽셀 안쪽)의 픽셀 값을 가져온다.
    # MATLAB과 달리 Python은 0-based 인덱싱을 사용하므로 인덱스에서 1을 뺀다.
    # (8, 8) -> (7, 7)
    # (8, w-8) -> (7, w-9)
    # (h-8, w-8) -> (h-9, w-9)
    # (h-8, 8) -> (h-9, 7)
    l1 = img[7, 7]
    l2 = img[7, w - 9]
    l3 = img[h - 9, w - 9]
    l4 = img[h - 9, 7]

    # --- 비균일성 지표 계산 ---
    # 중앙 픽셀 값 대비 각 모서리 픽셀 값의 상대적 차이의 절댓값을 구하고, 이들의 평균을 계산한다.
    # Ln = mean(abs(Lc - Li) / Lc) for i=1 to 4
    ln_val = (np.abs((lc - l1) / lc) + np.abs((lc - l2) / lc) + np.abs((lc - l3) / lc) + np.abs((lc - l4) / lc)) * (1 / 4)

    return ln_val
