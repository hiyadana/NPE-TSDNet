import numpy as np

def psnr(img1, img2, data_max):
    """
    두 이미지 간의 최대 신호 대 잡음비(PSNR)와 제곱 평균 오차 제곱근(RMSE)을 계산한다.

    Args:
        img1 (np.ndarray): 첫 번째 이미지. 원본 이미지에 해당.
        img2 (np.ndarray): 두 번째 이미지. 복원된 또는 왜곡된 이미지에 해당.
        data_max (float): 이미지의 최대 픽셀 값 (예: 8비트 이미지의 경우 255.0).

    Returns:
        tuple: (RMSE, PSNR) 두 이미지 간의 제곱 평균 오차 제곱근과 PSNR 값을 포함하는 튜플.
               PSNR 값의 단위는 데시벨(dB)이다.
    """
    # 정확한 계산을 위해 입력 이미지를 float 타입으로 변환.
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 두 이미지의 차이를 계산.
    diff = img1 - img2

    # 평균 제곱 오차(Mean Squared Error, MSE)를 계산.
    # np.mean() 함수는 배열의 모든 요소의 평균을 계산한다.
    mse = np.mean(diff ** 2)

    # 제곱 평균 오차 제곱근(Root Mean Squared Error, RMSE)을 계산.
    rmse = np.sqrt(mse)

    # PSNR(Peak Signal-to-Noise Ratio)을 계산.
    # PSNR (dB) = 20 * log10(MAX / RMSE)
    # mse가 0인 경우, 즉 두 이미지가 완벽히 동일한 경우 PSNR은 무한대가 되므로 0을 반환하지 않도록 처리.
    if mse == 0:
        return rmse, 100.0
    
    p = 20 * np.log10(data_max / rmse)

    return rmse, p
