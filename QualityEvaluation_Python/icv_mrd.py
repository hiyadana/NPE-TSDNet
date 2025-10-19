import numpy as np
from PIL import Image

def calculate_icv_mrd(rec_path, nuf_path):
    """
    비균일성 보정(NUC) 성능을 ICV와 MRD 지표로 평가한다.

    1. ICV (Inverse Coefficient of Variation): 평탄한 영역의 균일성 측정.
       값이 높을수록 노이즈 억제 성능이 우수함을 의미한다.
    2. MRD (Mean Relative Deviation): 디테일 영역의 상대적 오차 측정.
       값이 낮을수록 디테일 보존 성능이 우수함을 의미한다.

    참고: 이 함수는 'crop_1.png' 이미지에 특화된 평탄 영역 좌표를 포함하고 있다.
          다른 이미지에 적용하려면 좌표를 수정해야 한다.

    Args:
        rec_path (str): 보정 후 이미지 경로.
        nuf_path (str): 보정 전 원본 비균일 이미지 경로.

    Returns:
        tuple: (ICV, MRD, ZH)
            ICV (float): 계산된 ICV 값.
            MRD (float): 계산된 MRD 값.
            ZH (float): 최종 종합 점수 (ICV / MRD).
    """
    try:
        # --- 이미지 불러오기 및 전처리 ---
        rec_img = Image.open(rec_path).convert('L') # 흑백으로 변환
        nuf_img = Image.open(nuf_path).convert('L') # 흑백으로 변환
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. {e}")
        return None, None, None

    rec = np.array(rec_img, dtype=np.float64)
    nuf = np.array(nuf_img, dtype=np.float64)

    # 두 이미지의 크기가 다를 경우, rec 이미지의 크기를 nuf 이미지에 맞게 조절.
    if rec.shape != nuf.shape:
        h, w = nuf.shape
        rec_img_resized = Image.fromarray(rec).resize((w, h), Image.BILINEAR)
        rec = np.array(rec_img_resized, dtype=np.float64)

    # --- ICV (Inverse Coefficient of Variation) 계산 ---
    # crop_1.png 이미지의 평탄한 영역들을 수동으로 지정 (ROI: Region of Interest).
    # Python 인덱싱에 맞게 조정 (끝점은 +1 해줄 필요 없음, 슬라이싱의 특징)
    jy_1 = rec[0:50, 0:155]
    jy_2 = rec[0:85, 235:310] # 236 -> 235
    jy_3 = rec[0:30, 365:480] # 366 -> 365
    jy_4 = rec[90:200, 0:155] # 91 -> 90
    jy_5 = rec[115:200, 215:400] # 116 -> 115, 216 -> 215
    jy_6 = rec[300:400, 0:480] # 301 -> 300

    # 모든 평탄 영역의 픽셀들을 하나의 벡터로 결합.
    jy = np.concatenate([
        jy_1.flatten(), jy_2.flatten(), jy_3.flatten(),
        jy_4.flatten(), jy_5.flatten(), jy_6.flatten()
    ])

    # 평탄 영역의 평균과 표준편차 계산.
    mean_jy = np.mean(jy)
    std_jy = np.std(jy, ddof=1) # MATLAB의 std는 표본 표준편차(ddof=1)가 기본.

    # 0으로 나누는 것을 방지.
    icv = mean_jy / std_jy if std_jy != 0 else 0.0

    # --- MRD (Mean Relative Deviation) 계산 ---
    # 계산에 사용된 평탄 영역을 마스킹하기 위해 복사본 생성.
    rec_masked = np.copy(rec)
    rec_masked[0:50, 0:155] = -1000
    rec_masked[0:85, 235:310] = -1000
    rec_masked[0:30, 365:480] = -1000
    rec_masked[90:200, 0:155] = -1000
    rec_masked[115:200, 215:400] = -1000
    rec_masked[300:400, 0:480] = -1000

    # 마스킹되지 않은 영역(디테일 영역)의 인덱스를 찾는다.
    mask = rec_masked != -1000

    # 디테일 영역에서만 두 이미지 간의 차이를 계산.
    r_n = np.abs(rec - nuf)

    # 디테일 영역의 원본 이미지 픽셀 값.
    nuf_detail = nuf[mask]
    # 0으로 나누는 것을 방지하기 위해 0이 아닌 픽셀만 필터링.
    non_zero_mask = nuf_detail != 0
    
    # 상대적 오차를 계산.
    mm = r_n[mask][non_zero_mask] / nuf_detail[non_zero_mask]

    # 상대적 오차의 평균을 계산하여 MRD 값을 구한다.
    mrd = np.mean(mm)

    # --- 최종 종합 점수 계산 ---
    # MRD가 0인 경우를 방지.
    zh = icv / mrd if mrd != 0 else float('inf')

    return icv, mrd, zh

if __name__ == '__main__':
    # 사용 예시
    # MATLAB 스크립트와 동일한 로직을 수행하기 위해 rec_path와 nuf_path를 같게 설정.
    # 실제 보정 성능을 평가하려면 rec_path에 보정된 이미지 경로를 입력해야 한다.
    rec_image_path = '../dataset/RealIR/crop_1.png'
    nuf_image_path = '../dataset/RealIR/crop_1.png'
    
    # rec_image_path = '../dataset/RealIRresults/Test_Pre_crop_1.png'

    ICV, MRD, ZH = calculate_icv_mrd(rec_image_path, nuf_image_path)

    if ICV is not None:
        print(f"ICV: {ICV}")
        print(f"MRD: {MRD}")
        print(f"ZH (ICV/MRD): {ZH}")
