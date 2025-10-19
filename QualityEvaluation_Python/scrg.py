import numpy as np

def scrg(img_in, img_out, pw, lur, luc, rdr, rdc):
    """
    신호 대 잡음비(SCR, Signal-to-Clutter Ratio) 및 SCR 이득(Gain)을 계산한다.
    이 함수는 이미지 처리 전후에 특정 목표(target)가 주변 배경(clutter)으로부터
    얼마나 더 잘 식별되는지를 평가한다.

    Args:
        img_in (np.ndarray): 처리 전 원본 이미지.
        img_out (np.ndarray): 처리 후 이미지.
        pw (int): 목표 주변의 배경(clutter) 영역을 정의할 때 사용할 패딩 너비.
        lur (int): 목표 영역의 좌측 상단 행(row) 좌표 (0-based).
        luc (int): 목표 영역의 좌측 상단 열(column) 좌표 (0-based).
        rdr (int): 목표 영역의 우측 하단 행(row) 좌표 (0-based).
        rdc (int): 목표 영역의 우측 하단 열(column) 좌표 (0-based).

    Returns:
        tuple: (s, sg)
            s (float): 처리 후 이미지의 SCR 값.
            sg (float): SCR 이득 (처리 후 SCR / 처리 전 SCR).
    """
    # MATLAB은 1-based 인덱싱, Python은 0-based 인덱싱을 사용한다.
    # MATLAB의 `lur:rdr`은 Python에서 `lur:rdr+1`에 해당한다.
    # 좌표가 0-based로 주어진다고 가정하고 변환한다.

    # --- 목표(Target) 영역 추출 ---
    img_in_target = img_in[lur:rdr + 1, luc:rdc + 1]
    img_out_target = img_out[lur:rdr + 1, luc:rdc + 1]

    # --- 배경(Clutter) 영역 추출 ---
    # MATLAB 코드의 영역 정의를 Python 인덱싱에 맞게 변환한다.
    # 상단 배경 영역
    img_in_padding_a = img_in[lur - pw:lur, luc - pw:rdc + 1]
    # 우측 배경 영역
    img_in_padding_b = img_in[lur - pw:rdr + 1, rdc + 1:rdc + 1 + pw]
    # 하단 배경 영역
    img_in_padding_c = img_in[rdr + 1:rdr + 1 + pw, luc:rdc + 1 + pw]
    # 좌측 배경 영역
    img_in_padding_d = img_in[lur:rdr + 1 + pw, luc - pw:luc]
    
    # 추출된 모든 배경 영역의 픽셀들을 하나의 벡터로 결합한다.
    img_in_padding = np.concatenate([
        img_in_padding_a.flatten(),
        img_in_padding_b.flatten(),
        img_in_padding_c.flatten(),
        img_in_padding_d.flatten()
    ])

    # 처리 후 이미지에 대해서도 동일하게 배경 영역을 추출한다.
    img_out_padding_a = img_out[lur - pw:lur, luc - pw:rdc + 1]
    img_out_padding_b = img_out[lur - pw:rdr + 1, rdc + 1:rdc + 1 + pw]
    img_out_padding_c = img_out[rdr + 1:rdr + 1 + pw, luc:rdc + 1 + pw]
    img_out_padding_d = img_out[lur:rdr + 1 + pw, luc - pw:luc]
    
    img_out_padding = np.concatenate([
        img_out_padding_a.flatten(),
        img_out_padding_b.flatten(),
        img_out_padding_c.flatten(),
        img_out_padding_d.flatten()
    ])

    # --- SCR 및 SCR Gain 계산 ---
    # 모든 데이터 타입을 float64로 변환하여 계산 정확도를 높인다.
    img_in_target = img_in_target.flatten().astype(np.float64)
    img_out_target = img_out_target.flatten().astype(np.float64)
    img_in_padding = img_in_padding.astype(np.float64)
    img_out_padding = img_out_padding.astype(np.float64)

    # SCR 공식: (목표 평균 - 배경 평균) / 배경 표준편차
    # np.std는 기본적으로 ddof=0 (모집단 표준편차)를 사용. MATLAB의 std는 ddof=1 (표본 표준편차)이 기본.
    # 결과의 일관성을 위해 ddof=1로 설정.
    std_in_padding = np.std(img_in_padding, ddof=1)
    std_out_padding = np.std(img_out_padding, ddof=1)

    # 0으로 나누는 것을 방지
    if std_in_padding == 0 or std_out_padding == 0:
        return 0.0, 1.0

    # 처리 전 이미지의 SCR을 계산한다.
    scr_in = (np.mean(img_in_target) - np.mean(img_in_padding)) / std_in_padding
    # 처리 후 이미지의 SCR을 계산한다.
    scr_out = (np.mean(img_out_target) - np.mean(img_out_padding)) / std_out_padding
    
    # 0으로 나누는 것을 방지
    if scr_in == 0:
        sg = 1.0 if scr_out == scr_in else float('inf')
    else:
        sg = scr_out / scr_in
        
    s = scr_out

    return s, sg
