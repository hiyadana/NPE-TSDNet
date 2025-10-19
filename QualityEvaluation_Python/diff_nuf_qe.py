import os
import glob
import numpy as np
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

# 이전에 변환한 품질 평가 함수들을 임포트합니다.
from psnr import psnr
from coarseness import coarseness
from ln import ln
from scrg import scrg

def xml2mat(label_path, file_name):
    """
    XML 파일에서 객체 경계 상자(bounding box) 좌표를 파싱한다.
    MATLAB의 사용자 정의 함수 'xml2mat'의 기능을 대체한다.

    Args:
        label_path (str): XML 파일이 포함된 디렉터리 경로.
        file_name (str): 확장자를 제외한 파일 이름.

    Returns:
        np.ndarray: [[xmin, ymin, xmax, ymax]] 형태의 좌표 배열.
                    객체를 찾지 못하면 None을 반환한다.
    """
    xml_file = os.path.join(label_path, file_name + '.xml')
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return None
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # MATLAB은 1-based, Python은 0-based 인덱싱을 사용하므로 -1을 해준다.
        return np.array([[xmin - 1, ymin - 1, xmax - 1, ymax - 1]])
    except (FileNotFoundError, ET.ParseError, AttributeError):
        return None

def evaluate_quality_in_batch(base_path, lmh_nuf, method):
    """
    지정된 폴더의 모든 보정 결과에 대해 품질 평가 지표를 일괄 계산한다.

    Args:
        base_path (str): 테스트 데이터셋의 기본 경로.
        lmh_nuf (str): 평가할 노이즈 수준 ('Low', 'Medium', 'High').
        method (str): 평가할 보정 방법의 이름.
    """
    # --- 경로 설정 ---
    ori_path = os.path.join(base_path, lmh_nuf, 'Nuf')
    rec_path = os.path.join(base_path, 'Results', lmh_nuf, 'Nuc', method)
    save_path = os.path.join(base_path, 'Results', lmh_nuf, 'Nuc')
    label_path = os.path.join(base_path, 'masks', 'masks')

    # 보정된 이미지 파일 목록(.mat)을 가져온다.
    rec_files = glob.glob(os.path.join(rec_path, '*.mat'))
    
    if not rec_files:
        print(f"오류: '{rec_path}'에서 .mat 파일을 찾을 수 없습니다.")
        return

    results = []

    # --- 평가 루프 ---
    for rec_mat_path in rec_files:
        rec_name = os.path.basename(rec_mat_path)

        if rec_name == 'Misc_83.mat':
            continue

        try:
            # --- 데이터 불러오기 ---
            rec_data = loadmat(rec_mat_path)
            rec = rec_data['pre'].astype(np.float64)

            ori_mat_path = os.path.join(ori_path, rec_name)
            ori_data = loadmat(ori_mat_path)
            ori = ori_data['Ori'].astype(np.float64)
            nuf = ori_data['Nuf'].astype(np.float64) # SCRG 계산을 위한 원본 비균일 이미지

        except (FileNotFoundError, KeyError) as e:
            print(f"파일 로딩 오류 ({rec_name}): {e}. 건너뜁니다.")
            continue

        # 이미지 크기 확인 및 조절
        h, w = ori.shape
        if rec.shape != (h, w):
            rec_img = Image.fromarray(rec).resize((w, h), Image.BILINEAR)
            rec = np.array(rec_img, dtype=np.float64)

        # 목표 영역 좌표 읽기
        lab_mask = xml2mat(label_path, rec_name[:-4])
        if lab_mask is None:
            print(f"좌표 파일 오류 ({rec_name}). 건너뜁니다.")
            continue
        
        target = lab_mask[0]
        pw = 5
        data_max = 255.0

        luc, lur, rdc, rdr = target[0], target[1], target[2], target[3]

        if lur - pw < 0 or luc - pw < 0 or rdr + pw >= h or rdc + pw >= w:
            print(f"배경 영역이 이미지 경계를 벗어남 ({rec_name}). 건너뜁니다.")
            continue

        # --- 품질 지표 계산 ---
        rmse, psnr_val = psnr(ori, rec, data_max)
        ssim_val = ssim(ori, rec, data_range=data_max)
        ir_val = coarseness(rec)
        ln_val = ln(rec)
        
        # SCR Gain을 올바르게 계산하기 위해 보정 전(nuf)과 후(rec) 이미지를 모두 사용
        scr_val, _ = scrg(nuf, rec, pw, lur, luc, rdr, rdc)

        results.append([rmse, psnr_val, ssim_val, ir_val, ln_val, scr_val])
        print(f"{rec_name} 평가 완료.")

    # --- 결과 저장 및 출력 ---
    if not results:
        print("평가할 파일이 없습니다.")
        return

    df = pd.DataFrame(results, columns=['RMSE', 'PSNR', 'SSIM', 'IR', 'Ln', 'SCR'])
    
    # 결과를 Excel 파일로 저장
    os.makedirs(save_path, exist_ok=True)
    excel_name = os.path.join(save_path, f'{method}_RMSE_PSNR_SSIM_IR_Ln_SCR.xlsx')
    df.to_excel(excel_name, index=False)
    print(f"\n결과가 '{excel_name}'에 저장되었습니다.")

    # 각 지표의 평균값을 계산하여 출력
    mean_values = df.mean()
    print("\n--- 평균 결과 ---")
    print(mean_values)

if __name__ == '__main__':
    # 사용 예시
    # MATLAB 스크립트의 파라미터와 동일하게 설정
    base_path_arg = '../dataset/DiffNufTest'
    lmh_nuf_arg = 'Medium'
    method_arg = 'Ours' # 실제 'Ours' 폴더가 존재해야 함

    evaluate_quality_in_batch(base_path_arg, lmh_nuf_arg, method_arg)
