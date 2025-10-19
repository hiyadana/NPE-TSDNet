# NPE-TSDNet: 적외선 이미지 비균일성 보정을 위한 딥러닝 모델

이 프로젝트는 적외선 이미지의 고정 패턴 노이즈, 특히 줄무늬 형태의 비균일성(Non-uniformity)을 보정하기 위한 딥러닝 모델인 NPE-TSDNet의 TensorFlow 구현체이다. 모델은 이득(Gain)과 오프셋(Offset) 보정을 순차적으로 수행하는 2단계 직렬(cascaded) 네트워크 구조를 특징으로 한다.

## 프로젝트 구조

```
d:\workspace\NPE-TSDNet\
├── train.py                  # 전체 네트워크 Fine-tuning 스크립트
├── Test_DiffNuf.py           # 학습된 최종 모델 테스트 스크립트
├── gc_train/                 # G-네트워크(이득 보정) 사전 학습 관련 모듈
│   ├── train_gc.py           # G-네트워크 사전 학습 실행 스크립트
│   ├── g_cost.py
│   ├── g_getdata.py
│   └── ...
├── oc_train/                 # O-네트워크(오프셋 보정) 사전 학습 관련 모듈
│   ├── train_oc.py           # O-네트워크 사전 학습 실행 스크립트
│   ├── o_cost.py
│   ├── o_getdata.py
│   └── ...
├── dataset/
│   ├── GenerateSdata.py      # 학습 데이터셋 생성 스크립트
│   ├── GenerateDiffNuf.py    # 테스트 데이터셋 생성 스크립트
│   ├── trainset/             # 생성된 학습 데이터 저장 폴더
│   ├── valiset/              # 생성된 검증 데이터 저장 폴더
│   └── DiffNufTest/          # 생성된 테스트 데이터 저장 폴더
├── model/                    # 최종 학습된 모델(g_o_model.ckpt) 저장 폴더
├── QualityEvaluation/        # MATLAB 기반 성능 평가 스크립트
│   ├── DiffNufQe.m           # 종합 평가 실행 스크립트
│   ├── psnr.m
│   ├── ssim.m (필요 시 추가)
│   └── ...
├── g_output.py               # 이득 보정 적용 모듈
├── o_output.py               # 오프셋 보정 적용 모듈
├── go_cost.py                # 전체 모델 비용 함수 모듈
├── getdata.py                # 메인 학습용 데이터 로더
├── multi_scale_conv.py       # 다중 스케일 컨볼루션 블록
└── conv_layer.py             # 기본 컨볼루션 레이어
```

## 방법론

NPE-TSDNet은 두 개의 주요 서브 네트워크로 구성된 직렬 구조를 가진다.

1.  **G-Net (이득 보정 네트워크)**: 8개의 다중 스케일 컨볼루션 레이어로 구성된 CNN. 입력 이미지로부터 열(column) 단위의 이득(gain) 불균일성을 추정하고 보정한다.
2.  **O-Net (오프셋 보정 네트워크)**: 15개의 다중 스케일 컨볼루션 레이어로 구성된 더 깊은 CNN. G-Net의 출력을 입력으로 받아 열 단위의 오프셋(offset) 불균일성을 추정하고 보정한다.

### 3단계 학습 전략

이 모델은 효과적인 학습을 위해 3단계 전략을 사용한다.

1.  **G-Net 사전 학습**: `gc_train/train_gc.py`를 실행. G-Net이 노이즈 이미지로부터 "정답 이득 보정 계수"를 직접 예측하도록 학습시킨다.
2.  **O-Net 사전 학습**: `oc_train/train_oc.py`를 실행. O-Net이 노이즈 이미지로부터 "정답 오프셋 보정 계수"를 직접 예측하도록 학습시킨다.
3.  **종단간 Fine-tuning**: `train.py`를 실행. 사전 학습된 G-Net과 O-Net을 결합한 전체 모델을 불러와, 최종 보정 이미지와 원본 정답 이미지 간의 MSE 손실을 최소화하도록 미세 조정한다.

## 사용 방법

### 1. 사전 요구사항

Python 및 TensorFlow 1.x 환경이 필요하다. 다음 라이브러리들을 설치해야 한다.

-   `tensorflow-gpu==1.15` (또는 호환되는 1.x 버전)
-   `numpy`
-   `scipy`
-   `opencv-python`
-   `pillow`
-   `pandas`
-   `xlswrite` (MATLAB 결과 저장을 위해)

MATLAB (성능 평가를 위해 필요)

### 2. 데이터 준비

1.  **원본 이미지 다운로드**: 깨끗한 이미지 데이터셋(예: [MS-COCO](http://cocodataset.org))을 다운로드하여 `dataset/MS-COCO/` 폴더에 저장한다.
2.  **학습 데이터 생성**: `dataset/GenerateSdata.py`를 실행하여 `dataset/trainset/` 폴더에 학습용 `.mat` 파일들을 생성한다.
3.  **테스트 데이터 생성**: `dataset/GenerateDiffNuf.py`를 실행하여 `dataset/DiffNufTest/` 폴더에 고정된 노이즈 수준을 가진 테스트용 `.mat` 파일들을 생성한다.

### 3. 모델 학습

1.  **G-Net 사전 학습**:
    ```bash
    python gc_train/train_gc.py
    ```
    학습 완료 후, 생성된 `model/g_model.ckpt` 관련 파일들을 `gc_train/model/` 폴더로 이동시킨다.

2.  **O-Net 사전 학습**:
    ```bash
    python oc_train/train_oc.py
    ```
    학습 완료 후, 생성된 `model/o_model.ckpt` 관련 파일들을 `oc_train/model/` 폴더로 이동시킨다.

3.  **전체 모델 Fine-tuning**:
    ```bash
    python train.py
    ```
    최종 모델인 `g_o_model.ckpt`가 `model/` 폴더에 저장된다.

### 4. 테스트

-   `Test_DiffNuf.py`를 실행하여 `model/` 폴더의 최종 모델을 사용해 `dataset/DiffNufTest/`의 테스트 이미지들을 보정한다. 결과는 `dataset/DiffNufTest/Results/` 폴더에 저장된다.

    ```bash
    python Test_DiffNuf.py
    ```

### 5. 성능 평가

-   MATLAB을 실행한다.
-   `QualityEvaluation/` 디렉토리로 이동한다.
-   `DiffNufQe.m` 스크립트를 열어 평가하려는 노이즈 수준(`lmh_nuf`)과 방법(`method`)을 설정한 후 실행한다.
-   평가 결과 지표(RMSE, PSNR, SSIM 등)가 Excel 파일로 저장되고 MATLAB 명령 창에 평균값이 출력된다.

### 6. 성능 평가 (Python 버전)

`QualityEvaluation_Python/` 폴더는 MATLAB 라이선스 없이 성능 평가를 수행할 수 있도록 `QualityEvaluation/`의 MATLAB 스크립트들을 Python으로 변환한 버전이다.

-   **사전 요구사항**: `scikit-image`, `pandas`, `openpyxl` 라이브러리가 추가로 필요하다.
    ```bash
    pip install scikit-image pandas openpyxl
    ```
-   **실행 방법**: `QualityEvaluation_Python/` 디렉토리로 이동하여 `diff_nuf_qe.py`를 실행한다. 스크립트 내의 `base_path_arg`, `lmh_nuf_arg`, `method_arg` 변수를 자신의 환경에 맞게 수정한 후 실행할 수 있다.
    ```bash
    cd QualityEvaluation_Python
    python diff_nuf_qe.py
    ```

#### Python 평가 스크립트 상세

-   `diff_nuf_qe.py`: MATLAB의 `DiffNufQe.m`에 해당하는 메인 스크립트. 지정된 폴더의 모든 결과에 대해 아래의 모든 품질 지표를 일괄 계산하고 Excel 파일로 저장한다.
-   `psnr.py`: 두 이미지 간의 PSNR(최대 신호 대 잡음비)과 RMSE(평균 제곱근 오차)를 계산한다.
-   `coarseness.py`: 이미지의 거칠기(조잡도)를 계산하여 노이즈 수준을 평가한다.
-   `ln.py`: 이미지의 저주파 비균일성을 평가한다.
-   `scrg.py`: 특정 목표(target)와 주변 배경(clutter) 간의 SCR(신호 대 잡음비)을 계산하여 목표 식별 성능을 평가한다.
-   `icv_mrd.py`: 평탄한 영역의 균일성(ICV)과 디테일 영역의 상대 오차(MRD)를 계산하여 보정 성능을 종합적으로 평가한다.

```