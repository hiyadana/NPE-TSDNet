# NPE-TSDNet (PyTorch Version)

이 프로젝트는 기존 TensorFlow로 구현된 NPE-TSDNet을 PyTorch로 변환한 버전이다.

## 프로젝트 구조 (PyTorch Version)

```
NPE-TSDNet/
└── pytorch_version/
    ├── models.py             # G-Net, O-Net, NPE_TSDNet 모델 아키텍처
    ├── dataset.py            # PyTorch Dataset 및 DataLoader 정의
    ├── train_gc.py           # G-Net 사전 학습 스크립트
    ├── train_oc.py           # O-Net 사전 학습 스크립트
    ├── train.py              # 전체 모델 Fine-tuning 스크립트
    ├── test.py               # 최종 모델 테스트 스크립트
    ├── conv_layer.py         # 기본 컨볼루션 레이어
    ├── multi_scale_conv.py   # 다중 스케일 컨볼루션 블록
    └── saved_models/         # 학습된 PyTorch 모델(.pth)이 저장될 폴더
```

## 네트워크 아키텍처 (Network Architecture)

NPE-TSDNet은 두 개의 서브 네트워크(G-Net, O-Net)가 직렬(cascaded)로 연결된 구조를 가진다. 전체적인 데이터 흐름은 다음과 같다.

`입력 이미지` -> `G-Net` -> `이득 보정` -> `O-Net` -> `오프셋 보정` -> `최종 출력 이미지`

### MS_conv 블록

네트워크의 핵심 구성 요소는 다중 스케일 컨볼루션 블록(`MS_conv`)이다. 이 블록은 Inception-style 아키텍처와 유사하게, 다양한 크기의 필터를 병렬로 사용하여 입력 데이터로부터 여러 스케일의 특징을 동시에 추출한다.

-   **경로 1**: `1x1` 컨볼루션
-   **경로 2**: `3x3` 컨볼루션
-   **경로 3**: `1x5` 컨볼루션 -> `5x1` 컨볼루션 (비대칭 컨볼루션)

각 경로의 출력 특징 맵은 채널(channel) 차원에서 결합(concatenate)된 후, `1x1` 컨볼루션을 통해 최종적으로 통합된 특징 맵을 생성한다.

### G-Net (이득 보정 네트워크)

-   **구조**: 8개의 `MS_conv` 블록이 순차적으로 연결되고, 마지막에 `1x1` 컨볼루션 레이어가 추가된다.
-   **역할**: 입력 이미지의 이득(gain) 불균일성을 모델링한다.
-   **출력**: 네트워크를 통과한 최종 특징 맵은 높이(height) 차원을 기준으로 평균(`torch.mean`)되어, 각 열(column)에 대한 단일 이득 보정 값으로 구성된 `gain_vector`를 생성한다. 이 벡터가 원본 이미지에 곱해져 이득 보정이 수행된다.

### O-Net (오프셋 보정 네트워크)

-   **구조**: 15개의 `MS_conv` 블록이 순차적으로 연결되고, 마지막에 `1x1` 컨볼루션 레이어가 추가된 더 깊은 네트워크이다.
-   **역할**: 이득이 보정된 이미지에 남아있는 오프셋(offset) 불균일성을 모델링한다.
-   **출력**: G-Net과 유사하게, 최종 특징 맵으로부터 각 열에 대한 `offset_vector`를 생성한다. 이 벡터가 이득이 보정된 이미지에 더해져 최종 보정이 완료된다.

## 학습 전략 (Training Strategy)

이 모델은 효과적인 학습을 위해 3단계 전략을 사용한다.

1.  **G-Net 사전 학습**: `train_gc.py`를 실행. G-Net이 노이즈 이미지로부터 "정답 이득 보정 계수"를 직접 예측하도록 학습시킨다.
2.  **O-Net 사전 학습**: `train_oc.py`를 실행. O-Net이 노이즈 이미지로부터 "정답 오프셋 보정 계수"를 직접 예측하도록 학습시킨다.
3.  **종단간 Fine-tuning**: `train.py`를 실행. 사전 학습된 G-Net과 O-Net을 결합한 전체 모델을 불러와, 최종 보정 이미지와 원본 정답 이미지 간의 MSE 손실을 최소화하도록 미세 조정한다.

## 사용 방법 (PyTorch Version)

### 1. 사전 요구사항

Python 및 PyTorch 환경이 필요하다. 다음 라이브러리들을 설치해야 한다.

-   `torch` 및 `torchvision` (CUDA 버전에 맞게 설치)
-   `numpy`
-   `scipy`
-   `opencv-python`

### 2. 데이터 준비

데이터 준비 과정은 기존 TensorFlow 버전과 동일하다. 원본 프로젝트의 `dataset` 폴더에 있는 스크립트를 사용한다.

1.  **원본 이미지 다운로드**: 깨끗한 이미지 데이터셋(예: [MS-COCO](http://cocodataset.org))을 다운로드하여 `dataset/MS-COCO/` 폴더에 저장한다.
2.  **학습 데이터 생성**: `dataset/GenerateSdata.py`를 실행하여 `dataset/trainset/` 폴더에 학습용 `.mat` 파일들을 생성한다.
3.  **테스트 데이터 생성**: `dataset/GenerateDiffNuf.py`를 실행하여 `dataset/DiffNufTest/` 폴더에 테스트용 `.mat` 파일들을 생성한다.

### 3. 모델 학습 (PyTorch)

학습 스크립트들은 `pytorch_version` 디렉토리 내에 있다.

1.  **G-Net 사전 학습**:
    ```bash
    python pytorch_version/train_gc.py
    ```
    학습 완료 후, `pytorch_version/saved_models/g_model.pth` 파일이 생성된다.

2.  **O-Net 사전 학습**:
    ```bash
    python pytorch_version/train_oc.py
    ```
    학습 완료 후, `pytorch_version/saved_models/o_model.pth` 파일이 생성된다.

3.  **전체 모델 Fine-tuning**:
    ```bash
    python pytorch_version/train.py
    ```
    사전 학습된 G-Net과 O-Net의 가중치를 불러와 전체 모델을 학습한다. 최종 모델인 `npe_tsdnet_model.pth`가 `pytorch_version/saved_models/` 폴더에 저장된다.

### 4. 테스트 (PyTorch)

-   `pytorch_version/test.py`를 실행하여 최종 모델의 성능을 테스트한다. 결과는 `dataset/DiffNufTest/Results/Ours_PyTorch/` 폴더에 저장된다.

    ```bash
    python pytorch_version/test.py
    ```

### 5. 성능 평가 (Python 버전)

`QualityEvaluation_Python/` 폴더는 MATLAB 라이선스 없이 성능 평가를 수행할 수 있도록 `QualityEvaluation/`의 MATLAB 스크립트들을 Python으로 변환한 버전이다. PyTorch 모델의 결과물을 평가하는 데 사용할 수 있다.

-   **사전 요구사항**: `scikit-image`, `pandas`, `openpyxl` 라이브러리가 추가로 필요하다.
    ```bash
    pip install scikit-image pandas openpyxl
    ```
-   **실행 방법**: `diff_nuf_qe.py` 스크립트를 열어 `method_arg` 변수를 PyTorch 테스트 결과가 저장된 폴더 이름(예: `Ours_PyTorch`)으로 변경한다. 그 후, 스크립트를 실행한다.
    ```bash
    # 1. diff_nuf_qe.py 파일 수정:
    #    method_arg = 'Ours_PyTorch'

    # 2. 스크립트 실행:
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