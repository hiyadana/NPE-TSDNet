"""다중 스케일 특징 추출 유닛 모듈"""
import tensorflow as tf
from conv_layer import ConvLayer

class MS_conv(object):
    """
    다중 스케일 컨볼루션(Multi-Scale Convolution) 블록을 구현하는 클래스.

    이 클래스는 입력 텐서를 세 개의 병렬 컨볼루션 경로(1x1, 3x3, 1x5->5x1)로
    처리하여 다양한 스케일의 특징을 동시에 추출한다. 각 경로에서 나온 특징 맵들은
    채널 방향으로 결합(concatenate)된 후, 1x1 컨볼루션을 통해 최종 출력 특징 맵을 생성한다.
    이는 Inception-style 모듈과 유사한 아이디어에 기반한다.
    """
    def __init__(self, inpt, channels=64, ms_name="ms"):
        """
        MS_conv 객체를 초기화한다.

        Args:
            inpt (tf.Tensor): 입력 텐서.
            channels (int, optional): 입력 텐서의 채널 수. 기본값은 64.
            ms_name (str, optional): 레이어의 이름 접두사. 내부 레이어들의 이름에 사용됨. 기본값은 "ms".
        """
        # 내부 레이어들의 이름을 설정하기 위한 접두사 생성
        name_1 = ms_name + "_1"
        name_2 = ms_name + "_2"
        name_3_1 = ms_name + "_3_1"
        name_3_2 = ms_name + "_3_2"
        name_0 = ms_name + "_0"

        self.inpt = inpt
        self.channels = channels

        # 경로 1: 1x1 컨볼루션. 특징 맵의 채널 수를 조절하고 계산량을 줄이는 역할.
        MS_1_conv = ConvLayer(self.inpt, filter_shape=[1, 1, self.channels, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                     padding="SAME", cl_name=name_1)

        # 경로 2: 3x3 컨볼루션. 표준적인 크기의 특징을 추출.
        MS_2_conv = ConvLayer(self.inpt, filter_shape=[3, 3, self.channels, 64], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                padding="SAME", cl_name=name_2)

        # 경로 3: 1x5와 5x1 컨볼루션을 순차적으로 적용. 이는 5x5 컨볼루션을
        # 두 개의 비대칭(asymmetric) 컨볼루션으로 분해(factorize)한 것으로, 파라미터 수를 줄이면서
        # 유사한 수용 영역(receptive field)을 갖도록 한다.
        MS_3_1_conv = ConvLayer(self.inpt, filter_shape=[1, 5, self.channels, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                              padding="SAME", cl_name=name_3_1)
        MS_3_2_conv = ConvLayer(MS_3_1_conv.output, filter_shape=[5, 1, 32, 32], strides=[1, 1, 1, 1], activation=ConvLayer.LeakyRelu,
                                padding="SAME", cl_name=name_3_2)

        # 세 개의 병렬 경로에서 나온 출력 텐서들을 리스트에 담는다.
        frames = [MS_1_conv.output, MS_2_conv.output, MS_3_2_conv.output]
        # 채널(axis=3)을 기준으로 텐서들을 결합(concatenate)한다.
        layer_pj = tf.concat(frames, 3)
        
        # 최종 1x1 컨볼루션: 결합된 특징 맵들을 통합하고, 최종 출력 채널 수를 64로 맞춘다.
        # 활성화 함수가 없으므로 선형 변환 역할을 한다.
        layer_conv = ConvLayer(layer_pj, filter_shape=[1, 1, 128, 64], strides=[1, 1, 1, 1], activation=None,
                                   padding="SAME", cl_name=name_0)

        # 이 블록에 포함된 모든 학습 가능한 파라미터들을 수집한다.
        self.params = MS_1_conv.params + MS_2_conv.params + MS_3_1_conv.params + MS_3_2_conv.params + layer_conv.params
        # 블록의 최종 출력을 저장한다.
        self.output = layer_conv.output
