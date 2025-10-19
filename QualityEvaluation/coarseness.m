function IR = coarseness(graypic)
% 이미지의 조잡도(Coarseness) 또는 거칠기(Roughness)를 계산한다.
% 이 지표는 이미지의 전체 밝기 대비 수평 및 수직 방향의 변화량 합계를 나타낸다.
% 값이 높을수록 이미지가 더 거칠거나 노이즈가 많음을 의미한다.
%
% Args:
%   graypic (matrix): 분석할 흑백 이미지.
%
% Returns:
%   IR (double): 계산된 조잡도 값.

% --- 필터 정의 ---
% 수평 방향의 그래디언트(인접 픽셀 간의 차이)를 계산하기 위한 필터.
h_m = [1, -1];
% 수직 방향의 그래디언트를 계산하기 위한 필터.
v_m = h_m'; % h_m의 전치 행렬

% 정확한 계산을 위해 입력 이미지를 double 타입으로 변환.
graypic = double(graypic);

% --- 수평 방향 변화량 계산 ---
% conv2 함수를 사용하여 이미지와 수평 필터를 컨볼루션한다.
I_h = conv2(graypic, h_m);
% conv2의 'full' 옵션(기본값)은 입력보다 큰 결과를 생성하므로, 경계의 열들을 제거한다.
% 참고: conv2(graypic, h_m, 'same') 옵션을 사용하면 이 과정이 필요 없다.
I_h(:, 1) = [];
I_h(:, end) = [];
% 수평 방향 그래디언트의 절댓값 합(L1-norm)을 계산.
I_h_L1 = sum(sum(abs(I_h)));

% --- 수직 방향 변화량 계산 ---
% 이미지와 수직 필터를 컨볼루션한다.
I_v = conv2(graypic, v_m);
% 경계의 행들을 제거한다.
I_v(1, :) = [];
I_v(end, :) = [];
% 수직 방향 그래디언트의 절댓값 합(L1-norm)을 계산.
I_v_L1 = sum(sum(abs(I_v)));

% --- 정규화 및 최종 계산 ---
% 원본 이미지의 전체 픽셀 값 합(L1-norm)을 계산.
I_L1 = sum(sum(abs(graypic)));

% 조잡도(IR) 계산: (수평 변화량 + 수직 변화량) / 전체 픽셀 값 합
IR = (I_h_L1 + I_v_L1) / I_L1;

end