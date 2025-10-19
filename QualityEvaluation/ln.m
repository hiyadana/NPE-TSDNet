function Ln = ln(img)
% 이미지의 저주파 비균일성(Low-frequency Non-uniformity) 지표를 계산한다.
%
% 이 함수는 이미지 중앙 픽셀과 네 모서리 픽셀 간의 상대적 차이를 평균하여
% 이미지 전체의 대략적인 균일성 수준을 평가한다.
% 입력 이미지가 균일한 평면(flat field)이라고 가정하며, 그럴 경우 Ln 값은 0에 가깝다.
%
% Args:
%   img (matrix): 분석할 흑백 이미지.
%
% Returns:
%   Ln (double): 계산된 저주파 비균일성 값.

% 이미지의 높이(h)와 너비(w)를 가져온다.
h = size(img, 1);
w = size(img, 2);

% 정확한 계산을 위해 이미지를 double 타입으로 변환.
img = double(img);

% --- 중앙 및 모서리 픽셀 값 추출 ---
% 이미지의 중앙 픽셀 값을 가져온다.
Lc = img(round(h/2), round(w/2));

% 네 개의 모서리 부근(가장자리에서 8픽셀 안쪽)의 픽셀 값을 가져온다.
% 가장자리 픽셀을 피하는 것은 경계 아티팩트를 방지하기 위함일 수 있다.
L1 = img(8, 8);           % 좌측 상단
L2 = img(8, w-8);         % 우측 상단
L3 = img(h-8, w-8);       % 우측 하단
L4 = img(h-8, 8);           % 좌측 하단

% --- 비균일성 지표 계산 ---
% 중앙 픽셀 값 대비 각 모서리 픽셀 값의 상대적 차이의 절댓값을 구하고, 이들의 평균을 계산한다.
% Ln = mean(abs(Lc - Li) / Lc) for i=1 to 4
Ln = (abs((Lc - L1) / Lc) + abs((Lc - L2) / Lc) + abs((Lc - L3) / Lc) + abs((Lc - L4) / Lc)) * (1/4);

end
