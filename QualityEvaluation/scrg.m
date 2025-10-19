function [s, sg] = scrg(img_in, img_out, pw, lur, luc, rdr, rdc)
% 신호 대 잡음비(SCR, Signal-to-Clutter Ratio) 및 SCR 이득(Gain)을 계산한다.
% 이 함수는 이미지 처리 전후에 특정 목표(target)가 주변 배경(clutter)으로부터
% 얼마나 더 잘 식별되는지를 평가한다.
%
% Args:
%   img_in (matrix): 처리 전 원본 이미지.
%   img_out (matrix): 처리 후 이미지.
%   pw (integer): 목표 주변의 배경(clutter) 영역을 정의할 때 사용할 패딩 너비.
%   lur (integer): 목표 영역의 좌측 상단 행(row) 좌표.
%   luc (integer): 목표 영역의 좌측 상단 열(column) 좌표.
%   rdr (integer): 목표 영역의 우측 하단 행(row) 좌표.
%   rdc (integer): 목표 영역의 우측 하단 열(column) 좌표.
%
% Returns:
%   s (double): 처리 후 이미지의 SCR 값.
%   sg (double): SCR 이득 (처리 후 SCR / 처리 전 SCR).

% --- 목표(Target) 영역 추출 ---
% 입력 이미지와 출력 이미지에서 지정된 좌표를 사용하여 목표 영역을 잘라낸다.
img_in_target = img_in(lur:rdr, luc:rdc);
img_out_target = img_out(lur:rdr, luc:rdc);

% --- 배경(Clutter) 영역 추출 ---
% 목표 영역을 둘러싸는 네 개의 사각형 영역을 배경(clutter)으로 정의하고 추출한다.
% 상단 배경 영역
img_in_padding_a = img_in(lur-pw:lur-1, luc-pw:rdc);
% 우측 배경 영역
img_in_padding_b = img_in(lur-pw:rdr, rdc+1:rdc+pw);
% 하단 배경 영역
img_in_padding_c = img_in(rdr+1:rdr+pw, luc:rdc+pw);
% 좌측 배경 영역
img_in_padding_d = img_in(lur:rdr+pw, luc-pw:luc-1);
% 추출된 모든 배경 영역의 픽셀들을 하나의 벡터로 결합한다.
img_in_padding = [img_in_padding_a(:); img_in_padding_b(:); img_in_padding_c(:); img_in_padding_d(:)];

% 처리 후 이미지에 대해서도 동일하게 배경 영역을 추출한다.
img_out_padding_a = img_out(lur-pw:lur-1, luc-pw:rdc);
img_out_padding_b = img_out(lur-pw:rdr, rdc+1:rdc+pw);
img_out_padding_c = img_out(rdr+1:rdr+pw, luc:rdc+pw);
img_out_padding_d = img_out(lur:rdr+pw, luc-pw:luc-1);
img_out_padding = [img_out_padding_a(:); img_out_padding_b(:); img_out_padding_c(:); img_out_padding_d(:)];

% --- SCR 및 SCR Gain 계산 ---
% 모든 데이터 타입을 double로 변환하여 계산 정확도를 높인다.
img_in_target = double(img_in_target(:));
img_out_target = double(img_out_target(:));
img_in_padding = double(img_in_padding(:));
img_out_padding = double(img_out_padding(:));

% SCR 공식: (목표 평균 - 배경 평균) / 배경 표준편차
% 처리 전 이미지의 SCR을 계산한다.
scr_in = (mean(img_in_target) - mean(img_in_padding)) / std(img_in_padding);
% 처리 후 이미지의 SCR을 계산한다.
scr_out = (mean(img_out_target) - mean(img_out_padding)) / std(img_out_padding);

% SCR 이득(Gain)을 계산한다. 이 값이 1보다 크면 목표 식별 성능이 향상된 것이다.
sg = scr_out / scr_in;
% 처리 후 SCR 값을 반환한다.
s = scr_out;

end
