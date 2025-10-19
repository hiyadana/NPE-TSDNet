% 비균일성 보정(NUC) 성능 평가 스크립트 (ICV, MRD)
%
% 이 스크립트는 보정된 이미지의 품질을 두 가지 지표로 평가한다:
% 1. ICV (Inverse Coefficient of Variation): 평탄한 영역의 균일성 측정.
%    값이 높을수록 노이즈 억제 성능이 우수함을 의미한다.
% 2. MRD (Mean Relative Deviation): 디테일 영역의 상대적 오차 측정.
%    값이 낮을수록 디테일 보존 성능이 우수함을 의미한다.

% --- 초기화 ---
close all; % 모든 그림 창 닫기
clear all; % 작업 공간의 모든 변수 삭제
clc;       % 명령 창 내용 지우기

% --- 파일 경로 설정 ---
% recPath: 보정 후 이미지 경로
% nufPath: 보정 전 원본 비균일 이미지 경로
% 참고: 아래 코드에서 rec = nuf 로 설정되어 있어, 현재는 nufPath 이미지 자체의
%      특성을 분석하는 상태이다. 보정 성능을 평가하려면 recPath에 보정된
%      이미지를 실제로 불러와야 한다.
recPath = '..\dataset\RealIRresults\Test_Pre_crop_1.png';
nufPath = '..\dataset\RealIR\crop_1.png';

% --- 이미지 불러오기 및 전처리 ---
% rec = imread(recPath); % 보정된 이미지를 불러올 경우 이 라인의 주석을 해제.
% rec = double(rec);
nuf = imread(nufPath); % 원본 비균일 이미지를 불러온다.

% 이미지가 컬러일 경우 흑백으로 변환.
if size(nuf, 3) == 3
    nuf = rgb2gray(nuf);
end
nuf = double(nuf);

% 현재는 보정된 이미지(rec)를 원본 비균일 이미지(nuf)와 동일하게 설정.
% 이는 스크립트 디버깅 또는 단일 이미지 분석 목적일 수 있다.
rec = nuf;

% 두 이미지의 크기가 다를 경우, rec 이미지의 크기를 nuf 이미지에 맞게 조절.
h = size(nuf, 1);
w = size(nuf, 2);
hh = size(rec, 1);
ww = size(rec, 2);
if h ~= hh || w ~= ww
    rec = imresize(rec, [h, w]);
end

% --- ICV (Inverse Coefficient of Variation) 계산 ---
% ICV는 평탄한 영역의 노이즈 수준을 평가한다. (평균 / 표준편차)

% crop_1.png 이미지의 평탄한 영역들을 수동으로 지정 (ROI: Region of Interest).
% 이 좌표들은 해당 이미지에 특화되어 있다.
jy_1 = rec(1:50, 1:155);
jy_2 = rec(1:85, 236:310);
jy_3 = rec(1:30, 366:480);
jy_4 = rec(91:200, 1:155);
jy_5 = rec(116:200, 216:400);
jy_6 = rec(301:400, 1:480);

% 모든 평탄 영역의 픽셀들을 하나의 벡터로 결합.
jy = [jy_1(:); jy_2(:); jy_3(:); jy_4(:); jy_5(:); jy_6(:)];
jy = double(jy);

% 평탄 영역의 평균을 표준편차로 나누어 ICV 값을 계산.
ICV = mean(jy(:)) / std(jy(:))

% --- MRD (Mean Relative Deviation) 계산 ---
% MRD는 디테일 영역의 오차를 평가한다.

% ICV 계산에 사용된 평탄 영역을 마스킹 처리하여 제외한다.
% 해당 영역의 픽셀 값을 특정한 값(-1000)으로 설정.
rec(1:50, 1:155) = -1000;
rec(1:85, 236:310) = -1000;
rec(1:30, 366:480) = -1000;
rec(91:200, 1:155) = -1000;
rec(116:200, 216:400) = -1000;
rec(301:400, 1:480) = -1000;

% 마스킹되지 않은 영역(디테일 영역)의 인덱스를 찾는다.
mask = find(rec ~= -1000);

% 디테일 영역에서만 두 이미지 간의 차이를 계산.
r_n = abs(rec - nuf);

% 디테일 영역에서 원본 이미지 대비 상대적 오차를 계산.
% 0으로 나누는 것을 방지하기 위해 nuf(mask)가 0이 아닌 픽셀만 고려해야 함.
mm = r_n(mask) ./ nuf(mask);

% 상대적 오차의 평균을 계산하여 MRD 값을 구한다.
MRD = mean(mm(:))

% --- 최종 종합 점수 계산 ---
% ICV(균일성 점수)를 MRD(오차 점수)로 나누어 최종 점수를 계산.
% 이 값이 높을수록 전반적인 보정 성능이 우수함을 의미.
zh = ICV / MRD