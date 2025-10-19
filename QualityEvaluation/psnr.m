function [RMSE, p] = psnr(img1, img2, data_max)
% 두 이미지 간의 최대 신호 대 잡음비(PSNR)와 제곱 평균 오차 제곱근(RMSE)을 계산한다.
%
% Args:
%   img1 (matrix): 첫 번째 이미지. 원본 이미지에 해당.
%   img2 (matrix): 두 번째 이미지. 복원된 또는 왜곡된 이미지에 해당.
%   data_max (numeric): 이미지의 최대 픽셀 값 (예: 8비트 이미지의 경우 255).
%
% Returns:
%   RMSE (double): 두 이미지 간의 제곱 평균 오차 제곱근.
%   p (double): PSNR 값 (단위: dB).

% 정확한 계산을 위해 입력 이미지를 double 타입으로 변환.
img1 = double(img1);
img2 = double(img2);

% 두 이미지의 차이를 계산.
diff = img1 - img2;

% 평균 제곱 오차(Mean Squared Error, MSE)를 계산.
% (:) 연산자는 행렬을 1차원 벡터로 변환한다.
% .^2는 요소별 제곱 연산을 수행한다.
% mean() 함수는 벡터의 모든 요소의 평균을 계산한다.
MSE = mean(diff(:).^2);

% 제곱 평균 오차 제곱근(Root Mean Squared Error, RMSE)을 계산.
RMSE = sqrt(MSE);

% PSNR(Peak Signal-to-Noise Ratio)을 계산.
% PSNR (dB) = 20 * log10(MAX / RMSE)
p = 20 * log10(data_max / RMSE);

end
