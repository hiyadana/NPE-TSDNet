% 비균일성 보정(NUC) 결과에 대한 정량적 품질 평가 스크립트
%
% 이 스크립트는 지정된 폴더에 있는 모든 보정 결과 이미지에 대해
% 다양한 품질 평가 지표(RMSE, PSNR, SSIM, IR, Ln, SCR)를 일괄적으로 계산하고,
% 그 결과를 Excel 파일로 저장한 후 평균값을 출력한다.

% --- 초기화 ---
clear all;
clc;

% --- 경로 및 파라미터 설정 ---
base_path = '..\dataset\DiffNufTest'; % 테스트 데이터셋의 기본 경로
lmh_nuf = 'Medium'; % 평가할 노이즈 수준 ('Low', 'Medium', 'High')
method = 'Ours'; % 평가할 보정 방법의 이름

% 각 데이터의 전체 경로를 구성
ori_path = [base_path, filesep, lmh_nuf, 'Nuf', filesep]; % 원본 비균일 이미지 경로
rec_path = [base_path, filesep, 'Results', filesep, lmh_nuf, 'Nuc', filesep, method, filesep]; % 보정된 이미지 경로
save_path = [base_path, filesep, 'Results', filesep, lmh_nuf, 'Nuc', filesep]; % 결과 Excel 파일을 저장할 경로
label_path = [base_path, filesep, 'masks', filesep, 'masks', filesep]; % 목표 영역 좌표(mask) 파일 경로

% 보정된 이미지 파일 목록을 가져온다.
rec_files = dir([rec_path, '*.mat']);
rec_num = length(rec_files);

% 평가 결과를 저장할 셀(cell) 배열 초기화
QE = cell(1, 1);

% --- 평가 루프 ---
% 모든 보정 결과 파일을 순회
for nn = 1:rec_num
    rec_file = rec_files(nn);
    rec_name = rec_file.name;
    
    % 특정 파일을 평가에서 제외할 경우 사용
    if strcmp(rec_name, 'Misc_83.mat')
        continue;
    end
    
    % --- 데이터 불러오기 ---
    % 보정된 이미지(.mat)를 불러온다.
    rec_mat = [rec_path, rec_name];
    rec_data = load(rec_mat);
    rec = rec_data.pre(:,:); % 'pre' 필드에 보정된 이미지 데이터가 저장되어 있음

    % 원본 정답 이미지(.mat)를 불러온다.
    ori_mat = [ori_path, rec_name];
    ori_data = load(ori_mat);
    ori = ori_data.Ori(:,:); % 'Ori' 필드에 원본 정답 이미지가 저장되어 있음
    
    % 이미지 크기 확인 및 조절
    [h, w] = size(ori);
    [hh, ww] = size(rec);
    if h ~= hh || w ~= ww
        rec = imresize(rec, [h, w]);
    end
    
    % SCR 계산을 위한 목표 영역 좌표를 xml 파일로부터 읽어온다. (사용자 정의 함수 xml2mat 필요)
    lab_mask = xml2mat(label_path, rec_name(1:end-4));
    target = lab_mask(1,:);
    pw = 5; % SCR 계산 시 사용할 배경 영역의 너비
    data_max = 255; % 8비트 이미지의 최대 픽셀 값
    
    % 목표 영역 좌표
    lur = target(2); % 좌측 상단 행
    luc = target(1); % 좌측 상단 열
    rdr = target(4); % 우측 하단 행
    rdc = target(3); % 우측 하단 열
    
    % 배경 영역이 이미지 경계를 벗어나는 경우, 해당 파일은 건너뛴다.
    if lur-pw<1 || luc-pw<1 || rdr+pw>h || rdc+pw>w
        continue;
    end     
   
    % SSIM 계산을 위해 이미지를 0~1 범위로 정규화
    ori_s = double(ori) / 255.0;
    rec_s = double(rec) / 255.0;
    
    % --- 품질 지표 계산 ---
    IR = coarseness(rec); % 조잡도 (노이즈 수준)
    Ln = ln(rec); % 저주파 비균일성
    
    % SCR 및 SCRG 계산.
    % 참고: 아래 호출은 SCR Gain(SCRG)을 올바르게 계산하지 않는다.
    %       입력(img_in)과 출력(img_out)에 모두 보정 후 이미지(rec)를 사용했기 때문.
    %       올바른 SCRG를 계산하려면 img_in에 보정 전 이미지(ori_data.Nuf)를 사용해야 한다.
    %       예: [SCR, SCRG] = scrg(ori_data.Nuf, rec, pw, lur, luc, rdr, rdc);
    [SCR, ~] = scrg(rec, rec, pw, lur, luc, rdr, rdc);
    
    [RMSE, PSNR] = psnr(ori, rec, data_max); % RMSE 및 PSNR
    SSIM = ssim(ori_s, rec_s); % 구조적 유사성 (SSIM)
    
    % 계산된 지표들을 결과 행렬에 추가
    QE{1} = [QE{1}; RMSE, PSNR, SSIM, IR, Ln, SCR];
end

% --- 결과 저장 및 출력 ---
% 전체 결과를 Excel 파일로 저장
xls_name = [save_path, method, '_RMSE_PSNR_SSIM_IR_Ln_SCR.xlsx'];
xlswrite(xls_name, QE{1}, 1, 'A1');

% 각 지표의 평균값을 계산
RMSE_m = mean(QE{1}(:,1));
PSNR_m = mean(QE{1}(:,2));
SSIM_m = mean(QE{1}(:,3));
IR_m = mean(QE{1}(:,4));
Ln_m = mean(QE{1}(:,5));
SCR_m = mean(QE{1}(:,6));

% 평균값들을 명령 창에 출력
disp([num2str(RMSE_m), ';', num2str(PSNR_m), ';', num2str(SSIM_m), ';', num2str(IR_m), ';', num2str(Ln_m), ';', num2str(SCR_m)]);