clear all;
clc;
base_path = '..\dataset\DiffNufTest';
lmh_nuf = 'Medium';
method = 'Ours';
ori_path = [base_path,lmh_nuf,'Nuf\'];
rec_path = [base_path,'Results\',lmh_nuf,'Nuc\',method,'\'];
save_path = [base_path,'Results\',lmh_nuf,'Nuc\'];
label_path = [base_path,'\masks\masks\'];

rec_files = dir([rec_path '*.mat']);
rec_num = length(rec_files);

QE=cell(1,1);
for nn = 1:rec_num-1
    rec_file = rec_files(nn);
    rec_name = rec_file.name;
    if strcmp(rec_name,'Misc_83.mat')
        continue;
    end
%     rec_name = 'Misc_319.mat';
    
    rec_mat = [rec_path,rec_name];
    rec_data = load(rec_mat);
%     rec=rec_data.Rec(:,:);
    rec=rec_data.pre(:,:);
%     rec_png = [rec_path,rec_name(1:end-4),'.png'];

    ori_mat = [ori_path rec_name];
    ori_data = load(ori_mat);
    ori=ori_data.Ori(:,:);
%     rec=ori_data.Nuf(:,:);
    
    h = size(ori, 1);
    w = size(ori, 2);
    hh = size(rec, 1);
    ww = size(rec, 2);
    if h==hh && w==ww
        rec = rec;
    else
        rec = imresize(rec, [h,w]);
        1
    end
    
%     lab_xml = [label_path,rec_name(1:end-4)];
    lab_mask = xml2mat(label_path,rec_name(1:end-4));
    target = lab_mask(1,:);
    pw = 5;%3,10,1,2,5
    data_max = 255;
    lur = target(2); %目标的左上角的行数(第二坐标)
    luc = target(1); %目标的左上角的列数(第一坐标)
    rdr = target(4); %目标的右下角的行数(第二坐标)
    rdc = target(3); %目标的右下角的列数(第一坐标)
    if lur-pw<1
        continue;
    end
    if luc-pw<1
        continue;
    end
    if rdr+pw>h
        continue;
    end
    if rdc+pw>h
        continue;
    end     
   
    ori_s = double(ori)/255;
    rec_s = double(rec)/255;
    %% 结果输出
    IR = coarseness(rec);%图像粗糙度
    Ln = ln(rec);
    [SCR, SCRG] = scrg(rec, rec, pw, lur, luc, rdr, rdc);%信杂比，信杂比增益
    [RMSE,PSNR] = psnr(ori,rec,data_max);%均方误差根，峰值信噪比
    SSIM = ssim(ori_s,rec_s);%结构相似度，归一化/255
%     disp([num2str(RMSE),';',num2str(PSNR),';',num2str(SSIM),';',num2str(IR),';',num2str(SCR)]);    
    QE{1}=[QE{1};RMSE,PSNR,SSIM,IR,Ln,SCR];        
end
xls_name = [save_path,method,'_RMSE_PSNR_SSIM_IR_Ln_SCR.xlsx'];
xlswrite(xls_name,QE{1},1,'A1');
RMSE_m = mean(mean(QE{1}(:,1)));
PSNR_m = mean(mean(QE{1}(:,2)));
SSIM_m = mean(mean(QE{1}(:,3)));
IR_m = mean(mean(QE{1}(:,4)));
Ln_m = mean(mean(QE{1}(:,5)));
SCR_m = mean(mean(QE{1}(:,6)));
disp([num2str(RMSE_m),';',num2str(PSNR_m),';',num2str(SSIM_m),';',num2str(IR_m),';',num2str(Ln_m),';',num2str(SCR_m)]);  