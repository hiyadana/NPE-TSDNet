close all;
clear all;
clc;

recPath = '..\dataset\RealIRresults\Test_Pre_crop_1.png';
nufPath = '..\dataset\RealIR\crop_1.png';

% rec = imread(recPath);
% rec = double(rec);
nuf = imread(nufPath);
if size(nuf, 3) == 3
    nuf = rgb2gray(nuf);
end
nuf = double(nuf);

rec=nuf;

h = size(nuf, 1);
w = size(nuf, 2);
hh = size(rec, 1);
ww = size(rec, 2);
if h==hh && w==ww
    rec = rec;
else
    rec = imresize(rec, [h,w]);
    1
end

%crop1
jy_1 = rec(1:50, 1:155);
jy_2 = rec(1:85,236:310);
jy_3 = rec(1:30,366:480);
jy_4 = rec(91:200,1:155);
jy_5 = rec(116:200,216:400);
jy_6 = rec(301:400,1:480);
jy = [jy_1(:);jy_2(:);jy_3(:);jy_4(:);jy_5(:);jy_6(:)];
jy = double(jy);
ICV = mean(jy(:))/std(jy(:))

rec(1:50, 1:155)=-1000;
rec(1:85,236:310)=-1000;
rec(1:30,366:480)=-1000;
rec(91:200,1:155)=-1000;
rec(116:200,216:400)=-1000;
rec(301:400,1:480)=-1000;

%crop2
% jy_1 = rec(1:210, 1:40);
% jy_2 = rec(1:210,146:160);
% jy = [jy_1(:);jy_2(:)];
% jy = double(jy);
% ICV = mean(jy(:))/std(jy(:))
% 
% nuf(1:210, 1:40)=0;
% nuf(1:210,146:160)=0;

% mask = find(nuf~=0);
mask = find(rec~=-1000);

r_n = abs(rec-nuf);
mm = r_n(mask)./nuf(mask);
MRD = mean(mean(mm))

zh = ICV/MRD


